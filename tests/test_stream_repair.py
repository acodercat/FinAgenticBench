"""Tests for the DSML fence repair (core.stream_repair).

Each case is a real shape from the 2026-08-22 discovery round, fed through
the stream response with a fake provider so the rewrite is exercised exactly
as cave-agent would see it — token by token, including tags split across
deltas, and the end-of-stream rule.
"""

import asyncio
from types import SimpleNamespace

import pytest

from core.stream_repair import (
    FenceRepairingModel,
    _could_become_tag,
    _FenceState,
    _opens_code,
)


def _chunk(content: str | None, finish: str | None = None):
    delta = SimpleNamespace(content=content, reasoning_content=None, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


class _FakeIterator:
    """Yields chunks; a chunk that is an exception INSTANCE is raised instead.

    Raised from ``__anext__`` on purpose. An error thrown while reading a
    chunk's attributes lands in a different handler that never retries, so a
    test that wants the retry path has to fail the iterator itself.
    """

    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None
        if isinstance(item, BaseException):
            raise item
        return item


def _collect(pieces: list[str], finish: str = "stop") -> str:
    """Stream ``pieces`` as content deltas, the last carrying ``finish``, and
    return the concatenated content cave-agent's parser would see."""
    model = FenceRepairingModel.__new__(FenceRepairingModel)
    stream = model.stream([])
    chunks = [_chunk(p) for p in pieces[:-1]] + [_chunk(pieces[-1], finish)]

    async def open_stream():
        return _FakeIterator(chunks)

    stream._open_stream = open_stream

    async def drain():
        out = []
        async for delta in stream:
            if delta.content:
                out.append(delta.content)
        return "".join(out)

    return asyncio.run(drain())


def _repairs(caplog) -> list[str]:
    """The repairs this module logged, in order.

    The WARNING log is the only record it keeps — there is no attribute to
    inspect, because nothing in production would read one.
    """
    return [
        r.getMessage().removeprefix("stream repair: ")
        for r in caplog.records
        if r.name == "core.stream_repair"
    ]


# -- unit: the predicates --------------------------------------------------


def test_opens_code_names_only_code_tags():
    assert _opens_code("python")
    assert _opens_code("tool")
    assert _opens_code('parameter name="code" string="true"')
    assert not _opens_code("")
    assert not _opens_code("tool_calls")
    assert not _opens_code('invoke name="code"')
    assert not _opens_code('parameter name="language"')


def test_could_become_tag_accepts_every_prefix_and_rejects_plain_html():
    for n in range(1, len("<｜DSML｜python")):
        assert _could_become_tag("<｜DSML｜python"[:n]), n
    assert _could_become_tag("</｜DSML")
    assert not _could_become_tag("<div")
    assert not _could_become_tag("<｜DSML｜python>")  # complete: match it instead
    assert not _could_become_tag("<= 75")


def test_fence_state_survives_a_split_fence():
    state = _FenceState()
    state.observe("``")
    assert not state.open
    state.observe("`python\n")
    assert state.open
    state.observe("x = 1\n`")
    state.observe("``")
    assert not state.open


# -- the fence state must agree with its consumer --------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("```python\nx = 1\n", True),        # the block this module exists to close
        ("```python\nx = 1\n```\n", False),  # closed already
        ("```\nx = 1\n", False),             # a bare fence opens nothing
        ("```sql\nSELECT 1\n", False),       # nor does another language
        ("wrap it in ```python fences\n", False),  # nor a mid-sentence mention
        ("   ```python\nx = 1\n", True),     # Markdown tolerates three spaces
        ("    ```python\nx = 1\n", False),   # and not a fourth
        ("````\n", False),                   # four backticks are not an opener
        ('```python\ns = """\n```\n"""\n', True),  # not a closer inside a string
    ],
)
def test_fence_state_matches_the_documented_markdown_rules(text, expected):
    """The state is cave-agent's own, and these are the rules it applies.

    Pinned as literals rather than against a freshly built parser, so the test
    would still fail if someone replaced the delegation with a reimplementation
    that got them wrong. Disagreement is silent and costly in one direction:
    believing a fence open when the parser does not makes this module skip the
    opening fence a DSML code tag needs, and the code never runs.
    """
    state = _FenceState()
    state.observe(text)
    assert state.open is expected


@pytest.mark.parametrize(
    "text", ["````", "```python\nk=1\n```", "a```b", "``````", "```python\nk=1\n"]
)
def test_fence_state_is_independent_of_chunk_boundaries(text):
    whole = _FenceState()
    whole.observe(text)
    for cut in range(len(text) + 1):
        split = _FenceState()
        split.observe(text[:cut])
        split.observe(text[cut:])
        assert split.open is whole.open, (text, cut)


# -- a retry replays the stream, so per-connection state must reset ---------


def test_retry_replay_does_not_carry_a_half_read_tag(monkeypatch):
    """A transport failure before any delta is returned replays the stream.

    The first attempt here ends on a partial tag, which this class holds back
    and reports as no output — so the base retries. If the held prefix
    survived into the replay it would be prepended to the new stream's first
    characters and the tag would never match.
    """
    monkeypatch.setattr("cave_agent.models.base.get_retry_delay", lambda *a: 0)
    attempts = []

    model = FenceRepairingModel.__new__(FenceRepairingModel)
    stream = model.stream([])

    async def open_stream():
        attempts.append(len(attempts))
        if len(attempts) == 1:
            # Ends mid-tag: rewritten to nothing, so no delta is returned.
            return _FakeIterator(
                [_chunk("<｜DSML"), ConnectionError("connection reset")]
            )
        return _FakeIterator(
            [
                _chunk("<｜DSML｜tool>"),
                _chunk("\nv = 1\n"),
                _chunk("</｜DSML｜tool>", "stop"),
            ]
        )

    # Patched on the BASE class, not on the instance: the reset under test
    # lives in this class's own `_open_stream`, and replacing the bound
    # attribute would skip it and quietly test nothing.
    monkeypatch.setattr(
        "cave_agent.models.litellm._LiteLLMStreamResponse._open_stream",
        lambda self: open_stream(),
    )

    async def drain():
        out = []
        async for delta in stream:
            if delta.content:
                out.append(delta.content)
        return "".join(out)

    text = asyncio.run(drain())
    assert attempts == [0, 1], "the first attempt should have been retried"
    assert text == "```python\n\nv = 1\n\n```\n"


# -- integration: corpus shapes through the stream -------------------------


def test_markdown_open_dsml_close_is_closed_with_backticks(caplog):
    text = _collect(["```python\n", "x = 1\n", "</｜DSML｜>"])
    assert text == "```python\nx = 1\n\n```\n"
    assert _repairs(caplog) == ["closed fence at </｜DSML｜>"]


def test_native_tool_block_becomes_a_python_fence(caplog):
    text = _collect(
        ["Let me check.\n\n", "<｜DSML｜tool>", "\nprint(1)\n", "</｜DSML｜tool>"]
    )
    assert text == "Let me check.\n\n```python\n\nprint(1)\n\n```\n"
    assert _repairs(caplog) == [
        "opened fence at <｜DSML｜tool>",
        "closed fence at </｜DSML｜tool>",
    ]


def test_full_invoke_envelope_keeps_only_the_code_parameter(caplog):
    pieces = [
        "<｜DSML｜tool_calls>",
        '<｜DSML｜invoke name="code">',
        '<｜DSML｜parameter name="code" string="true">',
        "y = 2\n",
        "</｜DSML｜parameter>",
        "</｜DSML｜invoke>",
        "</｜DSML｜tool_calls>",
    ]
    text = _collect(pieces)
    assert text == "```python\ny = 2\n\n```\n"
    # One open, one close; the envelope tags are dropped without comment.
    assert _repairs(caplog) == [
        'opened fence at <｜DSML｜parameter name="code" string="true">',
        "closed fence at </｜DSML｜parameter>",
    ]


def test_tag_split_across_deltas_is_still_one_tag():
    text = _collect(["```python\n", "z = 3\n", "</", "｜DS", "ML｜", ">"])
    assert text == "```python\nz = 3\n\n```\n"


def test_bare_tag_opens_nothing_and_vanishes(caplog):
    text = _collect(["<｜DSML｜>", "\nThe record runs to July.\n"])
    assert text == "\nThe record runs to July.\n"
    assert _repairs(caplog) == []


def test_bare_tag_before_a_proper_fence_does_not_double_open(caplog):
    text = _collect(["<｜DSML｜>", "\n```python\nq = 1\n```\n"])
    assert text == "\n```python\nq = 1\n```\n"
    assert _repairs(caplog) == []


def test_unclosed_fence_at_clean_stop_is_closed(caplog):
    text = _collect(
        ["```python\n", "attainment_days = 269\n", "print(attainment_days)"], finish="stop"
    )
    assert text.endswith("print(attainment_days)\n```\n")
    assert _repairs(caplog) == ["closed a fence left open at a clean stop"]


def test_unclosed_fence_at_length_cut_stays_open(caplog):
    text = _collect(["```python\n", "for i in range("], finish="length")
    assert text == "```python\nfor i in range("
    assert _repairs(caplog) == []


def test_closing_tag_with_no_open_fence_is_dropped(caplog):
    text = _collect(["done.\n", "</｜DSML｜tool>"])
    assert text == "done.\n"
    assert _repairs(caplog) == []


def test_plain_angle_brackets_pass_through():
    text = _collect(["if a < b and c <= d:\n", "    pass  # <not a tag>\n"])
    assert text == "if a < b and c <= d:\n    pass  # <not a tag>\n"


def test_held_prefix_that_never_completes_is_flushed_as_text():
    text = _collect(["see <｜DSML"], finish="stop")
    assert text == "see <｜DSML"


def test_proper_fences_are_left_alone(caplog):
    body = "```python\nk = 1\n```\nand then\n```python\nk = 2\n```\n"
    assert _collect([body]) == body
    assert _repairs(caplog) == []


@pytest.mark.parametrize("tag", ["<｜DSML｜python>", "<｜DSML｜tool>"])
def test_code_opener_inside_an_open_fence_is_dropped(tag, caplog):
    text = _collect(["```python\n", tag, "\nw = 1\n```\n"])
    assert text == "```python\n\nw = 1\n```\n"
    assert _repairs(caplog) == []


# -- the ASCII envelope the pro endpoint leaks (census of 2026-08-29) -------


def test_singular_tool_call_envelope_becomes_a_python_fence(caplog):
    # qwen3.8-flash wraps bare code in <tool_call>, singular, where a fence was
    # asked for; unread, the turn runs nothing and every output stays unset.
    text = _collect(["I'll load the table.\n\n<tool_", "call>\n",
                     "print(1)\n", "</tool_call>"])
    assert text == "I'll load the table.\n\n```python\n\nprint(1)\n\n```\n"
    assert _repairs(caplog) == [
        "opened fence at <tool_call>",
        "closed fence at </tool_call>",
    ]


def test_plural_tool_calls_wrapper_still_opens_nothing():
    assert "```" not in _collect(["<tool_calls>\nno code here\n</tool_calls>"])


def test_ascii_invoke_envelope_becomes_a_python_fence(caplog):
    pieces = ['<invoke name="exec">\n', '<parameter name="code">\n',
              "import pandas as pd\n", "</parameter>\n", "</invoke>"]
    text = _collect(pieces)
    assert text == "\n```python\n\nimport pandas as pd\n\n```\n\n"
    assert _repairs(caplog) == [
        'opened fence at <parameter name="code">',
        "closed fence at </parameter>",
    ]


def test_system_calls_wrapper_is_dropped_around_the_envelope():
    pieces = ["<system_calls>\n", '<invoke name="execute_python">\n',
              '<parameter name="code">\n', "x = 1\n", "</parameter>\n",
              "</invoke>\n", "</system_calls>"]
    assert _collect(pieces) == "\n\n```python\n\nx = 1\n\n```\n\n\n"


def test_bare_fence_inside_the_code_parameter_is_swallowed(caplog):
    # gate_c_recall_fix, beijing_aqi_primary_pollutant: the model fenced the
    # code AGAIN inside the parameter. Without the swallow, that fence line
    # closes the block the tag just opened and the code runs as prose.
    pieces = ['\n<invoke name="python">\n<parameter name="code">\n',
              "```\n", "# orient\nprint(1)\n", "```\n", "</parameter>\n</invoke>\n"]
    text = _collect(pieces)
    assert text == "\n\n```python\n\n# orient\nprint(1)\n```\n\n\n"
    assert _repairs(caplog) == [
        'opened fence at <parameter name="code">',
        "swallowed a fence line after an opening tag",
    ]


def test_tool_calls_before_a_proper_fence_and_a_stray_execution_output_close():
    # gate_c_o3seam: the fence is the model's own; only the envelope is wrong.
    pieces = ["<tool_calls>```python\n", "y = 2\n", "```</execution_output>"]
    assert _collect(pieces) == "```python\ny = 2\n```"


def test_ascii_tag_split_across_deltas_is_still_one_tag():
    pieces = ["<par", 'ameter name="co', 'de">\n', "z = 3\n", "</param", "eter>"]
    assert _collect(pieces) == "```python\n\nz = 3\n\n```\n"


@pytest.mark.parametrize("text", [
    "a < b and b > c",
    "<p>prose</p>",
    "<ins class=\"final_variable_metadata\">- name: x</ins>",
    "<parameters>",
    "<invokes the rule>",
])
def test_ascii_tags_outside_the_family_pass_through(text):
    assert _collect([text]) == text


def test_could_become_tag_accepts_ascii_prefixes():
    assert _could_become_tag("<inv")
    assert _could_become_tag('<parameter name="co')
    assert _could_become_tag("</system_ca")
    assert not _could_become_tag("<involves")
    assert not _could_become_tag("<br")


def test_html_code_block_becomes_a_python_fence(caplog):
    # gate_c_pushback_r14_2, pro: the whole turn was one <pre><code> block.
    pieces = ["<pre><code class=\"language-python\">", "from x import y\n", "</code></pre>"]
    assert _collect(pieces) == "```python\nfrom x import y\n\n```\n"
    assert _repairs(caplog) == [
        'opened fence at <code class="language-python">',
        "closed fence at </code>",
    ]


def test_code_tag_without_python_opens_nothing():
    assert _collect(["<code>x</code> in prose"]) == "x in prose"


def test_function_envelope_is_dropped_like_invoke():
    # gate_c_peak_1, pro: the whole turn was <function name="search_hk_location">Hong Kong</function>
    assert _collect(['<function name="search_hk_location">', "Hong Kong", "</function>"]) == "Hong Kong"


def test_assistant_wrapper_is_dropped_so_the_fence_inside_is_reached():
    # gate_c_wkend_1, pro: <assistant>```python ... ```</assistant> — nothing ran.
    assert _collect(["<assistant>```python\n", "x = 1\n", "```</assistant>"]) == "```python\nx = 1\n```"


def test_bare_python_tag_becomes_a_python_fence(caplog):
    # gate_c_split, pro (hk_weather_variable_coverage T2, 2026-09-02): the
    # whole turn was <python> … </python> — the DSML tag with its marker gone.
    pieces = ["<python>\n", "from sqlalchemy import text\n", "</python>"]
    assert _collect(pieces) == "```python\n\nfrom sqlalchemy import text\n\n```\n"
    assert _repairs(caplog) == ["opened fence at <python>", "closed fence at </python>"]


def test_stray_angle_before_a_fence_at_line_start_is_dropped(caplog):
    # gate_c_split, pro (hk_weekend_ozone_ox_conservation T1, 2026-09-02):
    # the turn opened with <```python — a fence Markdown cannot see.
    pieces = ["<", "``", "`python\nx = 1\n```\n"]
    assert _collect(pieces) == "```python\nx = 1\n```\n"
    assert _repairs(caplog) == ["dropped a stray < before a fence"]


def test_angle_before_a_fence_mid_line_is_text():
    assert _collect(["see <```python", " in prose"]) == "see <```python in prose"


def test_cdata_section_becomes_a_python_fence(caplog):
    # gate_c_r22, pro (beijing_exceedance_exposure_bias T2, 2026-09-03): the
    # whole turn was a CDATA section, so no code ran and nothing was stored.
    pieces = ["\n<![CDATA[\n", "import pandas as pd\n", "]]>"]
    assert _collect(pieces) == "\n```python\n\nimport pandas as pd\n\n```\n"
    assert _repairs(caplog) == ["opened fence at <![CDATA[", "closed fence at ]]>"]


def test_cdata_close_without_an_open_fence_is_dropped():
    assert _collect(["plain text ]]> more"]) == "plain text  more"


def test_cdata_close_split_across_deltas_is_still_recognised(caplog):
    # The closer arriving in pieces must not leak into the code block: emitted
    # as text it would put "]]>" inside the Python and leave the fence to the
    # clean-stop fallback.
    assert _collect(["<![CDATA[\nx = 1\n", "]", "]>"]) == "```python\n\nx = 1\n\n```\n"
    assert _repairs(caplog) == ["opened fence at <![CDATA[", "closed fence at ]]>"]


def test_a_closing_bracket_in_ordinary_code_survives():
    assert _collect(["a = [1]"]) == "a = [1]"
    assert _collect(["a = [1]", "\n"]) == "a = [1]\n"


# -- the four shapes FinAgenticBench actually lost turns to -------------------------
#
# Each opening below is copied from a stored run under `experiments/` where
# `stop_reason == "completed"` and `executed_code == False`: the turn produced
# a response, ran nothing, and set no variable. Ten such turns exist, all on
# deepseek-v4-pro-0813. Nine are these three shapes; the tenth is covered by
# `test_a_fabricated_system_warning_is_not_repairable` below.


def test_finbench_html_code_block_opens_a_fence(caplog):
    """`apple_commitment_curve_discount_stress` T2/T3/T4 and two more, 5 turns."""
    out = _collect([
        '<pre><code class="language-python">',
        "import pandas as pd\n",
        "print(apple[['adsh','form']].drop_duplicates())\n",
        "</code></pre>",
    ])
    assert out.startswith("```python\n")
    assert out.rstrip().endswith("```")
    assert "import pandas as pd" in out
    assert "<pre>" not in out and "<code" not in out


def test_finbench_invoke_execute_python_keeps_only_the_code(caplog):
    """`disaster_credit_union_delinquency_concentration` T3/T4 and one more, 3 turns.

    The three stored turns name the tool two different ways -- `execute_python`
    twice and `exec_python` once -- so both are exercised here. The scanner
    matches on the tag head, which is why the attribute does not matter.
    """
    for tool in ('execute_python', 'exec_python'):
        out = _collect([
            f'<invoke name="{tool}">',
            '<parameter name="code">',
            "import pandas as pd\n",
            "</parameter>",
            "</invoke>",
        ])
        assert "```python\n" in out and "import pandas as pd" in out
        assert tool not in out
    out = _collect([
        '<invoke name="execute_python">',
        '<parameter name="code">',
        "import pandas as pd\n",
        "df = ncua_df\n",
        "</parameter>",
        "</invoke>",
    ])
    assert "```python\n" in out
    assert "import pandas as pd" in out and "df = ncua_df" in out
    assert "invoke" not in out and "parameter" not in out


def test_finbench_stray_bracket_before_a_fence(caplog):
    """`reg_cf_amendment_reg_d_market_context` T3, 1 turn."""
    out = _collect([
        "<```python\n",
        "dfd = sec_form_d_filings_df\n",
        "print(dfd.shape)\n",
        "```",
    ])
    assert out.startswith("```python\n")
    assert "<```" not in out


def test_a_fabricated_system_warning_is_not_repairable(caplog):
    """`insider_option_exercise_intrinsic_capture` T2, the tenth turn.

    The model invented a system diagnostic instead of running code. There is no
    code inside it, so no fence can be opened around one: this module must pass
    it through unchanged rather than manufacture a block. The turn stays lost,
    and the failure taxonomy — not this module — is where it belongs.
    """
    text = (
        "<system-warning>Potential data error detected: The comparison of the "
        "'trans_date' Series with '2024-01-01' failed because the column is "
        "stored as a string; ensure date parsing is applied before filtering."
        "</system-warning>"
    )
    assert _collect([text]) == text
    assert _repairs(caplog) == []
