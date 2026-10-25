"""Repair a model's code fences in the stream, before cave-agent parses them.

DeepSeek V4 answers an agent prompt with code written in its NATIVE tool
syntax some of the time — ``<｜DSML｜tool>`` … ``</｜DSML｜tool>``,
``<｜DSML｜python>`` … ``</｜DSML｜>``, or a markdown fence that it opens with
```` ```python ```` and then closes with a DSML tag instead of three backticks.
A serving stack that parses DSML strips the tags and leaves an unclosed
fence; one that does not parse it passes the tags straight through. Either
way cave-agent's parser, which knows only markdown fences, reaches the end of
the stream with a block it cannot call complete, and the agent — correctly,
for a stream that really was cut — refuses to execute an unfinished block.
The turn then ends with no code run and every variable unset.

Measured on an earlier benchmark run through the same harness, turns that
ended with no code run were 12 of 104 through one proxy and 8 of 104 on a
local sglang mirror against 1 of 45 on the hosted pro endpoint; with this module in place, 1 of 104 and 0 of 104. Counted
as failures they would have been the single largest category, and none of them
is a finding about the subject under test.

The FinAgenticBench measurement that motivated the port: of 953 stored turns across
``experiments/``, ten ended with ``stop_reason == "completed"`` and
``executed_code == False``, every one of them on ``deepseek-v4-pro-0813``
(2.6 per cent of that model's 389 turns). Nine carry an envelope this module
repairs -- five ``<pre><code class="language-python">``, three ``<invoke>``
envelopes with a ``<parameter name="code">`` body (twice naming
``execute_python`` and once ``exec_python``, which is why the scanner matches
the tag head and not the attribute), and one fence with a stray ``<`` in front
of it. The tenth is NOT in this family and
this module cannot help it: on `insider_option_exercise_intrinsic_capture` the
model emitted a fabricated ``<system-warning>`` describing a dtype problem, with
no code inside it at all. That is a hallucinated system message rather than a
mangled envelope, and it belongs in the failure taxonomy as its own category.

This module rewrites the content deltas as they stream:

- a DSML tag that names code (``python``, ``tool``, or a ``parameter`` called
  ``code``) opens a python fence if none is open;
- any DSML closing tag closes the fence if one is open;
- every other DSML tag is dropped — the bare ``<｜DSML｜>`` in particular is
  ambiguous (it precedes code, prose, and proper fences in the corpus) and
  opens nothing;
- when the provider ends the stream with ``finish_reason == "stop"`` and a
  fence is still open, the fence is closed. A ``length`` cut is left open:
  that block really is truncated, and cave-agent's refusal to run it stands.

The same model on the hosted pro endpoint leaks the envelope in ASCII
instead: ``<invoke name="exec">`` / ``<parameter name="code">`` … 
``</parameter>`` / ``</invoke>``, sometimes wrapped in ``<tool_calls>``,
``<system_calls>`` or ``<execution_tool>``, sometimes closed with a stray
``</execution_output>``, and once with a bare markdown fence INSIDE the code
parameter. Eight such responses were on disk by 2026-08-29, all pro, each
recorded as ``stop_reason: completed`` with ``errors: 0`` and no variable
set; on one case pro leaked on three attempts of five. Those ASCII tags are
handled by the same three rules — a bare fence line immediately after a
tag-opened fence is swallowed so it cannot close the block it was meant to
open — and every other ``<…>`` is passed through untouched. The HTML form,
``<pre><code class="language-python">`` … ``</code></pre>``, is in the same
family: the ``code`` tag opens when it names python, and ``</code>`` closes.
A CDATA section is handled too: ``<![CDATA[`` … ``]]>`` wraps real, runnable
code — the pro endpoint produced one on 2026-09-02 and another on 2026-09-03
(`beijing_exceedance_exposure_bias`), both times losing the turn — so it opens
a python fence and ``]]>`` closes it. It is the one envelope in this family
that is not an XML tag, so it is matched before the tag scanner runs.

So is the bare ``<python>`` … ``</python>`` the pro endpoint produced on
2026-09-02 (`hk_weather_variable_coverage`, Gate C of the split cases): the
DSML ``python`` tag with its marker stripped, handled by the same rule. The
same day's `hk_weekend_ozone_ox_conservation` turn opened with ``<```python``
— a proper fence with one stray ``<`` in front of it, which Markdown does not
read as a fence — so a ``<`` that stands at the head of a line immediately
before a fence is dropped. Nowhere else: a ``<`` mid-line is text.

What it does not do: guess that unfenced text after a bare tag is code. A
block the model never marked as code still does not run — that is today's
behaviour, and the residual is what a run reports as ``steps == 1`` with no
snippet. A ``<system_calls>`` holding prose and no code is that residual.

Every repair is logged at WARNING with the tag it acted on; counting those
lines in a run's output is how the figures above were measured.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cave_agent.models import LiteLLMModel
from cave_agent.models.base import StreamDelta
from cave_agent.models.litellm import _LiteLLMStreamResponse
from cave_agent.parsing.streaming import StreamingTextParser

logger = logging.getLogger(__name__)

# The ASCII envelope tags the pro endpoint leaks, plus the HTML code block
# (``<pre><code class="language-python">``) it produced once on 2026-08-29.
# Whole words only: ``<ins`` and ``<parameters`` are not in the family, and
# ``<p>`` in prose is not held.
_ASCII_HEADS = ("invoke", "parameter", "tool_calls", "tool_call", "system_calls",
                "execution_tool", "execution_output", "pre", "code", "function",
                "assistant", "python")
# The full tag, once a ``>`` has arrived. Attributes are free text up to the
# first ``>``; the corpus shows ``name="code"`` and ``string="true"``. The
# body of a DSML tag is what follows the marker; the body of an ASCII tag is
# the whole of it, so ``parameter name="code"`` reads the same either way.
_TAG = re.compile(
    r"<(/?)(?:｜DSML｜|(?=(?:%s)\b))([^>]*)>" % "|".join(_ASCII_HEADS))
# Text from a ``<`` that could still grow into a tag is held back until it
# either completes or proves to be something else. The cap is arbitrary and
# generous — the longest tag observed is 40 characters — and exists only so a
# ``<｜DSML｜`` whose ``>`` never arrives cannot buffer the whole answer.
_FENCE = "```"
_CDATA_OPEN = "<![CDATA["
_CDATA_CLOSE = "]]>"
_TAG_BODY_PREFIXES = ("｜DSML｜",) + _ASCII_HEADS + (_FENCE, _CDATA_OPEN[1:])
_MAX_HELD = 200


def _opens_code(body: str) -> bool:
    """Whether an OPENING tag's body names a code block."""
    head = body.split(None, 1)[0] if body else ""
    # ``tool_call`` is qwen3.8-flash's: asked for a fenced block it writes the code
    # bare between ``<tool_call>`` and ``</tool_call>``, and unread it runs nothing.
    if head in ("python", "tool", "tool_call"):
        return True
    if head == "code":
        return "language-python" in body
    return head == "parameter" and 'name="code"' in body


class _FenceState:
    """Whether cave-agent's parser currently holds an open code block.

    It answers by running that parser over the text this module emits, rather
    than by modelling it. The rules are not a backtick count — a fence opens
    only on ```` ```python ```` at the head of a line under Markdown's
    three-space tolerance, a closing fence inside a triple-quoted string does
    not close anything, and an empty block is handed back as text — and the
    parser's own source warns that two copies of the line-anchoring rule
    "can drift apart". A copy here would be the second one, and it would drift
    silently: believing a fence open when the parser does not makes this
    module skip the opening fence a DSML code tag needs, and the code never
    runs.

    Do not "simplify" this back to a backtick count: the version that did
    disagreed with the parser on inline mentions, on other languages, and on
    where the chunk boundaries happened to fall.
    """

    def __init__(self) -> None:
        self._parser = StreamingTextParser()

    @property
    def open(self) -> bool:
        return self._parser.in_code_block

    def observe(self, text: str) -> None:
        # Segments are the parser's product for its own caller; here only the
        # state it accumulates matters.
        self._parser.process_chunk(text)


def _could_become_tag(fragment: str) -> bool:
    """Whether ``fragment`` (starting at a ``<``) may still grow into a DSML
    tag once more of the stream arrives."""
    if ">" in fragment:
        return False
    body = fragment[2:] if fragment.startswith("</") else fragment[1:]
    if body.startswith(_FENCE):
        return False  # decided: a fence with a stray ``<`` before it
    return any(prefix.startswith(body[: len(prefix)]) for prefix in _TAG_BODY_PREFIXES)


class _RepairingStreamResponse(_LiteLLMStreamResponse):
    def __init__(self, model: LiteLLMModel, messages: list[dict[str, str]]):
        super().__init__(model, messages)
        self._reset_for_connection()

    def _reset_for_connection(self) -> None:
        """Discard everything learned from the text of one connection.

        ``StreamResponse`` retries a transport failure that happens before any
        semantic output, and a retry REPLAYS the stream from its first token.
        This class can return ``None`` for a chunk that held nothing but a
        dropped tag, and the base counts output by deltas returned — so a
        connection can die with this object's parser and held prefix already
        advanced while the base still believes nothing arrived. Carrying that
        state into the replay would prepend a half-read tag to the new stream's
        first characters.
        """
        self._fence = _FenceState()
        self._held = ""  # a possible partial tag, not yet decidable
        self._closed_at_end = False
        # True from a tag-opened fence until the first non-blank text after
        # it: a bare fence line there would close the block just opened.
        self._swallow_fence = False
        self._at_line_start = True

    async def _open_stream(self):
        # Called once per connection ATTEMPT, which is where a replay begins.
        self._reset_for_connection()
        return await super()._open_stream()

    # -- delta rewriting -------------------------------------------------

    def _process_stream_chunk(self, chunk: Any) -> StreamDelta | None:
        delta = super()._process_stream_chunk(chunk)
        if delta is None or not delta.content:
            return delta
        delta.content = self._rewrite(delta.content)
        if not delta.content and not delta.thinking:
            # The delta was a tag and nothing else: nothing for the parser.
            # ``__anext__`` moves on to the next chunk when given None.
            return None
        return delta

    def _emit(self, out: list[str], text: str) -> None:
        if self._swallow_fence and text.strip():
            self._swallow_fence = False
            if text.lstrip().startswith(_FENCE):
                line_end = text.find("\n", text.find(_FENCE))
                text = "" if line_end < 0 else text[line_end + 1:]
                logger.warning("stream repair: swallowed a fence line after an opening tag")
        if text:
            out.append(text)
            self._fence.observe(text)
            self._at_line_start = text.endswith("\n")

    def _rewrite(self, text: str) -> str:
        buf = self._held + text
        self._held = ""
        out: list[str] = []
        pos = 0
        while pos < len(buf):
            close = buf.find(_CDATA_CLOSE, pos)
            lt = buf.find("<", pos)
            if close >= 0 and (lt < 0 or close < lt):
                self._emit(out, buf[pos:close])
                if self._fence.open:
                    logger.warning("stream repair: closed fence at %s", _CDATA_CLOSE)
                    self._emit(out, "\n" + _FENCE + "\n")
                pos = close + len(_CDATA_CLOSE)
                continue
            if lt < 0:
                tail = buf[pos:]
                # A trailing "]" or "]]" may be the front of a CDATA close that
                # has not arrived yet. Hold it, the way a half-read tag is held:
                # emitted now it would land inside the code block as text, and
                # the fence would be closed only by the clean-stop fallback.
                for width in (len(_CDATA_CLOSE) - 1, 1):
                    if tail.endswith(_CDATA_CLOSE[:width]):
                        self._held, tail = tail[-width:], tail[:-width]
                        break
                self._emit(out, tail)
                break
            self._emit(out, buf[pos:lt])
            rest = buf[lt:]
            if rest.startswith(_CDATA_OPEN):
                if not self._fence.open:
                    logger.warning("stream repair: opened fence at %s", _CDATA_OPEN)
                    self._emit(out, _FENCE + "python\n")
                    self._swallow_fence = True
                pos = lt + len(_CDATA_OPEN)
                continue
            match = _TAG.match(rest)
            if match:
                closing, body = match.group(1) == "/", match.group(2)
                replacement = self._replace_tag(closing, body, match.group(0))
                self._emit(out, replacement)
                if replacement.startswith(_FENCE + "python"):
                    self._swallow_fence = True
                pos = lt + match.end()
            elif _could_become_tag(rest) and len(rest) < _MAX_HELD:
                self._held = rest
                break
            elif rest.startswith("<" + _FENCE) and self._at_line_start:
                logger.warning("stream repair: dropped a stray < before a fence")
                pos = lt + 1
            else:
                self._emit(out, "<")
                pos = lt + 1
        return "".join(out)

    def _replace_tag(self, closing: bool, body: str, raw: str) -> str:
        if closing:
            if not self._fence.open:
                return ""
            logger.warning("stream repair: closed fence at %s", raw)
            return "\n" + _FENCE + "\n"
        if _opens_code(body) and not self._fence.open:
            logger.warning("stream repair: opened fence at %s", raw)
            return _FENCE + "python\n"
        return ""

    # -- end of stream ---------------------------------------------------

    async def __anext__(self) -> StreamDelta:
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            if self._held:
                # A held prefix that never became a tag is ordinary text.
                text, self._held = self._held, ""
                self._fence.observe(text)
                return StreamDelta(content=text)
            # ``_closed_at_end`` is not belt-and-braces. A stream cut inside a
            # triple-quoted string leaves a block the parser will NEVER close:
            # its multiline-string guard swallows every ``` fed to it, so
            # ``open`` stays True and this branch would fire on every call
            # forever. Measured, not assumed.
            if self._fence.open and not self._closed_at_end and self.finish_reason == "stop":
                self._closed_at_end = True
                logger.warning("stream repair: closed a fence left open at a clean stop")
                closing = "\n" + _FENCE + "\n"
                self._fence.observe(closing)
                return StreamDelta(content=closing)
            raise


class FenceRepairingModel(LiteLLMModel):
    """:class:`LiteLLMModel` whose streams pass through the fence repair."""

    def stream(self, messages: list[dict[str, str]]) -> _RepairingStreamResponse:
        return _RepairingStreamResponse(self, messages)
