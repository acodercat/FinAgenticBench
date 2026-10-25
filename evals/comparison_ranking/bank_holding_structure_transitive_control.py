"""Trace direct controlled relationships through the current FFIEC NIC snapshot.

The FFIEC relationship table mixes relationship grains and control statuses. This
case builds its graph only from rows whose relationship level is 1 (direct) and
whose control indicator is 1 (controlled). The full source contains 35,922 rows
and three repeated parent-offspring pairs. After the graph filter, 34,159 entities
appear as endpoints and 3,599 appear as parents but never offspring. Four strongly
connected groups contain nine entities, so traversal still needs a visited set.

The median top-level entity has 1 direct controlled offspring and 2 reachable
entities, and 38.1773 percent reach beyond their direct offspring. Morgan Stanley
has the largest transitive reach: 73 direct offspring, 4,129 reachable entities, a
ratio of 56.5616, and a maximum shortest-path depth of 11. Goldman Sachs ranks
second with 3,690 reachable entities and Truist third with 2,313.

Legal-entity reach is not a balance-sheet size measure. Among the descendants of
Morgan Stanley, Goldman Sachs and Truist, respectively 2, 2 and 1 carry a nonmissing
FDIC certificate in the identifier table. Certificate presence is an operational
linkage flag here; the available identifier table does not establish current deposit
insurance status for every linked entity.

The source relationship level is not the graph depth derived from a selected top.
It reads 1 on 35,868 rows and takes another value on 54; non-1 rows already describe
indirect relationships and therefore are not one-step graph edges. Using shortest
distance on the direct controlled graph, the leader is widest at level 4 with 2,090
entities, has 3 at its deepest level, and reaches 20 identifiers absent from the
institution table.

Measured rejected alternatives include treating
every control status as controlled, treating indirect relationships as direct
edges, ranking by direct offspring, including the root, and shifting the depth
origin. The query pins the graph grain and control code, so these are baseline
instruction-following probes rather than a hidden-method trap. This is a hard
baseline case.

Boundaries: the file is a current snapshot, so the results do not date a structure
or survive a reorganization. A top-level entity here means a parent that never
appears as an offspring in the filtered snapshot, not an ultimate owner in an
absolute legal sense. Entities reachable from more than one top-level parent are
counted under each, so descendant counts across parents do not add.

Numeric validation uses the shared 0.6 x 10^-N rounding-boundary tolerance for
N requested decimals; counts and entity names match exactly.
"""

import collections
from functools import lru_cache

from cave_agent import Variable

from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs


PARENT_COLUMN = "parent_rssd_id"
OFFSPRING_COLUMN = "offspring_rssd_id"
LEVEL_COLUMN = "relationship_level"
DIRECT_LEVEL = "1"
CONTROLLED = "1"


def _v(name, description):
    return Variable(name, None, description)


TURN_1_NAMES = ["edge_rows", "duplicate_pairs", "entity_count", "top_level_count",
                "cyclic_groups", "entities_on_cycles"]
TURN_2_NAMES = ["leader_name", "leader_direct_offspring", "leader_descendants", "leader_ratio",
                "leader_depth", "median_direct_offspring", "median_descendants", "deeper_structure_share_pct"]
TURN_3_NAMES = ["leader_fdic_linked", "second_name", "second_descendants", "second_fdic_linked",
                "third_name", "third_descendants", "third_fdic_linked"]
TURN_4_NAMES = ["base_level_rows", "other_level_rows", "widest_level", "widest_level_count",
                "deepest_level_count", "unlisted_descendants"]

variables = [
    _v(TURN_1_NAMES[0], "Store how many relationship rows the file carries as an integer."),
    _v(TURN_1_NAMES[1], "Store how many rows repeat a parent-and-offspring pair already present as an integer."),
    _v(TURN_1_NAMES[2], "Store how many distinct entities appear in the pairs as an integer."),
    _v(TURN_1_NAMES[3], "Store how many appear only as a parent as an integer."),
    _v(TURN_1_NAMES[4], "Store how many groups of mutually reachable entities have more than one member as an integer."),
    _v(TURN_1_NAMES[5], "Store how many entities those groups hold in total as an integer."),
    _v(TURN_2_NAMES[0], "Store the name of the top-level entity controlling the most entities as text."),
    _v(TURN_2_NAMES[1], "Store how many direct offspring it reports as an integer."),
    _v(TURN_2_NAMES[2], "Store how many entities it controls in total as an integer."),
    _v(TURN_2_NAMES[3], "Store the second divided by the first, rounded to 4 decimals."),
    _v(TURN_2_NAMES[4], "Store how many levels deep its structure runs as an integer."),
    _v(TURN_2_NAMES[5], "Store the median direct offspring count across top-level entities, rounded to 4 decimals."),
    _v(TURN_2_NAMES[6], "Store the median total descendant count across top-level entities, rounded to 4 decimals."),
    _v(TURN_2_NAMES[7], "Store the percent of top-level entities controlling more than their direct offspring, rounded to 4 decimals."),
    _v(TURN_3_NAMES[0], "Store how many of the leader's descendants carry a nonmissing FDIC certificate as an integer."),
    _v(TURN_3_NAMES[1], "Store the name of the second-ranked top-level entity as text."),
    _v(TURN_3_NAMES[2], "Store how many entities it controls as an integer."),
    _v(TURN_3_NAMES[3], "Store how many of those carry a nonmissing FDIC certificate as an integer."),
    _v(TURN_3_NAMES[4], "Store the name of the third-ranked top-level entity as text."),
    _v(TURN_3_NAMES[5], "Store how many entities it controls as an integer."),
    _v(TURN_3_NAMES[6], "Store how many of those carry a nonmissing FDIC certificate as an integer."),
    _v(TURN_4_NAMES[0], "Store how many rows carry the base relationship level as an integer."),
    _v(TURN_4_NAMES[1], "Store how many carry any other value as an integer."),
    _v(TURN_4_NAMES[2], "Store the derived level that holds the most of the leader's descendants as an integer."),
    _v(TURN_4_NAMES[3], "Store how many descendants sit at that level as an integer."),
    _v(TURN_4_NAMES[4], "Store how many sit at the deepest level as an integer."),
    _v(TURN_4_NAMES[5], "Store how many of the leader's descendants appear in no institution record as an integer."),
]

DECIMALS = [0, 0, 0, 0, 0, 0,
            None, 0, 0, 4, 0, 4, 4, 4,
            0, None, 0, 0, None, 0, 0,
            0, 0, 0, 0, 0, 0]


def _normalized_name(value):
    return " ".join(str(value).split()).casefold() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def _graph():
    frame = load_expansion_table("ffiec_nic_ownership").copy()
    frame["parent"] = frame[PARENT_COLUMN].astype(str)
    frame["offspring"] = frame[OFFSPRING_COLUMN].astype(str)
    all_pairs = frame.drop_duplicates(["parent", "offspring"])
    graph_rows = frame.loc[
        frame[LEVEL_COLUMN].astype(str).eq(DIRECT_LEVEL)
        & frame.control_indicator.astype(str).eq(CONTROLLED)
    ]
    pairs = graph_rows.drop_duplicates(["parent", "offspring"])
    adjacency = collections.defaultdict(set)
    for parent, offspring in zip(pairs.parent, pairs.offspring):
        adjacency[parent].add(offspring)
    return frame, all_pairs, pairs, adjacency


@lru_cache(maxsize=1)
def _institutions():
    frame = load_expansion_table("ffiec_nic_institutions").copy()
    frame["rssd"] = frame.rssd_id.astype(str)
    names = dict(zip(frame.rssd, frame.legal_name.astype(str)))
    insured = set(frame.loc[frame.fdic_certificate.notna(), "rssd"])
    return names, insured


def _descendants(adjacency, root):
    depths = {root: 0}
    queue = collections.deque([root])
    while queue:
        node = queue.popleft()
        for child in adjacency.get(node, ()):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    depths.pop(root)
    return depths


def _cyclic_groups(adjacency, nodes):
    """Tarjan's strongly connected components, iteratively; returns those with more than one member."""
    index, low, on_stack, stack, counter, groups = {}, {}, set(), [], [0], []
    for start in nodes:
        if start in index:
            continue
        work = [(start, 0)]
        while work:
            node, position = work[-1]
            if position == 0:
                index[node] = low[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                on_stack.add(node)
            recursed = False
            children = sorted(adjacency.get(node, ()))
            for offset in range(position, len(children)):
                child = children[offset]
                if child not in index:
                    work[-1] = (node, offset + 1)
                    work.append((child, 0))
                    recursed = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if recursed:
                continue
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    groups.append(component)
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
    return groups


@lru_cache(maxsize=1)
def ground_truth():
    frame, all_pairs, pairs, adjacency = _graph()
    names, insured = _institutions()
    entities = set(pairs.parent) | set(pairs.offspring)
    tops = sorted(set(pairs.parent) - set(pairs.offspring))
    groups = _cyclic_groups(adjacency, sorted(entities))

    profiles = []
    for top in tops:
        depths = _descendants(adjacency, top)
        profiles.append((top, len(adjacency[top]), len(depths), max(depths.values()) if depths else 0))
    ranked = sorted(profiles, key=lambda row: (-row[2], row[0]))
    leader, second, third = ranked[0], ranked[1], ranked[2]

    direct_counts = [row[1] for row in profiles]
    totals = [row[2] for row in profiles]
    deeper = sum(1 for row in profiles if row[2] > row[1])

    leader_depths = _descendants(adjacency, leader[0])
    per_level = collections.Counter(leader_depths.values())
    widest_level = max(sorted(per_level), key=lambda level: per_level[level])
    deepest = max(per_level)

    def insured_count(root):
        return sum(1 for node in _descendants(adjacency, root) if node in insured)

    levels = frame[LEVEL_COLUMN].astype(str)

    return (
        int(len(frame)), int(len(frame) - len(all_pairs)), int(len(entities)), int(len(tops)),
        int(len(groups)), int(sum(len(group) for group in groups)),
        names.get(leader[0], leader[0]), int(leader[1]), int(leader[2]),
        leader[2] / leader[1], int(leader[3]),
        float(_median(direct_counts)), float(_median(totals)), deeper / len(profiles) * 100,
        insured_count(leader[0]),
        names.get(second[0], second[0]), int(second[2]), insured_count(second[0]),
        names.get(third[0], third[0]), int(third[2]), insured_count(third[0]),
        int(levels.eq(DIRECT_LEVEL).sum()), int(levels.ne(DIRECT_LEVEL).sum()),
        int(widest_level), int(per_level[widest_level]), int(per_level[deepest]),
        int(sum(1 for node in leader_depths if node not in names)),
    )


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


NAME_OUTPUTS = ("leader_name", "second_name", "third_name")


def _validate_subset(outputs, names):
    by_name = {variable.name: variable for variable in variables}
    truth = dict(zip(by_name, ground_truth()))
    decimals = dict(zip(by_name, DECIMALS))
    normalized = dict(outputs)
    for name in NAME_OUTPUTS:
        if name in normalized:
            normalized[name] = _normalized_name(normalized[name])
            truth[name] = _normalized_name(truth[name])
    return validate_ordered_outputs(
        normalized, [by_name[name] for name in names], [truth[name] for name in names],
        [decimals[name] for name in names],
    )


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [variable.name for variable in variables])


validators = {
    "validate_graph_shape": turn_validator(validate_turn_1),
    "validate_transitive_reach": turn_validator(validate_turn_2),
    "validate_fdic_linkage": turn_validator(validate_turn_3),
    "validate_derived_depth": turn_validator(validate_turn_4),
}
