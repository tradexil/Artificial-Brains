"""Handcraft ~200 relational seed traces.

Relational traces encode abstract relationships between concepts:
  cause-effect, part-whole, similarity, sequence, hierarchy, etc.

Each trace activates neurons across:
  language (relational sub-region 114000–116999),
  pattern, integration, executive (reasoning), memory_long.
"""

from __future__ import annotations

import random

from brain.structures.trace_store import Trace, TraceStore
from brain.utils.config import REGIONS, SEED_RELATIONAL_TRACES

# Relational concepts
RELATIONAL_CONCEPTS = [
    # Cause-effect
    "cause", "effect", "result", "because", "therefore",
    "if_then", "trigger", "consequence", "lead_to", "produce",
    # Part-whole
    "part_of", "whole", "component", "element", "piece",
    "fragment", "section", "member", "belong", "contain_rel",
    # Comparison
    "same", "different", "similar", "opposite", "equal",
    "greater", "lesser", "more", "less", "most",
    # Temporal sequence
    "before", "after", "during", "first", "last",
    "next", "previous", "begin", "end", "sequence",
    # Spatial relation
    "on", "under", "over", "beside", "through",
    "around", "toward", "away", "along", "across",
    # Hierarchy
    "is_a", "kind_of", "type_of", "example", "category",
    "parent_of", "child_of", "ancestor", "descendant", "root",
    # Logical
    "and_rel", "or_rel", "not_rel", "implies", "excludes",
    "requires", "allows", "prevents", "enables", "blocks",
    # Quantitative
    "one", "many", "few", "all", "none",
    "some", "every", "each", "pair", "group",
    # Possession
    "has", "owns", "lacks", "gains", "loses",
    "gives", "takes", "shares", "keeps", "returns",
    # Agency
    "does", "makes", "creates", "destroys", "changes",
    "helps", "hurts", "uses", "needs", "wants",
    # Similarity and identity
    "like", "unlike", "matches", "differs", "resembles",
    "identical", "unique", "common", "rare", "typical",
    # Transformation
    "becomes", "transforms", "converts", "evolves", "develops",
    "grows", "shrinks", "merges", "splits", "combines",
    # Dependency
    "depends_on", "independent", "connected", "isolated", "related",
    "linked", "separate", "attached", "detached", "bound",
    # Condition
    "when", "unless", "while", "until", "only_if",
    "always", "never", "sometimes", "usually", "rarely",
    # Evaluation
    "good", "bad", "better", "worse", "best",
    "worst", "correct", "wrong", "true_val", "false_val",
]


def spawn_relational_traces(
    store: TraceStore,
    rng: random.Random | None = None,
    count: int | None = None,
) -> int:
    """Add relational seed traces to an existing TraceStore.

    Returns number of traces created.
    """
    if rng is None:
        rng = random.Random(600)
    if count is None:
        count = SEED_RELATIONAL_TRACES

    concepts = RELATIONAL_CONCEPTS[:count] if count <= len(RELATIONAL_CONCEPTS) else RELATIONAL_CONCEPTS
    while len(concepts) < count:
        concepts.append(f"relation_{len(concepts):03d}")

    created = 0
    for i, concept in enumerate(concepts[:count]):
        neurons: dict[str, list[int]] = {}

        # Language relational sub-region (114000–116999)
        neurons["language"] = rng.sample(range(114_000, 117_000), 4)

        # Also token neurons for the language symbol
        l_start = REGIONS["language"][0]
        neurons["language"].extend(rng.sample(range(l_start, l_start + 9000), 2))

        # Pattern neurons: abstract relational pattern
        p_start, p_end = REGIONS["pattern"]
        neurons["pattern"] = rng.sample(range(p_start, p_end + 1), 4)

        # Integration neurons: cross-region binding
        int_start, int_end = REGIONS["integration"]
        neurons["integration"] = rng.sample(range(int_start, int_end + 1), 4)

        # Executive neurons: reasoning/planning
        e_start, e_end = REGIONS["executive"]
        neurons["executive"] = rng.sample(range(e_start, e_start + 7000), 3)  # excitatory only

        # Memory_long neurons: persistent
        ml_start, ml_end = REGIONS["memory_long"]
        neurons["memory_long"] = rng.sample(range(ml_start, ml_end + 1), 4)

        trace = Trace(
            id=f"relational_{i:04d}",
            neurons=neurons,
            strength=rng.uniform(0.3, 0.6),
            decay=1.0,
            polarity=0.0,
            abstraction=rng.uniform(0.5, 0.9),  # relational = highly abstract
            novelty=0.3,
            formation_tick=0,
            label=concept,
        )
        store.add(trace)
        created += 1

    return created
