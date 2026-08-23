import pandas as pd

from sqlshield.preprocessing import (
    strict_clean,
    remove_global_normalized_conflicts,
    group_aware_split,
)


def test_strict_clean_preserves_conflicting_exact_text_labels():
    raw = pd.DataFrame({
        "Query": ["#NAME?", "#NAME?", "select 1", "select 1"],
        "Label": [0, 1, 0, 0],
    })
    out = strict_clean(raw, "Query", "Label", "A")
    # Same (text,label) duplicate is removed, but the contradictory labels remain.
    assert len(out) == 3
    assert set(out.loc[out.text == "#NAME?", "label"]) == {0, 1}


def test_global_conflict_is_removed_from_both_sources():
    a = pd.DataFrame({
        "text": ["#NAME?", "#NAME?", "select 1"],
        "label": [0, 1, 0],
        "source": ["A"] * 3,
        "normalized_text": ["#name?", "#name?", "select 1"],
    })
    b = pd.DataFrame({
        "text": ["select 2"],
        "label": [0],
        "source": ["B"],
        "normalized_text": ["select 2"],
    })
    ac, bc, conflicts = remove_global_normalized_conflicts(a, b)
    assert len(conflicts) == 2
    assert "#name?" not in set(ac.normalized_text)
    assert "#name?" not in set(bc.normalized_text)


def test_group_aware_split_has_zero_normalized_overlap():
    rows = []
    # 200 unique groups gives enough examples per class for both stratified stages.
    for label in [0, 1]:
        for i in range(100):
            norm = f"g_{label}_{i}"
            rows.append({"text": norm, "label": label, "source": "toy", "normalized_text": norm})
            # Add a second row to some groups to prove groups stay intact.
            if i % 3 == 0:
                rows.append({"text": norm + "  ", "label": label, "source": "toy", "normalized_text": norm})
    df = pd.DataFrame(rows)
    tr, va, te = group_aware_split(df)
    assert set(tr.normalized_text).isdisjoint(set(va.normalized_text))
    assert set(tr.normalized_text).isdisjoint(set(te.normalized_text))
    assert set(va.normalized_text).isdisjoint(set(te.normalized_text))
