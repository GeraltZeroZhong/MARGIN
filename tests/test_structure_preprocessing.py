from __future__ import annotations

from margin.preprocessing.structure import _format_dssp


def test_dssp_formatter_uses_explicit_mapping_keys() -> None:
    class ValueIteratingAnnotations(dict):
        def __iter__(self):
            return iter(self.values())

    key = ("A", (" ", 7, " "))
    annotations = ValueIteratingAnnotations(
        {key: (0, "A", "H", 0.125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)}
    )

    assert _format_dssp(annotations) == {key: ("H", 0.125)}
