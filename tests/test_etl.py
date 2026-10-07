from sdrag.parse import clean


def test_clean_joins_hyphens_and_drops_references():
    body = "Autonomous vehicles use li-\ndar sensors for perception in urban scenes.\n\n" * 20
    raw = body + "\nReferences\n[1] Some cited paper, 2020."
    out = clean(raw)
    assert "lidar" in out
    assert "Some cited paper" not in out


def test_clean_removes_page_numbers_and_short_lines():
    raw = "12\n\nshort\n\nThis paragraph is long enough to be kept by the cleaner."
    out = clean(raw)
    assert out == "This paragraph is long enough to be kept by the cleaner."
