"""R2 — тесты для изотонного агрегатора доли (дорожка Д). Не меняет P8, только новая логика."""

from spanverify.r2_share import IsotonicFragmentAggregator


def test_r2_monotone_by_shape():
    # Простой детерминированный набор: raw -> true
    scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    targets = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    cal = IsotonicFragmentAggregator.fit(scores, targets, mode="demo", bundle_sha256="e"*64)
    # Монотонность по построению
    vals = [cal.transform(s) for s in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert vals == sorted(vals), "PAVA must be monotone"
    # Тождество: при обучающем совпадении
    assert cal.transform(0.0) <= cal.transform(1.0)


def test_r2_estimate_not_worse():
    cal = IsotonicFragmentAggregator.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], mode="demo", bundle_sha256="a"*64)
    # estimate через фрагменты
    class Tok:
        def __init__(self, s):
            self.start = s
    tokens = [Tok(0), Tok(5), Tok(10)]
    class Span:
        def __init__(self, s, e):
            self.start = s
            self.end = e
    spans = [Span(0, 10)]
    token_scores = [0.2, 0.6, 0.8]
    # Should not crash and return 0..1
    v = cal.estimate("hello world test", spans, token_scores, tokens)
    assert 0.0 <= v <= 1.0


def test_r2_save_load_roundtrip(tmp_path):
    cal = IsotonicFragmentAggregator.fit([0.1, 0.9], [0.0, 1.0], mode="hf", bundle_sha256="b"*64)
    p = tmp_path / "cal.json"
    cal.save(p)
    loaded = IsotonicFragmentAggregator.load(p)
    assert loaded.upper == cal.upper
    assert loaded.values == cal.values
    assert loaded.mode == cal.mode
