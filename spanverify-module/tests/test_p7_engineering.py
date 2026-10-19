"""P7 exact arithmetic/offset regressions, independent of HF availability."""

from __future__ import annotations

import random

from spanverify.core import tokenize_with_offsets
from spanverify.features import _is_content, _normalised_support, _span_overlap, map_token_positions


def test_interval_index_and_membership_match_legacy_byte_exact():
    import json

    rng = random.Random(731)
    for _ in range(300):
        spans = [(i, rng.randrange(-20, 100), rng.randrange(-20, 100)) for i in range(50)]
        tokens = [(rng.randrange(-20, 100), rng.randrange(-20, 100)) for _ in range(30)]
        shift = rng.randrange(-10, 10)
        expected = []
        for start, end in tokens:
            hits = [(p, _span_overlap((start + shift, end + shift), (a, b))) for p, a, b in spans]
            hits = [(p, n) for p, n in hits if n > 0]
            cutoff = max(1, max((n for _, n in hits), default=0) // 2)
            expected.append(sorted(p for p, n in hits if n >= cutoff))
        assert map_token_positions(tokens, spans, shift) == expected
    assert map_token_positions([], [], 0) == []
    assert map_token_positions([(0, 1)], [(3, 0, 1)], 0) == [[3]]
    from types import SimpleNamespace

    from spanverify.core import _SpanIndex

    for _ in range(300):
        spans = [
            SimpleNamespace(start=rng.randrange(-20, 100), end=rng.randrange(-20, 100))
            for _ in range(rng.randrange(60))
        ]
        index = _SpanIndex(spans)
        for _ in range(100):
            a, b = rng.randrange(-25, 110), rng.randrange(-25, 110)
            assert index.overlaps(a, b) == any(a < t.end and b > t.start for t in spans)
            assert index.contains(a) == any(t.start <= a < t.end for t in spans)
    for text in ["", "Один", ", !", "Ёлка 10 лет. " * 1000]:
        tokens = tokenize_with_offsets(text)
        mass = [rng.random() for _ in tokens]
        content = [i for i, t in enumerate(tokens) if _is_content(t)]
        background = sum(mass[i] for i in content) / len(content) if content else 0
        expected = (
            [
                min(1.0, max(0.0, v / max(background, 1e-9))) if i in content else min(1.0, max(0.0, v))
                for i, v in enumerate(mass)
            ]
            if content
            else [0.0] * len(mass)
        )
        assert json.dumps(_normalised_support(mass, tokens)) == json.dumps(expected)


def test_window_partition_preserves_unicode_boundaries_and_all_characters():
    from spanverify.hf_windows import character_windows

    assert character_windows("", [], 1) == []
    text = "ёжик  конец "
    offsets = [(0, 1), (0, 1), (1, 4), (6, 11)]
    spans = character_windows(text, offsets, 1)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(a < b for a, b in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_full_answer_stitches_every_token_once_without_zero_fill(monkeypatch, tmp_path):
    from dataclasses import fields

    from spanverify.features import FeatureMatrix
    from spanverify.hf_windows import full_answer_features

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    scorer_calls = 0

    def scorer(answer, context, **kwargs):
        nonlocal scorer_calls
        scorer_calls += 1
        count = len(kwargs["answer_tokens"])
        assert answer and kwargs["_windowed"]
        return FeatureMatrix(
            **{f.name: [0.75] * count for f in fields(FeatureMatrix) if f.name not in ("meta", "token_risk")},
            meta={"all_answer_tokens_mapped": True},
        )

    for answer in ["Ё" * 1300, "начало конец " * 150]:
        tokens = tokenize_with_offsets(answer)
        before_calls = scorer_calls
        result = full_answer_features(
            answer, "контекст " * 100, tokens, Tokenizer(), None, scorer, {"max_length": 512, "model_name": "fixture"}
        )
        assert result.meta["scored_answer_tokens"] == len(tokens)
        assert result.meta["answer_last_end"] == tokens[-1].end
        assert result.attention_entropy == [0.75] * len(tokens)
        assert result.meta["answer_windows"] > 1
        assert result.meta["model_calls"] == scorer_calls - before_calls
        assert result.meta["identical_window_cache_hits"] > 0
        assert (
            result.meta["logical_window_scores"]
            == result.meta["model_calls"] + result.meta["identical_window_cache_hits"]
        )

    # Cover lazy runtime identity and long-dispatch control flow without fitting
    # a model or silently replacing a requested real HF production backend.
    import sys
    import types

    from spanverify import features, hf_windows
    from spanverify.backends.hf import HFBackend

    fake_model = types.SimpleNamespace(config=types.SimpleNamespace())
    fake_model.eval = lambda: None
    fake_model.to = lambda device: None

    def fake_tokenizer(text, **kwargs):
        return {"input_ids": [0] * 513}

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: fake_tokenizer)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_model)
    features._cached_hf_runtime.cache_clear()
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "torch", torch)
        m.setitem(sys.modules, "transformers", transformers)
        m.setattr(HFBackend, "dependencies", staticmethod(lambda: (True, "")))
        runtime = features._load_hf_runtime("fixture", None)
        assert runtime == features._load_hf_runtime("fixture", None)
        sentinel = FeatureMatrix(attention_entropy=[0.75], meta={"full_answer": True})
        m.setattr(hf_windows, "full_answer_features", lambda *a, **kw: sentinel)
        assert features.hf_features("слово", "", model_name="fixture") is sentinel
        assert features.hf_features("", "", model_name="fixture").meta["empty"]
        import pytest

        def failed(*a, **kw):
            raise RuntimeError("synthetic backend failure")

        m.setattr(features, "hf_features", failed)
        with pytest.raises(hf_windows.FullAnswerCoverageError):
            features.hf_features_chunked(
                "слово",
                "Контекст содержит документы. " * 100,
                model_name="fixture",
                max_tokens_per_chunk=8,
            )

    features._cached_hf_runtime.cache_clear()

    # A frozen parent package may lack an optional stdlib submodule. The
    # portable Lib path must extend that package, not just top-level sys.path.
    import importlib.util
    from pathlib import Path

    libs = tmp_path / "libs"
    libs.mkdir()
    stdlib = tmp_path / "python" / "Lib"
    probe_path = stdlib / "p7_stdlib_probe"
    probe_path.mkdir(parents=True)
    (tmp_path / "python" / "DLLs").mkdir()
    probe = types.ModuleType("p7_stdlib_probe")
    probe.__path__ = ["frozen-only"]
    with monkeypatch.context() as m:
        m.setattr(sys, "path", list(sys.path))
        m.setitem(sys.modules, "p7_stdlib_probe", probe)
        m.chdir(Path(__file__).resolve().parents[1])
        spec = importlib.util.spec_from_file_location("p7_launcher_fixture", "scripts/exe_launcher.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        m.setattr(launcher, "configure_stdio", lambda: None)
        m.setattr(launcher, "free_port", lambda port: 18779)
        m.setattr(launcher, "serve", lambda **kwargs: None)
        assert launcher.main(["--runtime-dir", str(libs), "--quiet", "--no-browser"]) == 0
        assert str(libs) == sys.path[0]
        assert probe.__path__ == [str(probe_path), "frozen-only"]


def test_cached_cosine_norms_are_float_exact():
    from spanverify.features import _cosine_dense, _cosine_dense_normed, _dense_cosine_scores_exact

    rng = random.Random(19)
    for a, b in [([], []), ([0.0], [0.0]), ([1.0], [-1.0])] + [
        ([rng.random() - 0.5 for _ in range(768)], [rng.random() - 0.5 for _ in range(768)]) for _ in range(50)
    ]:
        na, nb = sum(x * x for x in a) ** 0.5, sum(x * x for x in b) ** 0.5
        assert _cosine_dense(a, b).hex() == _cosine_dense_normed(a, b, na, nb).hex()

    vectors = [[rng.random() - 0.5 for _ in range(768)] for _ in range(100)]
    query = vectors[0]
    norm = sum(x * x for x in query) ** 0.5
    norms = [sum(x * x for x in row) ** 0.5 for row in vectors]
    actual = _dense_cosine_scores_exact(query, vectors, norm, norms)
    assert [x.hex() for x in actual] == [_cosine_dense(query, row).hex() for row in vectors]


def test_sentence_index_matches_prefix_reference_and_style_rows():
    import re

    from spanverify.features import FeatureMatrix, is_scored_token
    from spanverify.participation import _cliche_rate, _len_norm, _sentence_bounds, _sentence_of, token_rows
    from spanverify.text import _SENT_BOUNDARY, ABBREV, sentences

    rng = random.Random(251)
    for _ in range(100):
        text = "".join(
            rng.choice(["А. ", "ст. 15. ", "г. Москва. ", "Десять лет!\n", "«ёлка». ", " \t", "\u00a0", "Один? "])
            for _ in range(20)
        )
        cuts = []
        for m in _SENT_BOUNDARY.finditer(text):
            last = re.split(r'[\s«"(]', text[: m.start()].rstrip())[-1].strip(".,!?…").lower()
            if last and len(last) <= 4 and last in ABBREV:
                continue
            cuts.append(m.end())
        starts = [0] + cuts
        ends = cuts + [len(text)]
        expected = [(a, b) for a, b in zip(starts, ends, strict=True) if text[a:b].strip()]
        assert sentences(text) == expected
        tokens = tokenize_with_offsets(text)
        matrix = FeatureMatrix(
            attention_entropy=[0.2] * len(tokens),
            ctx_attention_mass=[0.3] * len(tokens),
            embedding_density=[0.4] * len(tokens),
        )
        legacy = []
        for t in tokens:
            if is_scored_token(t.text):
                a, b = _sentence_of(t.start, _sentence_bounds(text))
                frag = text[a:b]
                legacy.append([0.2, 0.3, 0.4, _cliche_rate(frag), _len_norm(frag)])
        assert token_rows(text, matrix)[0] == legacy


def test_share_calibration_is_monotone_versioned_and_quality_invariant(monkeypatch):
    import json
    from pathlib import Path

    import pytest

    from spanverify.engine import Verifier, WeightsBundle
    from spanverify.share_calibration import AiShareCalibrator

    sha = "e0b13c693a058dfd66ed24a52fc14e9c70964e95885358e7b99cc1f328ad6892"
    c = AiShareCalibrator.fit([0.1, 0.1, 0.3, 0.5, 0.8], [1.0, 0.0, 0.9, 0.2, 0.4], mode="demo", bundle_sha256=sha)
    clone = AiShareCalibrator.from_dict(c.to_dict())
    assert [clone.transform(x / 100) for x in range(101)] == sorted(clone.transform(x / 100) for x in range(101))
    for bad in [float("nan"), -1.0, 2.0]:
        with pytest.raises(ValueError):
            clone.transform(bad)
    for bad in [
        {},
        {"v2": {}},
        {c.schema: {"extra": 1}},
        {c.schema: dict(upper=[1.0, 0.1], values=[0.2, 0.5], mode="demo", bundle_sha256=sha)},
        {c.schema: dict(upper=[], values=[], mode="demo", bundle_sha256=sha)},
    ]:
        with pytest.raises(ValueError):
            AiShareCalibrator.from_dict(bad)
    with pytest.raises(ValueError):
        AiShareCalibrator.fit([], [], mode="demo", bundle_sha256=sha)
    bundle = WeightsBundle.load(
        Path(__file__).parents[1] / "reports/experiments/p5_span_filter/run39170981235/combined_bundle.json"
    )
    raw = Verifier(mode="demo", weights=bundle, span_expansion="mask").verify(
        "Срок 10 лет.", "Срок 5 лет.", with_tokens=True
    )
    v = Verifier(mode="demo", weights=bundle, span_expansion="mask", ai_share_calibrator=c.to_dict())
    after = v.verify("Срок 10 лет.", "Срок 5 лет.", with_tokens=True)
    assert json.dumps(raw.tokens, sort_keys=True) == json.dumps(after.tokens, sort_keys=True)
    assert raw.score == after.score and raw.spans == after.spans and raw.verdict == after.verdict
    assert raw.ai_share == after.stats["ai_share_raw"] and after.ai_share == c.transform(raw.ai_share)
    assert v.verify("", "").ai_share == 0.0
    wrong = c.to_dict()
    wrong[c.schema]["mode"] = "hf"
    with pytest.raises(ValueError):
        Verifier(mode="demo", weights=bundle, ai_share_calibrator=wrong)
    monkeypatch.setenv("SPANVERIFY_AI_SHARE_CALIBRATOR", "missing-calibrator.json")
    with pytest.raises(FileNotFoundError):
        Verifier(mode="demo", weights=bundle)
