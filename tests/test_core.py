"""Unit tests for the parts that must not silently break (no GPU / model downloads needed,
except the tokenizer test which uses the cached MMS tokenizer)."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vaaksetu.data.seed_catalog import build_items  # noqa: E402
from vaaksetu.data.store import DataStore  # noqa: E402
from vaaksetu.registry import Registry  # noqa: E402
from vaaksetu.text import norm_for_scoring, term_hit  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return DataStore(tmp_path / "t.db", tmp_path / "audio")


def test_store_is_append_only(store):
    sha = store.put_audio(np.zeros(1600, dtype=np.float32))
    assert store.add_sample(uid="a", modality="stt", lang="ta", text="x", role="correction", audio_sha=sha)
    assert not store.add_sample(uid="a", modality="stt", lang="ta", text="y", role="correction")  # idempotent
    with pytest.raises(sqlite3.DatabaseError):
        with store._conn() as c:
            c.execute("DELETE FROM samples")
    store.retire("a")
    assert store.samples("stt") == []
    assert len(store.samples("stt", include_retired=True)) == 1


def test_audio_is_content_addressed(store):
    a = np.random.default_rng(0).normal(size=1600).astype(np.float32) * 0.1
    s1, s2 = store.put_audio(a), store.put_audio(a.copy())
    assert s1 == s2 and store.audio_path(s1).exists()


def test_snapshot_hash_is_stable(store, tmp_path, monkeypatch):
    import vaaksetu.data.store as m
    monkeypatch.setattr(m, "SNAPSHOT_DIR", tmp_path)
    store.add_sample(uid="a", modality="tts", lang="hi", text="₹500", spoken="पाँच सौ रुपये", role="correction")
    rows = store.samples("tts")
    assert store.snapshot("s1", rows)[1] == store.snapshot("s2", rows)[1]


def test_registry_versions_and_rollback(tmp_path):
    r = Registry(tmp_path / "reg.json")
    assert r.active("stt") == "base" and r.next_version("stt") == "v001"
    r.register("stt", "v001", {}); r.promote("stt", "v001")
    r.register("stt", "v002", {}); r.promote("stt", "v002")
    r.rollback("stt", "v001")
    assert Registry(tmp_path / "reg.json").active("stt") == "v001"


def test_scoring_normalisation():
    assert norm_for_scoring("₹2,500.") == norm_for_scoring("₹2500")
    assert norm_for_scoring("5 µg") == norm_for_scoring("5 μg")          # micro sign vs Greek mu
    assert term_hit("Vehicle number TN09AB1234.", "TN09AB1234")
    assert term_hit("Vehicle number TN 09 AB 1234", "TN09AB1234")         # spacing ignored
    assert not term_hit("alpha particles", "α particles")


def test_catalog_splits_are_disjoint():
    """round1/round2 deliberately use different train/test content to measure
    generalisation.  round3 is a single mandatory sample (a later, explicit project
    requirement) that must be reused identically for training, testing and the final
    demonstration, so it is excluded from this disjointness check by design."""
    items = [i for i in build_items() if i.get("round") != "round3"]
    for lang in ("en", "hi", "ta"):
        train = {i["text"] for i in items if i["lang"] == lang and i["set"].endswith("_train")}
        test = {i["text"] for i in items if i["lang"] == lang and i["set"].endswith("_test")}
        assert train and test and not (train & test)


def test_mas_is_monotonic():
    from vaaksetu.tts.train import _mas_single
    v = np.random.default_rng(0).normal(size=(40, 9)).astype(np.float32)
    p = _mas_single(v, 40, 9)
    assert (p.sum(1) == 1).all()                       # each frame -> one token
    idx = p.argmax(1)
    assert (np.diff(idx) >= 0).all() and idx[0] == 0 and idx[-1] == 8
    assert (np.diff(idx) <= 1).all()                   # no token skipped


def _rows(n_wrong, n_total, set_name="general"):
    """Single-word ref/hyp rows: each wrong item contributes exactly 1/n_total
    to corpus WER, so the expected number is exact and easy to hand-check."""
    return [{"item_id": f"i{i}", "set": set_name, "ref": "word",
             "hyp": "word" if i >= n_wrong else "nope"} for i in range(n_total)]


def test_gate_rejects_a_large_consistent_regression():
    from vaaksetu.learning import gate
    old_rows = _rows(0, 20)                 # parent: 0% WER
    new_rows = _rows(6, 20)                 # candidate: 30% WER, consistently
    ok, reasons = gate(new_rows, old_rows, "wer", general_tol=0.02, prev_tol=0.02, prev_sets=[])
    assert not ok
    assert "general" in reasons[0]


def test_gate_tolerates_single_item_noise_on_a_small_sample():
    from vaaksetu.learning import gate
    old_rows = _rows(0, 5)                  # parent: 0% WER
    new_rows = _rows(1, 5)                  # candidate: one item out of five
    ok, _ = gate(new_rows, old_rows, "wer", general_tol=0.02, prev_tol=0.02, prev_sets=[])
    assert ok                               # flat-threshold alone would reject this; bootstrap should not


def test_balanced_batch_rows_gives_each_batch_equal_weight():
    from vaaksetu.stt.train import balanced_batch_rows
    by_batch = {"round1": [{"batch": "round1", "i": i} for i in range(5)],
                "round2": [{"batch": "round2", "i": i} for i in range(50)]}
    rng = np.random.default_rng(0)
    rows = balanced_batch_rows(by_batch, target_per_batch=50, rng=rng)
    assert len(rows) == 100                                            # 50 + 50, not 5 + 50
    assert sum(r["batch"] == "round1" for r in rows) == 50
    assert sum(r["batch"] == "round2" for r in rows) == 50
    assert len({r["i"] for r in rows if r["batch"] == "round1"}) == 5   # small batch is resampled, not fabricated


def test_number_to_words_matches_seed_catalog_english():
    # Ground truth is the project's own hand-authored seed sentences
    # (vaaksetu/data/seed_catalog.py) - if the converter agrees with those,
    # it is producing the same phrasing a human already validated.
    from vaaksetu.tts.augment import currency_amount_to_words_en
    assert currency_amount_to_words_en(500) == "five hundred rupees"
    assert currency_amount_to_words_en(2_500) == "two thousand five hundred rupees"
    assert currency_amount_to_words_en(1_200) == "one thousand two hundred rupees"
    assert currency_amount_to_words_en(10_000) == "ten thousand rupees"
    assert currency_amount_to_words_en(750) == "seven hundred and fifty rupees"
    assert currency_amount_to_words_en(3_000) == "three thousand rupees"


def test_augment_currency_corrections_are_unique_and_well_formed():
    from vaaksetu.tts.augment import augment_currency_corrections_en
    pairs = augment_currency_corrections_en(20, seed=0)
    assert len(pairs) == 20
    texts = [t for t, _ in pairs]
    assert len(set(texts)) == 20                        # no duplicate amounts
    for text, spoken in pairs:
        assert text.startswith("₹") and text.endswith(".")
        assert spoken.endswith(" rupees")


def test_vocab_extension_slots():
    pytest.importorskip("transformers")
    from transformers import VitsModel
    from vaaksetu.tts.tokenization_vaaksetu import VaakSetuVitsTokenizer
    from vaaksetu.tts.vocab import dropped_fraction, extend_vocab, unknown_chars
    try:
        tok = VaakSetuVitsTokenizer.from_pretrained("facebook/mms-tts-eng")
        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    except OSError:
        pytest.skip("MMS checkpoint not cached")
    skip = " .,"
    assert dropped_fraction(tok, "α", skip) == 1.0
    n0 = model.text_encoder.embed_tokens.num_embeddings
    extend_vocab(model, tok, unknown_chars(tok, ["α rays"], skip), slots=4)
    assert dropped_fraction(tok, "α", skip) == 0.0
    assert model.text_encoder.embed_tokens.num_embeddings == n0 + 4
    assert tok.tokenize("α")[1::2] == ["α", "α#1", "α#2", "α#3"]
