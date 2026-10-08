"""Persistent, append-only storage for corrections and training history.

Layout
------
data/vaaksetu.db          SQLite: samples, training_runs (DELETE is blocked by triggers)
data/audio/ab/abcd...wav  content-addressed audio - a file is named by its SHA-256,
                          so it can never be silently overwritten or lost
data/snapshots/*.jsonl    exact manifest of the data used by each training run
data/exports/*.jsonl      full dataset export (HF `datasets`-compatible JSONL)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..audio import load_audio, sha256_bytes, wav_bytes
from ..config import AUDIO_STORE, DB_PATH, EXPORT_DIR, SNAPSHOT_DIR, ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    uid           TEXT UNIQUE NOT NULL,          -- stable id (idempotent inserts)
    modality      TEXT NOT NULL CHECK (modality IN ('stt','tts')),
    lang          TEXT NOT NULL,
    category      TEXT,                          -- name/number/date/alnum/symbol/unit/general
    text          TEXT NOT NULL,                 -- STT: correct transcript | TTS: input text
    spoken        TEXT,                          -- TTS: how the text must be pronounced
    audio_sha     TEXT,                          -- STT: input speech | TTS: target speech
    model_output  TEXT,                          -- what the model produced before correction
    model_version TEXT,                          -- which model version produced it
    was_error     INTEGER NOT NULL DEFAULT 1,    -- 1 = corrected error, 0 = verified correct
    role          TEXT NOT NULL CHECK (role IN ('correction','anchor')),
    batch         TEXT,                          -- collection batch, e.g. round1, ui-2026-09-23
    source        TEXT,                          -- human-ui | seed-sim | self-distill | gemma-assisted
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired')),
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_samples_mod_lang ON samples(modality, lang, role, status);

CREATE TABLE IF NOT EXISTS training_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_key      TEXT NOT NULL,                -- 'stt' or 'tts-ta' ...
    version        TEXT NOT NULL,
    parent_version TEXT,
    base_model     TEXT NOT NULL,
    snapshot_file  TEXT NOT NULL,
    snapshot_sha   TEXT NOT NULL,
    n_samples      INTEGER NOT NULL,
    params_json    TEXT,
    metrics_json   TEXT,
    promoted       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    UNIQUE (model_key, version)
);

CREATE TRIGGER IF NOT EXISTS samples_no_delete BEFORE DELETE ON samples
BEGIN SELECT RAISE(ABORT, 'samples are append-only: set status=retired instead'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON training_runs
BEGIN SELECT RAISE(ABORT, 'training history is append-only'); END;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DataStore:
    def __init__(self, db_path: Path = DB_PATH, audio_root: Path = AUDIO_STORE):
        self.db_path = Path(db_path)
        self.audio_root = Path(audio_root)
        self.audio_root.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ audio
    def put_audio(self, audio) -> str:
        """Store audio (np.ndarray @16k, bytes, or path) and return its SHA-256."""
        if isinstance(audio, (str, Path)):
            audio = load_audio(audio)
        data = wav_bytes(audio) if isinstance(audio, np.ndarray) else bytes(audio)
        sha = sha256_bytes(data)
        path = self.audio_path(sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return sha

    def audio_path(self, sha: str) -> Path:
        return self.audio_root / sha[:2] / f"{sha}.wav"

    # ---------------------------------------------------------------- samples
    def add_sample(self, *, uid: str, modality: str, lang: str, text: str, role: str,
                   category: str | None = None, spoken: str | None = None,
                   audio_sha: str | None = None, model_output: str | None = None,
                   model_version: str | None = None, was_error: bool = True,
                   batch: str | None = None, source: str = "human-ui") -> bool:
        """Insert a sample. Returns False if the uid already exists (idempotent)."""
        with self._conn() as c:
            cur = c.execute(
                """INSERT OR IGNORE INTO samples (uid, modality, lang, category, text, spoken,
                   audio_sha, model_output, model_version, was_error, role, batch, source, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, modality, lang, category, text, spoken, audio_sha, model_output,
                 model_version, int(was_error), role, batch, source, _now()))
            return cur.rowcount == 1

    def retire(self, uid: str) -> None:
        """Soft-delete: the row stays on disk but is excluded from training."""
        with self._conn() as c:
            c.execute("UPDATE samples SET status='retired' WHERE uid=?", (uid,))

    def restore(self, uid: str) -> None:
        """Undo `retire` (the sample becomes eligible for training again)."""
        with self._conn() as c:
            c.execute("UPDATE samples SET status='active' WHERE uid=?", (uid,))

    def samples(self, modality: str, lang: str | None = None, role: str | None = None,
                batch: str | None = None, include_retired: bool = False) -> list[dict]:
        q, args = "SELECT * FROM samples WHERE modality=?", [modality]
        if lang:
            q += " AND lang=?"; args.append(lang)
        if role:
            q += " AND role=?"; args.append(role)
        if batch:
            q += " AND batch=?"; args.append(batch)
        if not include_retired:
            q += " AND status='active'"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY id", args)]

    def stats(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                """SELECT modality, lang, role, batch, COUNT(*) n, SUM(was_error) errors
                   FROM samples WHERE status='active' GROUP BY modality, lang, role, batch
                   ORDER BY modality, lang, role, batch""")]

    # -------------------------------------------------------------- snapshots
    def snapshot(self, name: str, rows: list[dict]) -> tuple[Path, str]:
        """Freeze the exact training set of a run as JSONL; returns (path, sha256)."""
        path = SNAPSHOT_DIR / f"{name}.jsonl"
        lines = [json.dumps({k: r[k] for k in ("uid", "modality", "lang", "category", "text",
                                                "spoken", "audio_sha", "role", "batch", "source")},
                            ensure_ascii=False) for r in rows]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        path.write_bytes(data)
        return path, sha256_bytes(data)

    def record_run(self, *, model_key: str, version: str, parent_version: str | None,
                   base_model: str, snapshot_file: Path, snapshot_sha: str, n_samples: int,
                   params: dict, metrics: dict | None = None, promoted: bool = False) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO training_runs (model_key, version, parent_version, base_model,
                   snapshot_file, snapshot_sha, n_samples, params_json, metrics_json, promoted, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (model_key, version, parent_version, base_model,
                 str(Path(snapshot_file).relative_to(ROOT)), snapshot_sha, n_samples,
                 json.dumps(params, ensure_ascii=False), json.dumps(metrics or {}, ensure_ascii=False),
                 int(promoted), _now()))

    def update_run(self, model_key: str, version: str, metrics: dict, promoted: bool) -> None:
        with self._conn() as c:
            c.execute("UPDATE training_runs SET metrics_json=?, promoted=? WHERE model_key=? AND version=?",
                      (json.dumps(metrics, ensure_ascii=False), int(promoted), model_key, version))

    def runs(self, model_key: str | None = None) -> list[dict]:
        with self._conn() as c:
            if model_key:
                rows = c.execute("SELECT * FROM training_runs WHERE model_key=? ORDER BY id", (model_key,))
            else:
                rows = c.execute("SELECT * FROM training_runs ORDER BY id")
            return [dict(r) for r in rows]

    # ----------------------------------------------------------------- export
    def export_jsonl(self, path: Path | None = None) -> Path:
        """Export every sample (incl. retired) with relative audio paths."""
        path = path or EXPORT_DIR / f"vaaksetu_dataset_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        with self._conn() as c, open(path, "w", encoding="utf-8") as f:
            for r in c.execute("SELECT * FROM samples ORDER BY id"):
                d = dict(r)
                if d["audio_sha"]:
                    d["audio"] = str(self.audio_path(d["audio_sha"]).relative_to(ROOT)).replace("\\", "/")
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return path
