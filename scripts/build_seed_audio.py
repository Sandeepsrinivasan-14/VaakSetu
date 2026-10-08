"""Generate the seed recordings used by the demonstration pipeline.

Real users record through the Gradio app; for a reproducible, hands-free demo
we render each catalogue sentence (its *spoken* form) with two different
neural voices per language (Microsoft Edge read-aloud voices via `edge-tts`).
Voice A and voice B are both used, and held-out test sentences use different
carrier templates/values, so a retest is never the exact clip that was
corrected.

Output: data/seed/audio/<lang>/<item_id>__v<k>.wav  +  data/seed/manifest.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edge_tts  # noqa: E402

from vaaksetu.audio import load_audio, save_wav  # noqa: E402
from vaaksetu.config import LANGS, SEED_DIR, ROOT, setup_console  # noqa: E402
from vaaksetu.data.seed_catalog import build_items  # noqa: E402

MANIFEST = SEED_DIR / "manifest.jsonl"


async def render(text: str, voice: str, out: Path, sem: asyncio.Semaphore) -> None:
    async with sem:
        for attempt in range(4):
            try:
                with tempfile.TemporaryDirectory() as td:
                    mp3 = Path(td) / "a.mp3"
                    await edge_tts.Communicate(text, voice).save(str(mp3))
                    save_wav(out, load_audio(mp3))
                return
            except Exception as e:  # network hiccups
                if attempt == 3:
                    raise
                await asyncio.sleep(2 * (attempt + 1))


async def main(concurrency: int) -> None:
    items = build_items()
    sem = asyncio.Semaphore(concurrency)
    clips, jobs = [], []
    for it in items:
        for k, voice in enumerate(LANGS[it["lang"]]["voices"]):
            out = SEED_DIR / "audio" / it["lang"] / f"{it['item_id']}__v{k}.wav"
            clips.append({**it, "voice": voice, "voice_idx": k,
                          "audio": str(out.relative_to(ROOT)).replace("\\", "/")})
            if not out.exists():
                jobs.append(render(it["spoken"], voice, out, sem))
    print(f"{len(items)} items -> {len(clips)} clips ({len(jobs)} to render)")
    done = 0
    for fut in asyncio.as_completed(jobs):
        await fut
        done += 1
        if done % 50 == 0:
            print(f"  rendered {done}/{len(jobs)}", flush=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(ap.parse_args().concurrency))
