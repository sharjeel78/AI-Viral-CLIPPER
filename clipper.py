"""Command-line version.

Examples:
  python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python clipper.py "D:\\videos\\podcast.mp4" --clips 8 --layout split --llm none
  python clipper.py LINK1 LINK2 LINK3          (several videos, one after another)
"""
import argparse
import sys

from core.pipeline import run_pipeline
from core.utils import load_config, merge_config


def main() -> int:
    p = argparse.ArgumentParser(description="Turn a long video into viral 9:16 clips.")
    p.add_argument("sources", nargs="+", help="one or more YouTube URLs or local video files")
    p.add_argument("--clips", type=int, help="number of clips to export")
    p.add_argument("--min", type=float, dest="min_s", help="minimum clip length (seconds)")
    p.add_argument("--max", type=float, dest="max_s", help="maximum clip length (seconds)")
    p.add_argument("--layout", choices=["auto", "track", "split", "fit"])
    p.add_argument("--llm", choices=["ollama", "openai", "anthropic", "none"])
    p.add_argument("--whisper", help="whisper model: tiny, base, small, medium, large-v3")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--no-hook", action="store_true")
    p.add_argument("--config", help="path to a config.yaml")
    a = p.parse_args()

    cfg = merge_config(load_config(a.config), {
        "clips": {"count": a.clips, "min_seconds": a.min_s, "max_seconds": a.max_s},
        "reframe": {"layout": a.layout},
        "llm": {"provider": a.llm},
        "transcription": {"model": a.whisper},
        "captions": {"enabled": False if a.no_captions else None},
        "hook_title": {"enabled": False if a.no_hook else None},
    })

    def progress(frac: float, msg: str) -> None:
        bar = "#" * int(frac * 30)
        sys.stdout.write(f"\r[{bar:<30}] {frac:5.1%}  {msg[:70]:<70}")
        sys.stdout.flush()

    failed = 0
    for i, source in enumerate(a.sources, 1):
        if len(a.sources) > 1:
            print(f"\n=== Video {i} of {len(a.sources)}: {source}")
        try:
            summary = run_pipeline(source, cfg, progress)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 1
        except Exception as e:
            print(f"\n\nERROR: {e}")
            failed += 1
            continue
        print("\n")
        for c in summary["clips"]:
            flag = "  <- check framing" if c["needs_review"] else ""
            print(f"  [{c['score']:>3}] {c['file']}  ({c['duration']}s, {c['layout']}){flag}")
        print(f"\nSaved to: {summary['output_dir']}")
    if failed:
        print(f"\n{failed} of {len(a.sources)} videos failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
