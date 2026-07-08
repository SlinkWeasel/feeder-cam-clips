#!/usr/bin/env python3
"""
dslr_shorts.py — turn Lumix DSLR clips into 9:16 YouTube Shorts.

Usage:
  python3 dslr_shorts.py <folder_or_file> [--out ~/Desktop/Shorts]

For each MP4:
  - Clips ≤ 60s  → crop to 9:16 (center) and output as-is
  - Clips > 60s  → sample frames via OpenRouter vision to find the most
                   active 60s window, then crop and cut
  - Output: <filename>_short.mp4 in --out folder

Requires:
  pip install requests
  OPENROUTER_API_KEY env var (only needed for clips > 60s)
  ffmpeg in PATH
"""

import os, sys, json, base64, subprocess, tempfile, argparse, time
from pathlib import Path

MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


def extract_frame(path, t):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-ss", str(t), "-i", str(path),
         "-vframes", "1", "-q:v", "5", "-vf", "scale=960:-1", tmp],
        capture_output=True
    )
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    return data


def score_frame(frame_bytes, api_key):
    """Return an activity score 0-10 for the frame (10 = lots of action)."""
    b64 = base64.b64encode(frame_bytes).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text":
                    "Rate the visual interest of this wildlife camera frame on a scale of 0-10. "
                    "10 = animal clearly visible and active, 5 = animal partially visible, "
                    "0 = empty scene. Reply with ONLY a single integer, nothing else."}
            ]
        }],
        "max_tokens": 5
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=20
    )
    try:
        return int(r.json()["choices"][0]["message"]["content"].strip())
    except Exception:
        return 0


def find_best_window(path, duration, api_key, window=60, sample=8):
    """Sample frames across the clip and return the start time of the best 60s window."""
    step = max(5, int(duration / sample))
    timestamps = list(range(0, int(duration - window), step)) or [0]
    scores = []
    for t in timestamps:
        mid = t + window // 2
        try:
            frame = extract_frame(path, mid)
            s = score_frame(frame, api_key)
        except Exception:
            s = 0
        scores.append((t, s))
        sys.stdout.write(f"\r    scoring t={mid}s → {s}  ")
        sys.stdout.flush()
        time.sleep(0.1)
    print()
    best = max(scores, key=lambda x: x[1])
    return best[0]


def make_short(path, out_dir, start=0, duration=None, api_key=None):
    """Crop to 9:16 (center) and cut. Returns output path."""
    out = out_dir / (Path(path).stem + "_short.mp4")

    # Build filter: center-crop to 9:16, then scale to 1080x1920
    crop_filter = "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920"

    cmd = ["ffmpeg", "-nostdin", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", crop_filter, "-c:v", "libx264", "-crf", "18",
            "-preset", "fast", "-c:a", "aac", "-b:a", "128k", str(out)]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed: {result.stderr.decode()[-200:]}")
        return None
    return out


def process_clip(path, out_dir, api_key):
    duration = get_duration(path)
    print(f"  duration: {duration:.1f}s")

    if duration <= 60:
        print(f"  ≤60s — cropping whole clip")
        out = make_short(path, out_dir)
    else:
        if not api_key or not HAS_REQUESTS:
            print(f"  >60s but no API key — taking center 60s")
            start = max(0, duration / 2 - 30)
            out = make_short(path, out_dir, start=start, duration=60)
        else:
            print(f"  >60s — finding best 60s window...")
            start = find_best_window(path, duration, api_key)
            print(f"  best window starts at {start:.0f}s")
            out = make_short(path, out_dir, start=start, duration=60)

    if out:
        print(f"  -> {out.name}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Make 9:16 YouTube Shorts from Lumix DSLR clips.")
    parser.add_argument("input", help="MP4 file or folder of MP4s")
    parser.add_argument("--out", default=os.path.expanduser("~/Desktop/Shorts"),
                        help="Output folder (default: ~/Desktop/Shorts)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.input)
    if inp.is_file():
        clips = [inp]
    else:
        clips = sorted(inp.glob("**/*.MP4")) + sorted(inp.glob("**/*.mp4"))

    if not clips:
        print("No MP4 files found."); sys.exit(1)

    print(f"[*] Processing {len(clips)} clip(s) → {out_dir}")
    for i, clip in enumerate(clips, 1):
        print(f"\n[{i}/{len(clips)}] {clip.name}")
        try:
            process_clip(clip, out_dir, api_key)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n[✓] Done — Shorts in {out_dir}")


if __name__ == "__main__":
    main()
