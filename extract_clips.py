#!/usr/bin/env python3
"""
extract_clips.py — extract species-labeled bird clips from a feeder cam day file.

Usage:
  python3 extract_clips.py <video_file> [--interval 60] [--out ~/Desktop/Bird\ Clips]

Steps:
  1. Sample a frame every INTERVAL seconds via ffmpeg
  2. Send each frame to OpenRouter vision API for bird species ID
  3. Merge nearby detections into clip segments
  4. Cut labeled MP4 clips with ffmpeg (stream copy, no re-encode)

Requires:
  pip install requests
  OPENROUTER_API_KEY env var set
"""

import os, sys, json, base64, subprocess, tempfile, time, argparse
from pathlib import Path

import requests

# Cheapest free vision model on OpenRouter as of 2026-07
MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

CLIP_PAD  = 30   # seconds of padding before/after first/last hit in a segment
MERGE_GAP = 90   # merge hits within this many seconds into one clip


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


def extract_frame(path, t):
    """Return JPEG bytes for a single frame at time t (seconds)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp = f.name
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-ss", str(t), "-i", path,
         "-vframes", "1", "-q:v", "4", "-vf", "scale=1280:-1", tmp],
        capture_output=True
    )
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    return data


def ask_vision(frame_bytes, api_key):
    """Return list of species strings visible in the frame (may be empty)."""
    b64 = base64.b64encode(frame_bytes).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text":
                    "This is a frame from a backyard bird feeder camera in the Pacific Northwest. "
                    "List any birds visible as a JSON array of species common names, e.g. "
                    "[\"Band-tailed Pigeon\",\"Anna's Hummingbird\"]. "
                    "If no birds are visible return []. Return ONLY the JSON array, nothing else."}
            ]
        }],
        "max_tokens": 100
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=30
    )
    text = r.json()["choices"][0]["message"]["content"].strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def cut_clip(path, start, end, species_list, idx, out_dir):
    safe = "_".join(s.replace(" ", "_").replace("'", "") for s in sorted(set(species_list)))
    out = out_dir / f"{idx:03d}_{safe}.mp4"
    start = max(0, start)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-ss", str(start), "-t", str(end - start),
         "-i", path, "-c", "copy", str(out)],
        capture_output=True
    )
    print(f"  -> {out.name}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract labeled bird clips from feeder cam footage.")
    parser.add_argument("video", help="Path to input day-file MP4")
    parser.add_argument("--interval", type=int, default=60, help="Frame sample interval in seconds (default 60)")
    parser.add_argument("--out", default=os.path.expanduser("~/Desktop/Bird Clips"), help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); sys.exit(1)

    video = args.video
    out_dir = Path(args.out) / Path(video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Getting duration of {Path(video).name}...")
    duration = get_duration(video)
    print(f"    {duration/3600:.1f} hours ({duration:.0f}s)")

    timestamps = list(range(0, int(duration), args.interval))
    print(f"[*] Sampling {len(timestamps)} frames every {args.interval}s  (model: {MODEL})")

    hits = []
    for i, t in enumerate(timestamps):
        sys.stdout.write(f"\r    {i+1}/{len(timestamps)}  {t//3600:02.0f}:{(t%3600)//60:02.0f}  hits={len(hits)}  ")
        sys.stdout.flush()
        try:
            frame = extract_frame(video, t)
            if not frame:
                continue
            species = ask_vision(frame, api_key)
            if species:
                hits.append((t, species))
                print(f"\n    [{t//3600:02.0f}:{(t%3600)//60:02.0f}] {species}")
        except Exception as e:
            print(f"\n    [WARN] t={t}: {e}")
        time.sleep(0.1)

    print(f"\n[*] Found {len(hits)} frames with birds.")
    if not hits:
        print("No birds detected."); sys.exit(0)

    # Merge nearby hits into clip segments
    segments = []
    cur_start   = hits[0][0] - CLIP_PAD
    cur_end     = hits[0][0] + CLIP_PAD
    cur_species = list(hits[0][1])

    for t, sp in hits[1:]:
        if t - CLIP_PAD <= cur_end + MERGE_GAP:
            cur_end = t + CLIP_PAD
            cur_species.extend(sp)
        else:
            segments.append((cur_start, cur_end, cur_species))
            cur_start   = t - CLIP_PAD
            cur_end     = t + CLIP_PAD
            cur_species = list(sp)
    segments.append((cur_start, cur_end, cur_species))

    print(f"[*] Cutting {len(segments)} clips into {out_dir}...")
    for i, (s, e, sp) in enumerate(segments):
        cut_clip(video, s, e, sp, i + 1, out_dir)

    print(f"\n[✓] Done — {len(segments)} clips in {out_dir}")


if __name__ == "__main__":
    main()
