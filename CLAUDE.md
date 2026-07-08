# feeder-cam-clips

Extract species-labeled bird clips from feeder cam day files using OpenRouter vision.

## Usage

```bash
export OPENROUTER_API_KEY=sk-or-...
pip install requests

python3 extract_clips.py <video.mp4> [--interval 60] [--out ~/Desktop/Bird\ Clips]
```

## What it does

1. Samples a JPEG frame every `--interval` seconds from the input MP4 (default 60s)
2. Sends each frame to OpenRouter vision API (`nvidia/nemotron-nano-12b-v2-vl:free` by default) to identify bird species
3. Merges nearby detections (within 90s) into single clip segments
4. Cuts labeled MP4 clips using `ffmpeg -c copy` (no re-encode, fast)

Output clips are named `001_Band-tailed_Pigeon.mp4`, `002_Annas_Hummingbird_House_Finch.mp4`, etc.

## Day files

Feeder cam day files live on the NAS at:
```
//100.82.243.14/Crispy Cloud Photos/Bird Box Footage/1-SOURCE/outside-hummingbird/
```

NAS is accessible via Tailscale at `100.82.243.14` (hostname: `photodrive`).

## Model

Default: `nvidia/nemotron-nano-12b-v2-vl:free` (free tier on OpenRouter).  
To use a different model, edit the `MODEL` constant in `extract_clips.py`.

## Requirements

- Python 3
- `requests` pip package
- `ffmpeg` and `ffprobe` in PATH
- `OPENROUTER_API_KEY` env var
