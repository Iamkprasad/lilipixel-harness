# LiliPixel-V1 Harness

Small local webapp + harness for [Pinkstack/LiliPixel-V1-190M](https://huggingface.co/Pinkstack/LiliPixel-V1-190M), a 190M-parameter pixel-art diffusion transformer (16×16, 32×32, 64×64).

## What it does

- `run.sh` — harness: creates a venv, installs deps, starts the webapp.
- `server.py` — stdlib HTTP server; lazy-loads the diffusers pipeline on generate.
- `web/` — static UI for download, load, prompt, generate, and PNG download.

## Run

```bash
cd /root/lili_pixel_webapp
chmod +x run.sh
./run.sh
# open http://127.0.0.1:8090/
```

Environment:

- `LILI_PORT` — web port (default `8090`)
- `LILI_TORCH_INDEX` — pip torch index (default CPU wheels: `https://download.pytorch.org/whl/cpu`)

## Use

1. Start the app with `./run.sh`.
2. In the UI, click **Download model** (downloads `Pinkstack/LiliPixel-V1-190M` into `models/`).
3. Click **Generate**. First generation loads the model; the frozen text encoder (`MongoDB/voyage-4-nano`) may be downloaded by Transformers on first use.
4. Pick 16/32/64, adjust steps/CFG/seed/batch, download PNGs.

## Notes

- 16×16/32×32: sprite-style, transparent background auto-appended to prompt.
- 64×64: opaque scenes; prefix prompt with `realistic photo, ` for realism mode.
- Model weights are ~0.7 GB; text encoder download is separate on first load.
- CPU works but is slow; CUDA is used automatically when available.
