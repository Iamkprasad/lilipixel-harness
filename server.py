"""LiliPixel-V1 webapp — standalone local harness for Pinkstack/LiliPixel-V1-190M.

Stdlib HTTP server; torch/diffusers are imported lazily when generating or
downloading. Run via run.sh (creates venv + installs deps) or directly in an
env that has: torch diffusers transformers accelerate huggingface_hub safetensors.
"""
import base64
import json
import os
import queue
import re
import threading
import time
import traceback
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
STATE = ROOT / "state"

PORT = int(os.environ.get("LILI_PORT", "8090"))
MODEL_ID = "Pinkstack/LiliPixel-V1-190M"
MODEL_DIR = MODELS / "LiliPixel-V1-190M"
ALLOWED_SIZES = (16, 32, 64)

_lock = threading.RLock()
_gen_lock = threading.Lock()
_dl_lock = threading.Lock()
_pipe = None
_pipe_device = None
_pipe_dtype = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read(path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default if default is not None else {}


def _mem_info():
    info = {"total_mb": None, "free_mb": None}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["total_mb"] = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                info["free_mb"] = int(line.split()[1]) // 1024
    except Exception:
        pass
    return info


def _model_ready():
    # model_index.json is the diffusers sentinel
    return (MODEL_DIR / "model_index.json").is_file()


def _status():
    dl = _read(STATE / "download.json", {"status": "idle"})
    return {
        "model_id": MODEL_ID,
        "model_dir": str(MODEL_DIR),
        "installed": _model_ready(),
        "loaded": _pipe is not None,
        "device": _pipe_device,
        "download": dl,
        "memory": _mem_info(),
    }


# ---------------------------------------------------------------------------
# download

def _download_worker():
    state = {"status": "downloading", "message": "Starting download...", "percent": 0}
    _write(STATE / "download.json", state)
    try:
        from huggingface_hub import snapshot_download
        MODELS.mkdir(parents=True, exist_ok=True)
        state["message"] = "Downloading model files from HuggingFace..."
        _write(STATE / "download.json", state)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        state.update({"status": "completed", "percent": 100, "message": "Download complete"})
    except Exception as e:
        state.update({"status": "error", "error": str(e), "message": str(e)})
    _write(STATE / "download.json", state)


def start_download():
    if _model_ready():
        return {"status": "already_installed"}
    current = _read(STATE / "download.json", {})
    if current.get("status") == "downloading":
        return {"status": "in_progress"}
    with _dl_lock:
        current = _read(STATE / "download.json", {})
        if current.get("status") == "downloading":
            return {"status": "in_progress"}
        threading.Thread(target=_download_worker, daemon=True).start()
    return {"status": "started"}


# ---------------------------------------------------------------------------
# generation

def _load_pipe(on_progress=None):
    global _pipe, _pipe_device, _pipe_dtype
    with _lock:
        if _pipe is not None:
            return _pipe
        if not _model_ready():
            raise ValueError("Model not downloaded. Click Download first.")
        if on_progress:
            on_progress({"stage": "loading", "message": "Loading model..."})

        import torch
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        # bfloat16 on cpu is fine on recent CPUs; fall back to float32 if needed
        if device == "cpu":
            try:
                _ = torch.zeros(1, dtype=torch.bfloat16)
                dtype = torch.bfloat16
            except Exception:
                dtype = torch.float32

        # Prefer local files; pipeline.py lives in MODEL_DIR
        kwargs = dict(trust_remote_code=True, torch_dtype=dtype, local_files_only=True)
        try:
            pipe = DiffusionPipeline.from_pretrained(str(MODEL_DIR), **kwargs)
        except Exception:
            # fallback: try as custom_pipeline reference
            kwargs.pop("local_files_only", None)
            pipe = DiffusionPipeline.from_pretrained(
                str(MODEL_DIR), custom_pipeline=str(MODEL_DIR), **kwargs
            )
        pipe = pipe.to(device)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        _pipe = pipe
        _pipe_device = device
        _pipe_dtype = str(dtype).replace("torch.", "")
        return _pipe


def _normalize_prompt(prompt, size):
    p = " ".join(str(prompt or "").strip().split())
    if not p:
        raise ValueError("Prompt is required")
    if size in (16, 32):
        low = p.lower()
        if "transparent background" not in low:
            p = p.rstrip(", ") + ", transparent background"
    return p


def _generation_payload(body):
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Prompt is required")
    size = int(body.get("size") or 32)
    if size not in ALLOWED_SIZES:
        raise ValueError("Size must be one of: 16, 32, 64")
    width = int(body.get("width") or size)
    height = int(body.get("height") or size)
    if width not in ALLOWED_SIZES or height not in ALLOWED_SIZES:
        raise ValueError("Width/height must be one of: 16, 32, 64")
    steps = int(body.get("steps") or (50 if max(width, height) == 64 else 30))
    if steps < 1 or steps > 100:
        raise ValueError("Steps must be 1..100")
    cfg = float(body.get("cfg", 3.0))
    if cfg < 0 or cfg > 20:
        raise ValueError("CFG must be 0..20")
    seed = body.get("seed")
    seed = int(seed) if seed is not None and str(seed) != "" else None
    batch = int(body.get("batch") or 1)
    if batch < 1 or batch > 8:
        raise ValueError("Batch must be 1..8")
    prompt = _normalize_prompt(prompt, max(width, height))
    return {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "seed": seed,
        "batch": batch,
    }


def generate(payload, on_progress=None):
    pipe = _load_pipe(on_progress=on_progress)
    import torch

    images = []
    base_seed = payload["seed"] if payload["seed"] is not None else int(time.time() * 1000) % (2**31)
    total = payload["batch"]
    for i in range(total):
        if on_progress:
            on_progress({"stage": "generating", "current": i + 1, "total": total,
                         "message": "Generating %d/%d..." % (i + 1, total)})
        gen_device = _pipe_device if _pipe_device != "cpu" else "cpu"
        generator = torch.Generator(gen_device).manual_seed(base_seed + i)
        t0 = time.time()
        result = pipe(
            prompt=payload["prompt"],
            height=payload["height"],
            width=payload["width"],
            num_inference_steps=payload["num_inference_steps"],
            guidance_scale=payload["guidance_scale"],
            generator=generator,
        )
        ms = int((time.time() - t0) * 1000)
        img = result.images[0]
        images.append({"image": img, "ms": ms, "seed": base_seed + i})
    return images


# ---------------------------------------------------------------------------
# HTTP

def ok(data):
    return {"success": True, "data": data}


def fail(message):
    return {"success": False, "error": str(message)}


class Handler(BaseHTTPRequestHandler):
    server_version = "LiliPixelHarness/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, status=200, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload, status=200):
        self._send(payload, status=status, ctype="application/json")

    def _file(self, path):
        try:
            data = path.read_bytes()
        except Exception:
            return self._json(fail("Not found"), 404)
        ctype = "text/html; charset=utf-8"
        if path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            ctype = "text/javascript; charset=utf-8"
        elif path.suffix == ".png":
            ctype = "image/png"
        elif path.suffix == ".svg":
            ctype = "image/svg+xml"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _host_ok(self):
        name = (self.headers.get("Host") or "").strip().lower().rsplit(":", 1)[0]
        return name in ("127.0.0.1", "localhost", "::1", "")

    def _body(self, limit=16384):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > limit:
            raise ValueError("Bad request body")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Bad request body")
        return value

    def do_POST(self):
        if not self._host_ok():
            return self._json(fail("Forbidden"), 403)
        path = urllib.parse.urlparse(self.path).path.rstrip("/")

        if path == "/api/generate":
            try:
                body = self._body()
                payload = _generation_payload(body)
            except Exception as e:
                return self._json(fail(e), 200)
            if not _gen_lock.acquire(blocking=False):
                return self._json(fail("Generation already running"), 200)
            events = queue.Queue()

            def worker():
                try:
                    def progress(p):
                        events.put({"t": "progress", **p})

                    images = generate(payload, on_progress=progress)
                    (RESULTS / "images").mkdir(parents=True, exist_ok=True)
                    saved = []
                    for item in images:
                        image_id = uuid.uuid4().hex
                        out_path = RESULTS / "images" / (image_id + ".png")
                        item["image"].save(out_path, format="PNG")
                        saved.append({
                            "id": image_id,
                            "path": "/api/image/" + image_id + ".png",
                            "ms": item["ms"],
                            "seed": item["seed"],
                        })
                    events.put({"t": "done", "images": saved, "params": {
                        "prompt": payload["prompt"],
                        "width": payload["width"],
                        "height": payload["height"],
                        "steps": payload["num_inference_steps"],
                        "cfg": payload["guidance_scale"],
                        "seed": payload["seed"],
                        "batch": payload["batch"],
                    }})
                except Exception as e:
                    tb = traceback.format_exc(limit=3)
                    events.put({"t": "error", "e": str(e), "detail": tb})
                finally:
                    events.put(None)

            threading.Thread(target=worker, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                while True:
                    event = events.get()
                    if event is None:
                        break
                    self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                _gen_lock.release()
            return

        try:
            body = self._body(4096)
            if path == "/api/download":
                return self._json(ok(start_download()))
            if path == "/api/load":
                # explicit load
                if not _gen_lock.acquire(blocking=False):
                    return self._json(fail("Busy"), 200)
                try:
                    _load_pipe()
                    return self._json(ok(_status()))
                finally:
                    _gen_lock.release()
            if path == "/api/unload":
                global _pipe, _pipe_device, _pipe_dtype
                with _lock:
                    _pipe = None
                    _pipe_device = None
                    _pipe_dtype = None
                try:
                    import gc, torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return self._json(ok(_status()))
        except Exception as e:
            return self._json(fail(str(e)), 200)
        return self._json(fail("Not found"), 404)

    def do_GET(self):
        if not self._host_ok():
            return self._json(fail("Forbidden"), 403)
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/api/health":
                return self._json(ok({"status": "ok"}))
            if path == "/api/status":
                return self._json(ok(_status()))
            if path == "/api/download/status":
                return self._json(ok(_read(STATE / "download.json", {"status": "idle"})))
            if path.startswith("/api/image/"):
                raw = path[len("/api/image/"):]
                image_id = raw[:-4] if raw.endswith(".png") else raw
                if not re.fullmatch(r"[0-9a-f]{32}", image_id):
                    return self._json(fail("Invalid image id"), 400)
                target = (RESULTS / "images" / (image_id + ".png")).resolve()
                if not str(target).startswith(str((RESULTS / "images").resolve())):
                    return self._json(fail("Forbidden"), 403)
                if not target.is_file():
                    return self._json(fail("Not found"), 404)
                return self._file(target)
        except Exception as e:
            return self._json(fail(str(e)), 500)

        # static
        rel = "index.html" if path == "/" else parsed.path.lstrip("/")
        target = (WEB / rel).resolve()
        try:
            target.relative_to(WEB.resolve())
        except ValueError:
            return self._json(fail("Not found"), 404)
        if not target.is_file():
            return self._json(fail("Not found"), 404)
        return self._file(target)


def main():
    for d in (WEB, MODELS, RESULTS / "images", STATE):
        d.mkdir(parents=True, exist_ok=True)
    print("LiliPixel harness on http://127.0.0.1:%d/" % PORT)
    print("Model: %s" % MODEL_ID)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
