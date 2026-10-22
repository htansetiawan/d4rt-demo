"""Normalized UV coordinates, zero-based clip indices, NumPy results.

The adapter caches the encoder output and chunks independent decoder queries.
No trained weights are bundled, and model failures never select an illustration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import copy
import importlib
import sys

import numpy as np


@dataclass
class Video:
    frames: np.ndarray  # uint8 RGB [T,H,W,3]
    fps: float = 12.0
    name: str = "video"
    original_indices: np.ndarray | None = None

    def __post_init__(self):
        self.frames = np.asarray(self.frames)
        if (self.frames.ndim != 4 or self.frames.shape[-1] != 3
                or min(self.frames.shape[:3]) < 1 or self.frames.dtype != np.uint8):
            raise ValueError("frames must be nonempty uint8 RGB [T,H,W,3].")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite.")

    def __len__(self):
        return len(self.frames)


def load_video(path, *, max_frames=32, stride=1) -> Video:
    """Read the first clip from a video or an OpenD4RT WorldTrack NPZ.

    NPZ loading only accesses JPEG images, never trajectories or depth labels.
    WorldTrack's object arrays require pickle; use only trusted dataset files.
    Indices in query() refer to the sampled clip, not the original video.
    """
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts/setup_demo.py --example first.")
    if int(max_frames) != max_frames or max_frames < 1 or int(stride) != stride or stride < 1:
        raise ValueError("max_frames and stride must be positive integers.")
    max_frames, stride = int(max_frames), int(stride)
    frames, ids = [], []
    fps = 12.0
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as pack:
            if "images_jpeg_bytes" not in pack:
                raise ValueError("Expected a WorldTrack NPZ with images_jpeg_bytes.")
            for i in range(0, min(len(pack["images_jpeg_bytes"]), max_frames * stride), stride):
                raw = np.frombuffer(pack["images_jpeg_bytes"][i], dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Could not decode WorldTrack frame {i}.")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ids.append(i)
    else:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open {path}.")
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 12.0
            for i in range(max_frames * stride):
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    ids.append(i)
        finally:
            cap.release()
    if len(frames) < 2:
        raise ValueError("Choose a clip with at least two frames.")
    return Video(np.stack(frames), fps / stride, path.stem, np.asarray(ids))


def pixel_grid(cols=36, rows=24, margin=0.05):
    """Return normalized UV arrays [rows,cols]."""
    if cols < 1 or rows < 1 or not 0 <= margin < 0.5:
        raise ValueError("Use positive grid sizes and 0 <= margin < 0.5.")
    return np.meshgrid(np.linspace(margin, 1 - margin, cols),
                       np.linspace(margin, 1 - margin, rows))


@dataclass(frozen=True)
class QueryResult:
    xyz: np.ndarray
    uv: np.ndarray
    visibility: np.ndarray  # sigmoid of upstream visibility logits
    confidence_raw: np.ndarray  # upstream raw head, not a probability
    backend: str

    @property
    def depth(self):
        """Camera Z; interpretable as depth when src == tgt == cam."""
        return self.xyz[..., 2]


def _queries(u, v, t_src, t_tgt, t_cam, n_frames):
    try:
        arrays = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64)
                                      for x in (u, v, t_src, t_tgt, t_cam)])
    except (TypeError, ValueError) as e:
        raise ValueError("Query fields must be numeric and NumPy-broadcastable.") from e
    if any(a.size == 0 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Query fields must be nonempty and finite.")
    if any(((a < 0) | (a > 1)).any() for a in arrays[:2]):
        raise ValueError("u and v are normalized coordinates in [0,1].")
    for name, a in zip(("t_src", "t_tgt", "t_cam"), arrays[2:]):
        if ((a != np.floor(a)) | (a < 0) | (a >= n_frames)).any():
            raise ValueError(f"{name} must contain integer clip indices in [0,{n_frames-1}].")
    return arrays[0].shape, dict(zip(("u", "v", "t_src", "t_tgt", "t_cam"),
                                   [a.reshape(-1) for a in arrays]))


class D4RT:
    """Configure once; call with a video to create a queryable scene.

    d4rt = D4RT.from_pretrained(repo_dir=..., checkpoint=..., config=...)
    d = d4rt(video)
    result = d.query(u, v, t_src, t_tgt, t_cam)
    """

    def __init__(self, *, model=None, model_config=None, device="cpu", chunk_size=64,
                 backend="opend4rt", precision="float32"):
        if backend not in ("opend4rt", "illustration"):
            raise ValueError("backend must be 'opend4rt' or 'illustration'.")
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if backend == "opend4rt" and model is None:
            raise ValueError("Use D4RT.from_pretrained() to load the real model.")
        self.model, self.config, self.device = model, model_config, device
        self.chunk_size, self.backend, self.precision = int(chunk_size), backend, precision

    @classmethod
    def illustration(cls):
        """Explicit geometry-only illustration. No learned inference."""
        return cls(backend="illustration")

    @classmethod
    def from_pretrained(cls, *, repo_dir, checkpoint, config, device="cuda",
                        precision="auto", chunk_size=64):
        import torch
        import yaml

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select the GPU notebook kernel.")
        if precision == "auto":
            precision = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if device.startswith("cuda") else "float32"
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("precision must be auto, float32, float16, or bfloat16.")
        if device == "cpu" and precision != "float32":
            raise ValueError("Use float32 for CPU inference.")
        for path in (Path(repo_dir) / "src/model/d4rt.py", Path(checkpoint), Path(config)):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run scripts/setup_demo.py --weights.")
        repo_dir = Path(repo_dir).resolve()
        sys.path.insert(0, str(repo_dir))
        upstream = importlib.import_module("src.model")
        if not Path(upstream.__file__).resolve().is_relative_to(repo_dir):
            raise ImportError("Another package named src is loaded. Restart the notebook kernel.")
        cfg = copy.deepcopy(yaml.safe_load(Path(config).read_text())["model"])
        cfg.setdefault("encoder", {})["pretrained"] = {"enabled": False}

        # Avoid allocating a second 14 GB parameter copy on the host.
        with torch.device("meta"):
            model = upstream.build_model(cfg)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = payload
        for key in ("state_dict", "model", "module", "network", "net"):
            if isinstance(payload, dict) and isinstance(payload.get(key), dict):
                state = payload[key]
                break
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Checkpoint does not contain a tensor state dictionary.")
        if all(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True, assign=True)
        # Upstream's one nonpersistent buffer is absent from the state dict.
        uv = model.query_embedder.uv_encoder
        uv.frequencies = 2.0 ** torch.arange(uv.num_bands, dtype=torch.float32) * torch.pi
        remaining = [n for n, b in model.named_buffers() if b.is_meta]
        if remaining:
            raise RuntimeError(f"Uninitialized upstream buffers: {remaining}")
        # Upstream adds fp32 positional features to encoder activations.
        # Keep LayerNorm parameters fp32 to accept those residuals; autocast
        # handles linear/attention operations with reduced-precision weights.
        model = model.to(device=device, dtype=getattr(torch, precision)).eval()
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.float()
        return cls(model=model, model_config=cfg, device=device, chunk_size=chunk_size,
                   precision=precision)

    def __call__(self, video: Video):
        if not isinstance(video, Video):
            raise TypeError("Pass a Video returned by load_video() or make_illustration().")
        if self.backend == "illustration":
            from .illustration import IllustrationScene
            return IllustrationScene(video)
        return Scene(self, video)


class Scene:
    backend = "opend4rt"

    def __init__(self, engine, video):
        import cv2
        import torch

        max_frames = int(engine.config["input"]["clip_frames"])
        if not 2 <= len(video) <= max_frames:
            raise ValueError(f"Use 2..{max_frames} frames for this checkpoint; got {len(video)}.")
        self.engine, self.video = engine, video
        self.stats = {"encoder_calls": 0, "query_calls": 0, "queries": 0, "decode_seconds": 0.0}
        h, w = engine.config["input"]["image_size"]
        # Match upstream's RGB /255, resize, native aspect-ratio input.
        frames = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                           for f in video.frames])
        self._video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
        self._video = self._video.to(device=engine.device, dtype=torch.float32) / 255.0
        aspect = torch.tensor([[video.frames.shape[2] / video.frames.shape[1]]],
                              dtype=torch.float32, device=engine.device)
        self._sync()
        start = perf_counter()
        with torch.inference_mode(), self._autocast():
            self._memory = engine.model.encode_video(video=self._video, aspect_ratio=aspect)
        self._sync()
        self.stats.update(encoder_calls=1, encode_seconds=perf_counter() - start)

    def _autocast(self):
        import torch
        from contextlib import nullcontext
        if self.engine.device.startswith("cuda") and self.engine.precision != "float32":
            return torch.autocast("cuda", dtype=getattr(torch, self.engine.precision))
        return nullcontext()

    def _sync(self):
        import torch
        if self.engine.device.startswith("cuda"):
            torch.cuda.synchronize(self.engine.device)

    def query(self, u, v, t_src, t_tgt, t_cam):
        """Predict point(s) at t_tgt, expressed in camera t_cam coordinates.

        All arguments broadcast. xyz.shape == broadcast_shape + (3,).
        uv and visibility describe the target frame (use t_cam=t_tgt for
        image overlays, matching upstream's tracking inference).
        """
        import torch

        shape, q = _queries(u, v, t_src, t_tgt, t_cam, len(self.video))
        outputs = {}
        self._sync()
        start = perf_counter()
        with torch.inference_mode(), self._autocast():
            for start_i in range(0, len(q["u"]), self.engine.chunk_size):
                chunk = {k: torch.as_tensor(a[start_i:start_i+self.engine.chunk_size].copy(),
                                           device=self.engine.device,
                                           dtype=torch.float32 if k in ("u", "v") else torch.long)[None]
                         for k, a in q.items()}
                pred = self.engine.model.decode_queries(video=self._video, query=chunk, memory=self._memory)
                for key in ("xyz_3d", "uv_2d", "visibility", "confidence"):
                    value = pred[key][0]
                    if key == "visibility":
                        value = value.sigmoid()
                    outputs.setdefault(key, []).append(value.float().cpu().numpy())
        self._sync()
        self.stats["query_calls"] += 1
        self.stats["queries"] += len(q["u"])
        self.stats["decode_seconds"] += perf_counter() - start
        arrays = {k: np.concatenate(v).reshape(shape + (() if k in ("visibility", "confidence") else (3 if k == "xyz_3d" else 2,)))
                  for k, v in outputs.items()}
        return QueryResult(arrays["xyz_3d"], arrays["uv_2d"], arrays["visibility"],
                           arrays["confidence"], self.backend)
