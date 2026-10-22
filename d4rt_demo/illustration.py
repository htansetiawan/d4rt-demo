"""Analytic scene for teaching the query semantics without a GPU.

This is a geometric oracle, not a model, a prediction, or a quality benchmark.
Source pixels are ray-cast against a moving sphere, a fixed occluder, and a wall.
"""

import numpy as np

from .api import QueryResult, Video, _queries


def _camera(t, n):
    a = np.asarray(t) / max(n - 1, 1)
    return np.stack([0.6 * a, 0.08 * np.sin(2 * np.pi * a), np.zeros_like(a)], -1)


def _center(t, n):
    a = np.asarray(t) / max(n - 1, 1)
    return np.stack([-0.85 + 1.7 * a, -0.12 * np.sin(2 * np.pi * a),
                     4.0 - 0.45 * np.sin(np.pi * a)], -1)


def _project(p, t, n):
    q = p - _camera(t, n)
    return np.stack([0.5 + 0.85 * q[..., 0] / q[..., 2],
                     0.5 + 1.36 * q[..., 1] / q[..., 2]], -1)


def _surface(u, v, t, n):
    origin = _camera(t, n)
    ray = np.stack([(u - 0.5) / 0.85, (v - 0.5) / 1.36, np.ones_like(u)], -1)
    # Back wall at z=6.
    lam = np.full(np.shape(u), 6.0)
    identity = np.zeros(np.shape(u), dtype=int)
    # Foreground strip stays in front of the sphere throughout its trajectory.
    strip_p = origin + 2.6 * ray
    hit = (np.abs(strip_p[..., 0] - 0.17) < 0.16) & (np.abs(strip_p[..., 1]) < 0.95)
    lam = np.where(hit, 2.6, lam)
    identity = np.where(hit, 2, identity)
    # A translated sphere, radius 0.63.
    offset = origin - _center(t, n)
    aa = (ray * ray).sum(-1)
    bb = 2 * (ray * offset).sum(-1)
    cc = (offset * offset).sum(-1) - 0.63 ** 2
    disc = bb * bb - 4 * aa * cc
    root = (-bb - np.sqrt(np.maximum(disc, 0))) / (2 * aa)
    sphere = (disc >= 0) & (root > 0) & (root < lam)
    lam, identity = np.where(sphere, root, lam), np.where(sphere, 1, identity)
    return origin + lam[..., None] * ray, identity


def _color(p, identity, t, n):
    grid = (np.floor((p[..., 0] + 5) * 2) + np.floor((p[..., 1] + 5) * 2)) % 2
    color = np.stack([30 + 14 * grid, 42 + 17 * grid, 66 + 20 * grid], -1)
    local = (p - _center(t, n)) / 0.63
    stripes = (np.floor((local[..., 0] + 1) * 6) % 2)[..., None]
    sphere = np.array([28, 210, 190]) * (1 - 0.28 * stripes)
    shade = np.clip(0.65 - 0.28 * local[..., 1] - 0.25 * local[..., 2], 0.25, 1)[..., None]
    sphere = sphere * shade
    color = np.where((identity == 1)[..., None], sphere, color)
    color = np.where((identity == 2)[..., None], [243, 151, 69], color)
    return np.clip(color, 0, 255).astype(np.uint8)


def make_illustration(n_frames=32, width=480, height=300):
    u, v = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    frames = []
    for t in range(n_frames):
        ts = np.full(u.shape, t)
        p, ids = _surface(u, v, ts, n_frames)
        frames.append(_color(p, ids, ts, n_frames))
    video = Video(np.stack(frames), 12.0, "geometric illustration")
    video._illustration_scene = True
    return video


class IllustrationScene:
    backend = "illustration"

    def __init__(self, video):
        if not getattr(video, "_illustration_scene", False):
            raise ValueError("The illustration only accepts make_illustration() videos; it cannot infer real video.")
        self.video = video
        self.stats = {"encoder_calls": 0, "query_calls": 0, "queries": 0,
                      "encode_seconds": 0.0, "decode_seconds": 0.0}

    def query(self, u, v, t_src, t_tgt, t_cam):
        shape, q = _queries(u, v, t_src, t_tgt, t_cam, len(self.video))
        n = len(self.video)
        p, identity = _surface(q["u"], q["v"], q["t_src"], n)
        motion = _center(q["t_tgt"], n) - _center(q["t_src"], n)
        p = p + (identity == 1)[..., None] * motion
        uv = _project(p, q["t_tgt"], n)
        nearest, _ = _surface(uv[..., 0], uv[..., 1], q["t_tgt"], n)
        visible = ((np.linalg.norm(p - nearest, axis=-1) < 0.03)
                   & (uv >= 0).all(-1) & (uv <= 1).all(-1))
        xyz = p - _camera(q["t_cam"], n)
        self.stats["query_calls"] += 1
        self.stats["queries"] += len(q["u"])
        return QueryResult(xyz.reshape(shape + (3,)), uv.reshape(shape + (2,)),
                           visible.astype(float).reshape(shape),
                           np.full(shape, np.nan), self.backend)
