"""Behavioral checks for query semantics and the upstream encode/decode adapter."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from d4rt_demo import D4RT, Video, load_video
from d4rt_demo.illustration import make_illustration


class IllustrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video = make_illustration(n_frames=8, width=64, height=40)

    def setUp(self):
        self.scene = D4RT.illustration()(self.video)

    def test_source_identity_and_depth(self):
        u, v = np.array([.2, .32, .7]), np.array([.3, .49, .6])
        result = self.scene.query(u, v, 0, 0, 0)
        np.testing.assert_allclose(result.uv, np.stack([u, v], -1), atol=1e-6)
        self.assertTrue((result.depth > 0).all())
        self.assertTrue((result.visibility == 1).all())

    def test_broadcast_and_camera_frame(self):
        times = np.arange(8)
        fixed = self.scene.query(np.array([.2,.7])[:,None], .2, 0, times, 0)
        moving = self.scene.query(np.array([.2,.7])[:,None], .2, 0, times, times)
        self.assertEqual(fixed.xyz.shape, (2,8,3))
        # Background is stationary in a fixed reference but changes with the camera.
        np.testing.assert_allclose(fixed.xyz, np.repeat(fixed.xyz[:, :1], 8, axis=1))
        np.testing.assert_allclose(moving.xyz[:,-1,0], fixed.xyz[:,-1,0] - .6)

    def test_occlusion_does_not_erase_geometry(self):
        result = self.scene.query(.32, .49, 0, np.arange(8), 0)
        self.assertTrue(np.isfinite(result.xyz).all())
        self.assertTrue((result.visibility == 0).any())
        self.assertTrue((result.visibility == 1).any())

    def test_invalid_queries_are_rejected(self):
        invalid = [(-.1,.5,0,1,0), (.5,.5,0,8,0), (.5,.5,.2,1,0),
                   (np.nan,.5,0,1,0), ([],[],0,1,0), ([.1,.2],[.1,.2,.3],0,1,0)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.scene.query(*args)

    def test_illustration_rejects_real_video(self):
        real = Video(np.zeros((2,8,8,3),dtype=np.uint8))
        with self.assertRaises(ValueError):
            D4RT.illustration()(real)


class VideoLoadingTests(unittest.TestCase):
    def test_resize_preserves_aspect_ratio_and_sample_indices(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24, (80, 40))
            self.assertTrue(writer.isOpened())
            for i in range(8):
                writer.write(np.full((40,80,3), i*20, dtype=np.uint8))
            writer.release()
            video = load_video(path, max_frames=4, stride=2, max_height=20)
            self.assertEqual(video.frames.shape, (4,20,40,3))
            np.testing.assert_array_equal(video.original_indices, [0,2,4,6])
            self.assertEqual(video.fps, 12)
            self.assertGreater(float(video.frames[-1].mean()), float(video.frames[0].mean()))


class UpstreamAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(os.environ.get("D4RT_UPSTREAM_ROOT", ".cache/Open-d4rt")).resolve()
        if not (cls.repo / "src/model/d4rt.py").exists():
            raise unittest.SkipTest("Fetch the pinned upstream source with scripts/setup_demo.py.")
        import sys
        sys.path.insert(0, str(cls.repo))
        import torch
        import yaml
        from src.model import build_model

        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = Path(cls.tmp.name) / "model.yaml"
        cls.checkpoint = Path(cls.tmp.name) / "model.ckpt"
        cls.cfg = {"name":"d4rt", "input":{"clip_frames":4,"image_size":[32,32],"use_aspect_ratio_token":True},
                   "encoder":{"hidden_dim":32,"num_layers":2,"num_heads":4,
                              "patch_size_t_h_w":[2,16,16],"pretrained":{"enabled":False}},
                   "decoder":{"hidden_dim":32,"num_layers":1,"num_heads":4},
                   "query_embedding":{"local_rgb_patch":{"enabled":True,"patch_size":3}}}
        torch.manual_seed(0)
        # Random weights only exercise the adapter; they never appear in demo outputs.
        model = build_model(cls.cfg)
        torch.save({"model":model.state_dict()}, cls.checkpoint)
        cls.config.write_text(yaml.safe_dump({"model":cls.cfg}))
        cls.video = Video(np.random.default_rng(1).integers(0,256,(4,40,60,3),dtype=np.uint8))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load_engine(self, **kwargs):
        return D4RT.from_pretrained(repo_dir=self.repo, checkpoint=self.checkpoint,
                                    config=self.config, device="cpu", **kwargs)

    def test_chunking_broadcast_independence_and_encode_cache(self):
        engine = self.load_engine(chunk_size=2)
        encode_calls = []
        encode = engine.model.encode_video
        def counted(*args, **kwargs):
            encode_calls.append(1)
            return encode(*args, **kwargs)
        engine.model.encode_video = counted
        scene = engine(self.video)
        batch = scene.query(np.array([.2,.5])[:,None], .4, 0, np.arange(4), 0)
        alone = scene.query(.2,.4,0,3,0)
        np.testing.assert_allclose(alone.xyz, batch.xyz[0,3], atol=1e-5)
        self.assertEqual(batch.xyz.shape, (2,4,3))
        self.assertEqual(batch.visibility.shape, (2,4))
        self.assertTrue(((batch.visibility >= 0) & (batch.visibility <= 1)).all())
        self.assertEqual(len(encode_calls), 1)
        self.assertEqual(scene.stats["queries"], 9)

    def test_meta_loading_restores_nonpersistent_buffer(self):
        engine = self.load_engine()
        self.assertTrue(all(not x.is_meta for x in engine.model.buffers()))

    def test_strict_loading_rejects_incomplete_checkpoint(self):
        import torch
        bad = Path(self.tmp.name) / "incomplete.ckpt"
        torch.save({"model":{"bogus":torch.ones(1)}},bad)
        with self.assertRaises(RuntimeError):
            D4RT.from_pretrained(repo_dir=self.repo, checkpoint=bad, config=self.config, device="cpu")

    def test_rejects_clip_longer_than_temporal_embeddings(self):
        engine = self.load_engine()
        with self.assertRaises(ValueError):
            engine(Video(np.zeros((5,32,32,3),dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
