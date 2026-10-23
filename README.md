# D4RT demo

Open **[D4RT_simple.ipynb](D4RT_simple.ipynb)** for the presentation. It uses real
OpenD4RT predictions on the DAVIS `kid-football` soccer clip and has five steps:

1. Initialize the model and encode the video: `d = d4rt(video)`.
2. Select one `(u, v)` pixel on the ball in frame 0 and query its position in frame 31.
3. Query a grid of pixels to make a depth map.
4. Track that same ball point through all 32 frames.
5. Reconstruct a colored point cloud for every frame in a shared camera-0 reference.

Every query variable is visible in the notebook. The point example shows frame 0
and frame 31 side by side with large markers. The final player has Play/Pause,
a frame slider, a predicted point marker, and its path.
The point-cloud player adds rotation and a slider covering all 32 frames, with
2,560 newly sampled pixels and their RGB colors in each frame.

```python
d4rt = D4RT.from_pretrained(precision="float32")
d = d4rt(video)
p = d.query(u, v, t_src=0, t_tgt=31, t_cam=31)
```

## Run in the prepared workspace

```bash
.venv/bin/python -m jupyterlab D4RT_simple.ipynb
```

Select the **D4RT demo** kernel, then Run All Cells. The notebook also contains
saved outputs. Download and open `artifacts/D4RT_simple.html` for the offline
walkthrough, `artifacts/simple_point_track.html` for the one-point player, or
`artifacts/simple_point_clouds.html` for the point clouds.

The Mac browser can connect to the GPU machine's Jupyter server over an SSH tunnel;
the Python kernel and model remain on the GPU machine.

## Set up a fresh clone

Install a GPU-compatible PyTorch build using the [official selector](https://pytorch.org/get-started/locally/),
then install the notebook dependencies. A working existing CUDA environment can be used.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for your GPU and driver, then:
python -m pip install -r requirements.txt
python scripts/setup_demo.py --weights --example soccer
python -m ipykernel install --prefix .venv --name d4rt-demo --display-name "D4RT demo"
python -m jupyterlab D4RT_simple.ipynb
```

The released checkpoint is approximately 14 GB on disk. The tested environment
uses an NVIDIA RTX PRO 5000 Blackwell and PyTorch 2.11/CUDA 12.8. GPU memory needs
depend on clip length, query chunk size, and precision. `precision="auto"` uses
reduced precision to lower memory use. Source, checkpoint, environment and input
video downloads are ignored by Git.

The soccer notebook uses 32 frames at stride 2. Query indices 0–31 refer to that
sampled clip. RGB frames are resized to at most 480 pixels high for display;
OpenD4RT receives its configured 256×256 input. No ground-truth geometry, camera
parameters, or tracking annotations are passed to the model.

## Regenerate the saved presentation

```bash
python scripts/render_demo.py --simple
```

This updates `D4RT_simple.ipynb` and creates `artifacts/D4RT_simple.html` plus
`artifacts/simple_point_track.html` and `artifacts/simple_point_clouds.html`. All HTML outputs embed their assets
and work offline. They display saved real predictions without running the model.

To rebuild the notebook's source cells from its authoring script:

```bash
python scripts/build_notebook.py --simple
```

This resets that notebook's cells and saved outputs; save any edits first.

## Query conventions

`u, v` are normalized coordinates from 0 to 1 in the source image. `t_src` identifies
the frame containing that source pixel. `t_tgt` chooses when to predict its position.
`t_cam` chooses the camera coordinate system for the 3D answer. For image overlays,
the notebook uses `t_cam=t_tgt`. All time indices must lie within the supplied clip.

Results include `xyz`, `uv`, `visibility`, and `confidence_raw`. `depth` is camera Z.
Geometry is in model units, not calibrated meters. Visibility is a model score;
fast motion and occlusion can produce tracking errors.

## Verification and attribution

```bash
python -m unittest discover -s tests -v
```

Tests cover input validation, broadcasting, frame sampling, resizing, camera
coordinates, visibility, strict checkpoint loading, and reuse of encoded video.
The full presentation is also executed on the released checkpoint and its offline
player checked in Chrome.

- [Original D4RT paper and project](https://d4rt-paper.github.io/).
- [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt): unofficial implementation, Apache 2.0;
  pinned source `403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3`.
- [Released checkpoint](https://huggingface.co/Lijiaxin0111/OpenD4RT): revision
  `7099b1fc760475de3b7409acd6b63f801b015d07`.
- DAVIS `kid-football` from [Lightricks' public mirror](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset),
  revision `2b7f1b86150dac8b519ad8cf6d5d4a0351c6dc03`. The video retains its original terms.

The earlier extended notebooks (`D4RT_demo.ipynb`, `D4RT_soccer.ipynb`) and the
explicitly labeled CPU illustration (`D4RT_rehearsal.ipynb`) are still available.
