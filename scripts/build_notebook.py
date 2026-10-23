"""Author the presentation notebooks; execution is a separate, explicit step."""

from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def make_notebook(illustration=False, soccer=False):
    if illustration and soccer:
        raise ValueError("The soccer demo uses real OpenD4RT inference.")
    video_path = "data/davis/kid-football.mp4" if soccer else "data/worldtrack/pstudio_mini/juggle_5.npz"
    example = "kid-football" if soccer else "juggle_5"
    source_uv = [.29, .18] if soccer else [.62, .65]
    anchors = ([[.29,.18], [.61,.65], [.52,.13], [.48,.85], [.18,.65], [.88,.80]] if soccer
               else [[.62,.65], [.63,.43], [.56,.86], [.597,.475], [.23,.48], [.80,.30]])
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text, *, setup=False):
        cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
        cell.metadata["slideshow"] = {"slide_type": "skip" if setup else "fragment"}
        if setup:
            cell.metadata["jupyter"] = {"source_hidden": True}
            cell.metadata["tags"] = ["hide-input"]
        cells.append(cell)

    md("""
    # D4RT
    ### A video becomes a queryable 4D scene.

    ```python
    d = d4rt(video)
    p = d.query(u, v, t_src, t_tgt, t_cam)
    ```

    **Choose a pixel. Choose a moment. Choose a camera.**

    A 5–7 minute walkthrough of depth, correspondence, 3D motion, and reference frames
    through one interface. This small Python wrapper uses
    [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt), an unofficial implementation of
    [D4RT](https://d4rt-paper.github.io/).
    """)
    if illustration:
        md("""
        > **CPU REHEARSAL — ANALYTIC ILLUSTRATION.** All geometry in this notebook is
        > calculated from a known moving sphere, wall, and occluder. These are **not
        > OpenD4RT predictions**. Use `D4RT_demo.ipynb` for the real GPU demonstration
        > on the OpenD4RT `juggle_5` sequence.
        """)
    else:
        md(f"""
        > **Live GPU demo.** Before opening this notebook, run the setup commands in
        > `README.md` to fetch OpenD4RT, its released 32-frame checkpoint, and the
        > `{example}` example. This notebook does not download weights while presenting.
        > Run all cells once before the talk and save the outputs. A separate executed
        > `D4RT_rehearsal.ipynb` teaches the same API with explicitly labeled analytic geometry.
        """)
    if soccer:
        md("""
        > **Soccer edition.** DAVIS `kid-football`: a large foreground ball, a moving player,
        > and a camera that pulls back. Watch the ball's visibility as it passes the player,
        > and compare a background point in fixed versus moving camera coordinates.
        > This is RGB-only inference on a DAVIS clip, separate from the WorldTrack example.
        """)
    code("""
    from pathlib import Path
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    ROOT = Path.cwd()
    if not (ROOT / "d4rt_demo").is_dir():
        ROOT = ROOT.parent
    if not (ROOT / "d4rt_demo").is_dir():
        raise RuntimeError("Start Jupyter from the d4rt project directory.")
    sys.path.insert(0, str(ROOT))
    import os
    os.chdir(ROOT)
    from d4rt_demo import D4RT, load_video, pixel_grid
    from d4rt_demo import visuals as viz
    viz.style()
    display(HTML('''<style>
      .jp-RenderedHTMLCommon h1 {font-size:52px; letter-spacing:-2px; color:#20bfa9}
      .jp-RenderedHTMLCommon h2 {font-size:30px; letter-spacing:-.5px}
      .jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon li {font-size:17px; line-height:1.65}
      .jp-RenderedHTMLCommon blockquote {border-left:4px solid #20bfa9; padding:8px 20px}
      .jp-RenderedHTMLCommon pre {border-radius:10px; padding:20px}
    </style>'''))
    """, setup=True)
    code(f'''
    # Configuration — prepare this cell before the presentation.
    BACKEND = {"'illustration'" if illustration else "'opend4rt'"}
    RUN_NAME = {"'soccer'" if soccer else "BACKEND"}
    VIDEO_PATH = ROOT / {video_path!r}
    CHECKPOINT_DIR = ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG"
    N_FRAMES = 32
    STRIDE = 2  # Every other original frame; query indices always refer to this sampled clip.

    if BACKEND == "opend4rt":
        video = load_video(VIDEO_PATH, max_frames=N_FRAMES, stride=STRIDE, max_height=480)
        d4rt = D4RT.from_pretrained(
            repo_dir=ROOT / ".cache/Open-d4rt",
            checkpoint=CHECKPOINT_DIR / "opend4rt.ckpt",
            config=CHECKPOINT_DIR / "model.yaml",
            device="cuda", precision="float32", chunk_size=64,  # auto trades precision for lower VRAM.
        )
        source_uv = np.array({source_uv!r})  # Ball in kid-football; shirt patch in juggle_5.
        anchors = np.array({anchors!r})
    elif BACKEND == "illustration":
        from d4rt_demo.illustration import make_illustration
        video = make_illustration(n_frames=N_FRAMES)
        d4rt = D4RT.illustration()
        source_uv = np.array([.32, .49])
        anchors = np.array([[.32,.49], [.36,.55], [.28,.45], [.55,.5], [.75,.25], [.18,.72]])
    else:
        raise ValueError("Select opend4rt or illustration explicitly.")

    print(f"{{BACKEND}} | {{video.name}} | {{len(video)}} frames | RGB {{video.frames.shape[2]}}×{{video.frames.shape[1]}}")
    ''', setup=True)
    md("""
    ## 01 · Encode once. Ask many questions.

    The encoder processes the clip into a global scene representation.
    Each subsequent query uses those cached features and the source pixel's local RGB patch.

    *Say: “The expensive part happens here. From now on, I’m asking questions about the same video.”*
    """)
    code("""
    d = d4rt(video)
    viz.banner(d)
    viz.video_player(video)
    """)
    md("""
    ## 02 · The whole interface

    ```python
    d.query(u, v, t_src, t_tgt, t_cam)
    ```

    | Argument | Question |
    |:--|:--|
    | `u, v` | Which source pixel? Normalized coordinates in `[0, 1]`. |
    | `t_src` | In which frame do I identify that physical point? |
    | `t_tgt` | At which frame do I want its position? |
    | `t_cam` | In which frame's camera coordinates should the 3D answer be expressed? |

    All frame indices are **zero based** and refer to the sampled input clip.
    The result contains `xyz`, target-frame `uv`, a visibility score, and a raw confidence head.
    `xyz` uses the model's learned scale; it is not a calibrated distance measurement in meters.

    *Say: “Source time identifies the point. Target time chooses its state. Camera time chooses the coordinate system.”*
    """)
    code("""
    u, v = source_uv
    t_src, t_tgt, t_cam = 0, len(video) - 1, 0

    p = d.query(u, v, t_src, t_tgt, t_cam)
    print("3D position:", np.round(p.xyz, 4))
    print("Visibility:", round(float(p.visibility), 3))
    # Use the target camera to obtain the predicted pixel for the target image.
    p_image = d.query(u, v, t_src, t_tgt, t_cam=t_tgt)
    viz.filmstrip(video, source_uv, source=t_src, prediction=p_image, target=t_tgt)
    """)
    md("""
    ## 03 · Depth is one query pattern

    Ask about each pixel **at its own time, in its own camera**. Keep the Z coordinate.
    We sample a grid for speed; increasing its resolution asks more of the same queries.

    *Say: “There isn’t a new depth model here. I’ve only changed the arguments.”*
    """)
    code("""
    U, V = pixel_grid(cols=64, rows=40, margin=0.0)
    t = 0
    surface = d.query(U, V, t_src=t, t_tgt=t, t_cam=t)
    depth = surface.depth

    viz.depth_view(video, surface, t)
    """)
    md("""
    ## 04 · A point becomes a 3D trajectory

    Hold the source point fixed. Sweep target time. Hold camera 0 as the shared reference.
    Scalar and array arguments broadcast just like NumPy, so a batch of points and times is one call.

    For 2D overlays, query again with `t_cam=t_tgt`, matching OpenD4RT's local-camera inference.
    Visibility is the sigmoid of the model's visibility logit; it is not a guarantee of correctness.

    *Say: “It can return a position even when the point is occluded. The visibility output tells us whether it is seen.”*
    """)
    code("""
    times = np.arange(len(video))
    u = anchors[:, 0, None]
    v = anchors[:, 1, None]

    tracks = d.query(u, v, t_src=0, t_tgt=times, t_cam=0)
    local = d.query(u, v, t_src=0, t_tgt=times, t_cam=times)

    print("points × times × xyz:", tracks.xyz.shape)
    viz.track_view(video, anchors, tracks, local, times)
    """)
    md("""
    ## 05 · Scene time and camera time are independent

    The same trajectory can be expressed in camera 0 coordinates, or in each target camera's coordinates.
    A fixed reference makes the object's trajectory comparable across time. A moving camera reference also
    reflects camera motion. Camera 0 is a convenient reference, not an externally calibrated world frame.

    These are direct predictions in the requested frame, so this wrapper does not estimate camera poses
    and transform the 3D answer afterward. Independently decoded coordinates can have consistency errors.

    *Say: “This is the subtle part: when does the point exist, and from whose camera do I describe it?”*
    """)
    code("""
    # tracks and local differ only in t_cam. Compare a background point.
    viz.camera_view(tracks, local, times, point=4)

    # Freeze the physical instant; change only the reference camera.
    t = len(video) // 2
    camera_sweep = d.query(*source_uv, t_src=0, t_tgt=t, t_cam=times)
    print("Same physical instant expressed in", len(camera_sweep.xyz), "camera frames.")
    print("First and last reference:", np.round(camera_sweep.xyz[[0, -1]], 3))
    """)
    md("""
    ## 06 · Scale the same idea to a 4D surface

    Track a grid of source pixels through time in one shared reference. Drag to rotate, then move the time slider.
    Colors come from the source RGB pixels. Low-visibility points are shown faintly.

    This is a **sampled first-frame surface**, not a complete reconstruction of surfaces hidden in that frame.
    Tracking all pixels is the same query pattern at a larger sampling budget.
    The player below uses saved predictions; rotating and scrubbing it require no GPU.

    *Say: “The data structure didn’t change. We asked more questions and got a surface moving through time.”*
    """)
    code("""
    U, V = pixel_grid(cols=36, rows=24)
    cloud_times = np.unique(np.linspace(0, len(video) - 1, 16, dtype=int))
    cloud = d.query(
        U.reshape(-1, 1), V.reshape(-1, 1),
        t_src=0, t_tgt=cloud_times, t_cam=0,
    )
    colors = viz.sample_colors(video, U, V)
    player_path = viz.scene_player(cloud, colors, cloud_times,
                                   path=f"artifacts/{RUN_NAME}_4d_scene.html")
    print("Standalone offline player:", player_path)
    """)
    md("""
    ## 07 · Random access, including backward in time

    Source time does not have to be frame 0. We can identify a point in a later frame and ask about an earlier one.
    Target frames must remain within the supplied clip: this notebook demonstrates reconstruction from the full
    observed video, not causal forecasting beyond it.

    The decoder handles queries independently. Changing a query's order or its batch companions should not
    materially change its answer; minor floating point differences are expected.
    """)
    code("""
    backward = d.query(.62, .65, t_src=len(video) - 1, t_tgt=0, t_cam=0)
    print("Later-frame point, earlier-time position:", np.round(backward.xyz, 4))

    # Demonstrate that the same encoded scene supports out-of-order queries.
    query_times = np.array([len(video)-1, 0, len(video)//2])
    batch = d.query(*source_uv, t_src=0, t_tgt=query_times, t_cam=0)
    alone = d.query(*source_uv, t_src=0, t_tgt=int(query_times[0]), t_cam=0)
    print("Single vs batch max |Δxyz|:", float(np.max(np.abs(alone.xyz - batch.xyz[0]))))
    print("Encoder calls:", d.stats["encoder_calls"], "| Queries answered:", d.stats["queries"])
    if d.backend == "opend4rt":
        print("Measured encode seconds:", round(d.stats["encode_seconds"], 2))
        print("Measured decode seconds:", round(d.stats["decode_seconds"], 2))
    """)
    md("""
    ## The takeaway

    | What we want | What we vary |
    |:--|:--|
    | Depth | Sweep pixels; `t_src = t_tgt = t_cam`; read Z |
    | 3D tracking | Fix a source point; sweep `t_tgt` |
    | A common reference frame | Hold `t_cam` fixed |
    | A moving camera reference | Set `t_cam = t_tgt` |
    | 4D surface tracking | Sweep pixels **and** target time |

    **One encoded video. Independently decoded queries. Several geometric tasks.**

    D4RT's distinctive design is the queryable 3D representation and its separation of source, target,
    and camera time. This demo illustrates that design; it does not establish a quality or speed comparison
    with other methods. OpenD4RT and the original D4RT model have different checkpoints and reported results.
    """)
    md("""
    ## Optional · Ask your own question

    Adjust the pixel and three time indices, then press **Query this point**.
    This section needs a running Python kernel; the saved 4D player above also works offline.
    """)
    code("""
    viz.query_explorer(d, *source_uv)
    """)
    md("""
    ## Optional · Save tonight's results

    Run this before presenting to create a portable player and a NumPy archive of the queries shown above.
    The archive records its backend so illustration data cannot be mistaken for model inference.
    """)
    code("""
    import json
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    metadata = {
        "backend": d.backend, "video": video.name, "frames": len(video),
        "source_commit": "403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3",
        "checkpoint_revision": "7099b1fc760475de3b7409acd6b63f801b015d07" if d.backend == "opend4rt" else None,
        "original_indices": video.original_indices.tolist() if video.original_indices is not None else None,
        "stats": d.stats,
    }
    np.savez_compressed(output / f"{RUN_NAME}_results.npz",
        metadata=json.dumps(metadata), frames=video.frames,
        depth=depth, anchors=anchors, times=times,
        tracks_xyz=tracks.xyz, local_xyz=local.xyz, tracks_uv=local.uv,
        visibility=local.visibility, cloud_xyz=cloud.xyz,
        cloud_visibility=cloud.visibility, cloud_times=cloud_times, colors=colors)
    (output / f"{RUN_NAME}_provenance.json").write_text(json.dumps(metadata, indent=2))
    print("Saved", output / f"{RUN_NAME}_results.npz")
    """, setup=True)
    md("""
    ### Sources and attribution

    - [Original D4RT project and paper](https://d4rt-paper.github.io/): the query formulation and task patterns.
    - [OpenD4RT repository](https://github.com/Lijiaxin0111/Open-d4rt): unofficial PyTorch model and inference conventions.
    - [Released checkpoint](https://huggingface.co/Lijiaxin0111/OpenD4RT): 32CLIP / 9Dataset / NoAUG.
    - [WorldTrack release linked by OpenD4RT](https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV):
      `pstudio_mini/juggle_5.npz`. Only its JPEG images are passed to the model; no ground-truth geometry is used.

    This notebook's convenience API is local demo code. The original paper, third-party implementation,
    checkpoints, and dataset retain their respective licenses and terms.
    """)
    nb = nbf.v4.new_notebook(cells=cells)
    if not soccer and not illustration:
        for cell in nb.cells:
            if cell.cell_type == "markdown" and cell.source.startswith("## 05"):
                cell.source = cell.source.replace(
                    "These are direct predictions",
                    "The `juggle_5` capture has a largely fixed camera, so both references may look similar here. "
                    "The soccer edition uses a moving camera for a clearer comparison.\n\nThese are direct predictions")
    if soccer:
        nb.cells[0].source = nb.cells[0].source.replace("# D4RT\n", "# D4RT · Soccer\n", 1)
        nb.cells[-1].source = nb.cells[-1].source.replace(
            "- [WorldTrack release linked by OpenD4RT](https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV):\n  `pstudio_mini/juggle_5.npz`. Only its JPEG images are passed to the model; no ground-truth geometry is used.",
            "- [DAVIS dataset](https://davischallenge.org/davis2017/code.html): `kid-football`.\n"
            "- [Lightricks' DAVIS video mirror](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset): "
            "revision `2b7f1b86150dac8b519ad8cf6d5d4a0351c6dc03`, `kid-football.mp4`. "
            "Only RGB images are model input; no masks, depth, or ground-truth tracks are used.")
    nb.metadata.update(kernelspec={"display_name":"Python 3", "language":"python", "name":"python3"},
                       language_info={"name":"python", "version":"3.11"})
    for c in cells:
        if c.cell_type == "markdown":
            c.metadata["slideshow"] = {"slide_type":"slide" if c.source.startswith("#") else "notes"}
    return nb


def make_simple_notebook():
    """Simple presentation steps with explicit queries and no hidden setup."""
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md("""
    # D4RT · Encode once, query points

    A soccer video, one selected point, and one interface:

    ```python
    d = d4rt(video)
    p = d.query(u, v, t_src, t_tgt, t_cam)
    ```

    We will query one point, make a depth map, track that same point over time,
    then reconstruct a point cloud for every frame.
    """)
    md("""
    ## 1 · Initialize and encode

    Load the trained model and the soccer clip. We use **32 frames, numbered 0–31**.
    """)
    code("""
    import numpy as np
    from d4rt_demo import D4RT, load_video, pixel_grid
    from d4rt_demo import visuals as viz

    viz.style()
    d4rt = D4RT.from_pretrained(precision="float32")
    video = load_video("data/davis/kid-football.mp4", stride=2, max_height=480)
    """)
    md("""
    **Encode the video once.** The queries below reuse this encoded video.
    """)
    code("""
    d = d4rt(video)
    viz.video_player(video)
    """)
    md("""
    ## 2 · Pick a point in frame 0

    Choose a pixel on the ball: **`u = 0.29`, `v = 0.18`**.
    Coordinates run from 0 to 1: `u` goes left to right; `v` goes top to bottom.

    Ask: **“Where is this point in frame 31?”**

    | Argument | Value | Meaning |
    |:--|:--|:--|
    | `u, v` | `0.29, 0.18` | Select the ball pixel in the source frame |
    | `t_src` | `0` | Identify the point in frame 0 |
    | `t_tgt` | `31` | Predict its position in frame 31 |
    | `t_cam` | `31` | Express the answer in frame 31's camera coordinates |

    The yellow marker shows the selected point; the teal marker shows the predicted position.
    """)
    code("""
    u, v = 0.29, 0.18

    p = d.query(u, v, t_src=0, t_tgt=31, t_cam=31)

    print("3D position (X, Y, Z):", np.round(p.xyz, 3))
    print("Visibility:", round(float(p.visibility), 2))
    viz.point_view(video, (u, v), p, source=0, target=31)
    """)
    md("""
    ## 3 · A depth map is a grid of point queries

    Ask about **many pixels in frame 0**, at time 0, in camera 0.
    The **Z coordinate** of each 3D answer gives its depth.
    Here we sample a 64 × 40 grid, or **2,560 pixels**.
    """)
    code("""
    U, V = pixel_grid(cols=64, rows=40, margin=0)
    surface = d.query(U, V, t_src=0, t_tgt=0, t_cam=0)
    depth = surface.depth  # Z coordinate at each grid pixel

    viz.depth_view(video, surface)
    """)
    md("""
    ## 4 · Track the same point over time

    Keep the **same `(u, v)` from frame 0**. Ask for every target frame instead of just frame 31.
    We use each target frame's camera so its predicted pixel can be drawn on that image.

    This returns **32 positions for one point**. Press Play or scrub the slider to follow it.
    The visibility score can drop when tracking becomes difficult or the point is occluded.
    """)
    code("""
    times = np.arange(len(video))  # 0, 1, ..., 31
    track = d.query(u, v, t_src=0, t_tgt=times, t_cam=times)

    print("32 positions × 3 coordinates:", track.xyz.shape)
    viz.point_track_player(video, track)
    """)
    md("""
    ## 5 · Point clouds for all frames

    Ask about a grid of pixels in **each frame**, at that same time.
    Keep `t_cam=0` so all the clouds use the same 3D coordinate system.

    Each cloud has **2,560 sampled points**, colored from its own RGB frame.
    Press Play or scrub frames **0–31**. Drag to rotate the cloud.
    These are the surfaces visible in each frame, not a complete model of hidden surfaces.
    """)
    code("""
    U, V = pixel_grid(cols=64, rows=40, margin=0)
    clouds, colors = [], []

    for t in range(len(video)):
        clouds.append(d.query(U, V, t_src=t, t_tgt=t, t_cam=0))
        colors.append(viz.sample_colors(video, U, V, t_src=t))

    viz.point_cloud_player(clouds, colors)
    """)
    md("""
    *Real predictions from [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt),
    an unofficial implementation of [D4RT](https://d4rt-paper.github.io/).
    Video: DAVIS `kid-football`, from [Lightricks' public mirror](https://huggingface.co/datasets/Lightricks/Canny-Control-Dataset).
    Geometry is in model units; visibility is a model score.*
    """)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(kernelspec={"display_name":"D4RT demo", "language":"python", "name":"d4rt-demo"},
                       language_info={"name":"python", "version":"3.13"})
    for c in cells:
        c.metadata["slideshow"] = {"slide_type":"slide" if c.cell_type == "markdown" and c.source.startswith("## ") else "fragment"}
    return nb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--soccer", action="store_true", help="Create only the separate soccer edition.")
    mode.add_argument("--simple", action="store_true", help="Create only the simple presentation notebook.")
    args = parser.parse_args()
    if args.simple:
        path = ROOT / "D4RT_simple.ipynb"
        nbf.write(make_simple_notebook(), path)
        print(path)
        raise SystemExit(0)
    targets = [("D4RT_soccer.ipynb",False)] if args.soccer else [("D4RT_demo.ipynb",False),("D4RT_rehearsal.ipynb",True)]
    for name, illustration in targets:
        path = ROOT / name
        nbf.write(make_notebook(illustration, soccer=args.soccer), path)
        print(path)
