"""Presentation visuals. Plotly exports embed their own JS for offline use."""

from pathlib import Path
import base64
import html
import io

import matplotlib.pyplot as plt
import numpy as np

TEAL, ORANGE, BLUE = "#24d6be", "#ffac60", "#7eacff"


def style():
    plt.rcParams.update({"figure.facecolor": "#0c1424", "axes.facecolor": "#111d32",
                        "axes.edgecolor": "#61728c", "text.color": "#e8efff",
                        "axes.labelcolor": "#d4e0f6", "xtick.color": "#a8b8d0",
                        "ytick.color": "#a8b8d0", "grid.color": "#33445f",
                        "font.size": 12, "axes.titlesize": 15, "figure.dpi": 115,
                        "savefig.facecolor": "#0c1424"})


def banner(scene):
    from IPython.display import HTML, display
    real = scene.backend == "opend4rt"
    label = "OpenD4RT · learned predictions" if real else "ILLUSTRATION · analytic geometry · no learned inference"
    color = TEAL if real else ORANGE
    display(HTML(f'<div style="padding:14px 20px;border:1px solid {color};border-radius:10px;'
                 f'color:{color};background:#111d32;font:600 15px system-ui">{label}</div>'))


def _finish(fig, label):
    fig.suptitle(label, color="#a8b8d0", fontsize=10, y=1.01)
    fig.tight_layout()
    plt.show()


def filmstrip(video, uv=None, source=0, *, marker_size=900):
    ids = np.linspace(0, len(video)-1, 4, dtype=int)
    # Include the selected source even when it is not one of the regular samples.
    if uv is not None and source not in ids:
        ids[np.argmin(np.abs(ids - source))] = source
        ids.sort()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    h, w = video.frames.shape[1:3]
    for ax, t in zip(axes, ids):
        ax.imshow(video.frames[t]); ax.set_title(f"frame {t}"); ax.axis("off")
        if uv is not None and t == source:
            x, y = uv[0] * (w-1), uv[1] * (h-1)
            ax.scatter(x, y, s=marker_size, facecolors="none", edgecolors="black", linewidths=7, zorder=5)
            ax.scatter(x, y, s=marker_size, facecolors="none", edgecolors="#ffe600", linewidths=3, zorder=6)
            ax.scatter(x, y, s=40, facecolors="white", edgecolors="black", linewidths=1, zorder=7)
            ax.annotate("SOURCE POINT", xy=(x, y), xytext=(.05, .08),
                        textcoords="axes fraction", color="#ffe600", fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=.35", facecolor="black", edgecolor="#ffe600", alpha=.9),
                        arrowprops=dict(arrowstyle="->", color="#ffe600", lw=2), zorder=8)
    _finish(fig, video.name)


def depth_view(video, result, t=0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    axes[0].imshow(video.frames[t]); axes[0].set_title("RGB observation"); axes[0].axis("off")
    depth = np.where(np.isfinite(result.depth) & (result.depth > 0), result.depth, np.nan)
    if not np.isfinite(depth).any():
        axes[1].text(.5, .5, "No finite positive depth predictions", ha="center")
    else:
        lo, hi = np.nanpercentile(depth, [2, 98])
        im = axes[1].imshow(depth, cmap="magma", vmin=lo, vmax=max(hi, lo+1e-6))
        fig.colorbar(im, ax=axes[1], label="Camera Z · model units")
    axes[1].set_title("Same query, sampled across pixels → depth"); axes[1].axis("off")
    _finish(fig, result.backend)


def track_view(video, uv0, tracks, local, times):
    fig = plt.figure(figsize=(14, 4.5))
    ax = fig.add_subplot(121)
    t = int(times[-1]); ax.imshow(video.frames[t]); ax.set_title(f"2D correspondence · frame {t}")
    h, w = video.frames.shape[1:3]
    for k in range(len(uv0)):
        color = plt.get_cmap("Set2")(k % 8)
        uv = local.uv[k] * [w-1, h-1]
        visible = local.visibility[k] > .5
        ax.plot(uv[:, 0], uv[:, 1], color=color, alpha=.45)
        ax.scatter(uv[visible, 0], uv[visible, 1], color=[color], s=12)
        if visible[-1]:
            ax.scatter(*uv[-1], color=[color], s=65, edgecolors="white")
    ax.set_xlim(0, w-1); ax.set_ylim(h-1, 0); ax.axis("off")
    ax = fig.add_subplot(122, projection="3d")
    for k, xyz in enumerate(tracks.xyz):
        ax.plot(*xyz.T, color=plt.get_cmap("Set2")(k % 8), linewidth=2)
        ax.scatter(*xyz[0], s=25)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="3D trajectories · camera 0 reference")
    ax.view_init(elev=18, azim=-65)
    _finish(fig, tracks.backend + " · dots: visibility > 0.5; lines connect queried times")


def camera_view(fixed, moving, times, point=0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), sharey=True)
    for ax, result, title in zip(axes, (fixed, moving),
                                 ("t_cam = 0 · a fixed reference", "t_cam = t_tgt · each frame's camera")):
        for j, (name, color) in enumerate(zip("XYZ", (TEAL, ORANGE, BLUE))):
            ax.plot(times, result.xyz[point, :, j], label=name, color=color, linewidth=2)
        ax.set(title=title, xlabel="Target frame", ylabel="Coordinate · model units"); ax.legend(); ax.grid(alpha=.3)
    _finish(fig, fixed.backend + " · same source point, same target times, different reference cameras")


def sample_colors(video, u, v, t_src=0):
    h, w = video.frames.shape[1:3]
    return video.frames[t_src, np.rint(v*(h-1)).astype(int), np.rint(u*(w-1)).astype(int)]


def scene_player(result, colors, times, path="artifacts/4d_scene.html"):
    """Precomputed 3D time slider; rotating it requires neither Python nor GPU."""
    import plotly.graph_objects as go
    from IPython.display import HTML, display

    xyz = result.xyz  # [points,times,3]
    colors = np.asarray(colors).reshape(-1, 3)
    css_colors = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors]

    def traces(i):
        p = xyz[:, i]
        finite = np.isfinite(p).all(-1) & (p[:, 2] > 0)
        visible = (result.visibility[:, i] > .5) & finite
        hidden = (~(result.visibility[:, i] > .5)) & finite
        out = []
        for mask, opacity, name in ((visible, .95, "Visible > 0.5"), (hidden, .15, "Occluded / low visibility")):
            out.append(go.Scatter3d(x=p[mask, 0], y=p[mask, 1], z=p[mask, 2], mode="markers", name=name,
                                   marker=dict(size=3, color=np.asarray(css_colors)[mask].tolist(), opacity=opacity),
                                   hovertemplate="X %{x:.3f}<br>Y %{y:.3f}<br>Z %{z:.3f}<extra></extra>"))
        return out

    fig = go.Figure(data=traces(0), frames=[go.Frame(name=str(int(t)), data=traces(i)) for i, t in enumerate(times)])
    finite = xyz[np.isfinite(xyz).all(-1)]
    ranges = {}
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99], axis=0)
        span = max(float((hi-lo).max()), 0.1)
        for j, axis in enumerate("xyz"):
            center = float((lo[j]+hi[j])/2)
            ranges[axis+"axis"] = dict(title=axis.upper(), range=[center-span*.55, center+span*.55],
                                       gridcolor="#30415a", backgroundcolor="#111d32")
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0c1424", plot_bgcolor="#0c1424",
                      title=f"{result.backend.upper()} · a sampled surface through time · fixed camera 0",
                      height=620, margin=dict(l=0, r=0, t=60, b=90),
                      scene=dict(aspectmode="cube", uirevision="keep-view",
                                 camera=dict(up=dict(x=0,y=-1,z=0), eye=dict(x=1.5,y=-.5,z=-1.8)), **ranges),
                      updatemenus=[dict(type="buttons", direction="left", showactive=False, x=.02, y=.02,
                                        bgcolor="#24334c", bordercolor="#60748f", font=dict(color="#e8efff"), buttons=[
                          dict(label="▶ Play", method="animate", args=[None, {"frame":{"duration":160,"redraw":True}, "transition":{"duration":0}, "fromcurrent":True}]),
                          dict(label="Pause", method="animate", args=[[None], {"mode":"immediate", "frame":{"duration":0,"redraw":False}}])])],
                      sliders=[dict(x=.25, len=.72, y=.02, currentvalue=dict(prefix="Target frame "),
                                    steps=[dict(label=str(int(t)), method="animate", args=[[str(int(t))],
                                         {"mode":"immediate", "frame":{"duration":0,"redraw":True}, "transition":{"duration":0}}]) for t in times])])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = fig.to_html(include_plotlyjs=True, full_html=True, auto_play=False,
                          config={"displaylogo":False, "responsive":True})
    path.write_text(content)
    display({"text/html": '<iframe title="Interactive 4D scene" style="width:100%;height:660px;border:0;border-radius:12px" '
             'srcdoc="' + html.escape(content, quote=True) + '"></iframe>'}, raw=True)
    return path


def video_player(video):
    """Embed a compact GIF, requiring no codec or server in exported HTML."""
    from PIL import Image
    from IPython.display import HTML, display

    h, w = video.frames.shape[1:3]
    frames = [Image.fromarray(f).resize((480, round(480*h/w))) for f in video.frames]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=round(1000/video.fps), loop=0)
    uri = base64.b64encode(buf.getvalue()).decode()
    display(HTML(f'<img alt="{html.escape(video.name)}" src="data:image/gif;base64,{uri}" '
                 'style="width:640px;max-width:100%;border-radius:12px"/>'))


def query_explorer(scene, u=.35, v=.5):
    """Query on button press; sliders never trigger GPU work automatically."""
    import ipywidgets as widgets
    from IPython.display import display

    n = len(scene.video)
    us = widgets.FloatSlider(value=u, min=0, max=1, step=.01, description="u", continuous_update=False)
    vs = widgets.FloatSlider(value=v, min=0, max=1, step=.01, description="v", continuous_update=False)
    ts = [widgets.IntSlider(value=0, min=0, max=n-1, description=label, continuous_update=False)
          for label in ("t_src", "t_tgt", "t_cam")]
    ts[1].value = n-1
    button = widgets.Button(description="Query this point", button_style="info")
    output = widgets.Output()

    def update(_):
        with output:
            output.clear_output(wait=True)
            r = scene.query(us.value, vs.value, *(s.value for s in ts))
            print(f"{scene.backend} | xyz = {np.round(r.xyz, 4)} | visibility = {float(r.visibility):.3f}")
            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            h, w = scene.video.frames.shape[1:3]
            axes[0].imshow(scene.video.frames[ts[0].value])
            axes[0].scatter(us.value*(w-1), vs.value*(h-1), s=90, facecolors="none", edgecolors=ORANGE)
            axes[0].set_title(f"Source {ts[0].value}"); axes[0].axis("off")
            axes[1].imshow(scene.video.frames[ts[1].value]); axes[1].set_title(f"Target {ts[1].value}")
            # Query in target camera to obtain the image overlay, as upstream does.
            local = scene.query(us.value, vs.value, ts[0].value, ts[1].value, ts[1].value)
            if np.isfinite(local.uv).all():
                axes[1].scatter(*(local.uv*[w-1, h-1]), s=90, facecolors="none", edgecolors=TEAL)
            axes[1].set_xlim(0,w-1); axes[1].set_ylim(h-1,0); axes[1].axis("off")
            fig.tight_layout(); plt.show()

    button.on_click(update)
    display(widgets.VBox([widgets.HBox([us, vs]), widgets.HBox(ts), button, output]))
