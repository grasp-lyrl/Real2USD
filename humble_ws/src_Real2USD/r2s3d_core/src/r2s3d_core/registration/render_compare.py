"""Render-and-compare pose+scale refinement (multi-view silhouette + depth).

Motivation (verified on ProcTHOR val, docs/STATUS.md): the 2D masks are high quality
while the fused 3D cloud is detector-mask-contaminated, so registering to the cloud
(ICP) degrades pose and depth-extent scale-fit degrades scale. Fitting pose + per-axis
scale so the mesh's RENDERED silhouette (+depth) matches the observed masks across the
track's kept views instead recovers both far better *when camera poses are clean*
(sim per-object 3D-IoU 0.37->0.62 median; scale err 0.48->0.29). On the real rig its
gain is capped by the ~0.5 m extrinsic miscalibration (a 3D-correct object reprojects
to the wrong pixels), which is a calibration issue, not the method.

Params (7): yaw about world-up + translation + per-axis scale, seeded at the layout
pose. Scale is L2-regularized toward the SAM3D prior (factor 1.0) so unconstrained
(thin/occluded) axes keep the generated aspect instead of exploding, with a loose
safety clamp. Gradient-free (Powell); needs a GPU EGL context for the Open3D renderer.
"""
from __future__ import annotations
from typing import List
import numpy as np

_RENDERERS: dict = {}
_RENDER_WH = (640, 480)
_CALLS = 0
_RECYCLE_EVERY = 1000   # Filament leaks under heavy geometry churn -> segfaults after a few
                        # thousand renders; recreate the offscreen renderer periodically.


def _get_renderer(W: int, H: int):
    """Return a (possibly freshly recycled) offscreen renderer for (W,H)."""
    global _CALLS
    import open3d as o3d
    key = (int(W), int(H))
    if key not in _RENDERERS or _CALLS >= _RECYCLE_EVERY:
        _RENDERERS.pop(key, None)          # drop old ref so Filament resources are freed
        import gc; gc.collect()
        r = o3d.visualization.rendering.OffscreenRenderer(W, H)
        r.scene.set_background([0, 0, 0, 0])
        _RENDERERS[key] = r
        _CALLS = 0
    return _RENDERERS[key]


def _silhouette_depth(verts, faces, K, Twc):
    global _CALLS
    import open3d as o3d
    W, H = _RENDER_WH
    _CALLS += 1
    rend = _get_renderer(W, H)
    rend.scene.clear_geometry()
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                  o3d.utility.Vector3iVector(faces))
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"
    m.paint_uniform_color([0.6, 0.6, 0.6])
    rend.scene.add_geometry("m", m, mat)
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    rend.setup_camera(intr, np.linalg.inv(Twc))
    d = np.asarray(rend.render_to_depth_image(z_in_view_space=True))
    return (np.isfinite(d) & (d > 1e-6)), d


def _decimate(verts, faces, target=3000):
    import open3d as o3d
    if len(faces) <= target:
        return np.asarray(verts, float), np.asarray(faces, np.int32)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(verts, float)),
                                   o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    om = om.simplify_quadric_decimation(target)
    return np.asarray(om.vertices, float), np.asarray(om.triangles, np.int32)


def _param_transform(p, centroid, clamp):
    yaw, tx, ty, tz, sx, sy, sz = p
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    S = np.diag([np.clip(sx, *clamp), np.clip(sy, *clamp), np.clip(sz, *clamp)])
    A = Rz @ S
    T = np.eye(4); T[:3, :3] = A; T[:3, 3] = np.asarray([tx, ty, tz]) + centroid - A @ centroid
    return T


def _mean_iou(views, verts, faces):
    out = []
    for v in views:
        sil, _ = _silhouette_depth(verts, faces, v["K"], v["Twc"])
        u = float((sil | v["mask"]).sum())
        out.append(float((sil & v["mask"]).sum()) / u if u else 0.0)
    return out


def refine_render_compare(verts_world, faces, views, config=None):
    """Return (delta_4x4, info). ``views``: list of {mask(HxW bool), K(3x3), Twc(4x4), depth(HxW)}.

    ``delta`` is the world-frame transform to apply to the posed mesh. Falls back to
    identity if too few views or the renderer is unavailable.
    """
    from scipy.optimize import minimize
    cfg = config or {}
    lam_d = float(cfg.get("rc_lambda_depth", 0.0))  # depth enters via the ICP pose (fixed
    #   distance), not the objective; the silhouette + SAM3D-aspect prior carry the scale.
    lam_s = float(cfg.get("rc_lambda_scale", 0.30))
    clamp = (float(cfg.get("rc_scale_lo", 0.33)), float(cfg.get("rc_scale_hi", 3.0)))
    maxit = int(cfg.get("rc_maxiter", 40))
    max_views = int(cfg.get("rc_max_views", 4))
    if len(views) < 1:
        return np.eye(4), {"fallback": "no_views", "n_views": 0}
    # cap views (largest mask first): multi-view in sim (clean poses jointly constrain
    # all axes); single-view on real (max_views=1) is drift-robust since the mesh is
    # rendered back into the same frame it was placed from, so odometry drift cancels.
    views = sorted(views, key=lambda v: -int(v["mask"].sum()))[:max_views]

    global _RENDER_WH
    H, W = views[0]["mask"].shape
    _RENDER_WH = (W, H)
    verts0, faces = _decimate(verts_world, faces)
    centroid = verts0.mean(0)
    try:
        iou0 = _mean_iou(views, verts0, faces)
    except Exception as e:                       # no GPU/EGL etc.
        return np.eye(4), {"fallback": f"renderer_unavailable:{e}"}

    # scale_only: freeze pose to the (good) layout and optimize only per-axis scale, so the
    # silhouette can fix SIZE without the pose search overfitting orientation on
    # viewpoint-clustered tracks (which degrades rotation at scene scale).
    scale_only = bool(cfg.get("rc_scale_only"))

    def _full(q):
        return np.concatenate([[0.0, 0.0, 0.0, 0.0], q]) if scale_only else q

    def loss(q):
        p = _full(q)
        T = _param_transform(p, centroid, clamp)
        vh = (T[:3, :3] @ verts0.T + T[:3, 3:4]).T
        tot = 0.0
        for v in views:
            sil, dr = _silhouette_depth(vh, faces, v["K"], v["Twc"])
            u = float((sil | v["mask"]).sum())
            term = 1.0 - (float((sil & v["mask"]).sum()) / u if u else 0.0)
            ov = sil & v["mask"] & (v["depth"] > 0)
            if ov.sum() > 20:
                term += lam_d * float(np.mean(np.minimum(np.abs(dr[ov] - v["depth"][ov]), 0.3)))
            tot += term
        return tot / len(views) + lam_s * float(np.mean((np.asarray(p[4:7]) - 1.0) ** 2))

    q0 = np.array([1, 1, 1], float) if scale_only else np.array([0, 0, 0, 0, 1, 1, 1], float)
    res = minimize(loss, q0, method="Powell",
                   options={"maxiter": maxit, "xtol": 2e-3, "ftol": 2e-3})
    pf = _full(res.x)
    delta = _param_transform(pf, centroid, clamp)
    vh = (delta[:3, :3] @ verts0.T + delta[:3, 3:4]).T
    iou1 = _mean_iou(views, vh, faces)
    return delta, {"fallback": None, "n_views": len(views),
                   "silhouette_iou_before": float(np.mean(iou0)),
                   "silhouette_iou_after": float(np.mean(iou1)),
                   "scale_only": scale_only,
                   "yaw_deg": float(np.degrees(pf[0])),
                   "scale": np.round(pf[4:7], 3).tolist()}
