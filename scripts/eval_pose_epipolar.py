#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pose accuracy evaluation by epipolar error of independent correspondences.

Compares camera pose sets (ARKit raw vs. BA-refined) written by
`train-capture` with `POSE_DUMP=<json>` against SIFT correspondences that are
NOT used by CaptureBA (which uses ORB + ARKit feature tracks).

Outliers are rejected with a fundamental-matrix RANSAC fitted to the matches
alone (no pose involved), so the inlier set is identical for every pose set
and does not favour either one. For each pose set the fundamental matrix is
built from the relative pose and intrinsics, and the symmetric point-to-line
distance (px) is measured. The epipolar error is invariant to a global
similarity transform, so it measures relative-pose (+ intrinsics) error only.

The error of the data-fitted F is reported as the measurement noise floor.

Requires: numpy, opencv-python-headless (SIFT).

Usage:
  python scripts/eval_pose_epipolar.py <dataset.valset> <pose_dump.json> [--gaps 1,2,3,5,8]
         [--perturb-deg 0.1,0.3] [--max-pairs-per-gap 200]
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np


def load_frames(valset, pose_path):
    manifest = json.load(open(os.path.join(valset, "manifest.json")))
    by_ts = {round(f["timestamp"], 6): f for f in manifest["frames"]}
    dump = json.load(open(pose_path))["frames"]
    frames = []
    for d in sorted(dump, key=lambda f: f["timestamp"]):
        f = by_ts.get(round(d["timestamp"], 6))
        if f is None:
            continue
        frames.append({
            "ts": d["timestamp"],
            "image": os.path.join(valset, f["imageFile"]),
            "w": d["width"], "h": d["height"],
            "K": d["intrinsics"],
            "V": {"ARKit": np.array(d["viewmatPreBA"], float).reshape(4, 4),
                  "BA": np.array(d["viewmat"], float).reshape(4, 4)},
        })
    return frames


def gray(fr):
    a = np.fromfile(fr["image"], np.uint8).reshape(fr["h"], fr["w"], 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)


def kmat(k):
    fx, fy, cx, cy = k
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def fundamental(V1, V2, K1, K2):
    """F with x2^T F x1 = 0 for pixels in the +0.5-center convention."""
    rel = V2 @ np.linalg.inv(V1)
    R, t = rel[:3, :3], rel[:3, 3]
    E = skew(t) @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def sym_epi(F, x1, x2):
    """Symmetric point-to-epipolar-line distance (px), per match."""
    h1 = np.c_[x1, np.ones(len(x1))]
    h2 = np.c_[x2, np.ones(len(x2))]
    l2 = h1 @ F.T          # lines in image 2
    l1 = h2 @ F            # lines in image 1
    d2 = np.abs(np.sum(l2 * h2, 1)) / np.hypot(l2[:, 0], l2[:, 1])
    d1 = np.abs(np.sum(l1 * h1, 1)) / np.hypot(l1[:, 0], l1[:, 1])
    return 0.5 * (d1 + d2)


def rot_perturb(V, deg, rng):
    """Rotate the camera by `deg` degrees about a random axis (camera frame)."""
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R, _ = cv2.Rodrigues(ax * math.radians(deg))
    P = np.eye(4)
    P[:3, :3] = R
    return P @ V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("valset")
    ap.add_argument("pose_json")
    ap.add_argument("--gaps", default="1,2,3,5,8")
    ap.add_argument("--perturb-deg", default="0.1,0.3")
    ap.add_argument("--max-pairs-per-gap", type=int, default=200)
    ap.add_argument("--min-inliers", type=int, default=40)
    ap.add_argument("--revisit-pairs", type=int, default=150,
                    help="number of camera pairs that are far apart in time (>= --revisit-min-gap) but look at the same place (counted as gap=0)")
    ap.add_argument("--revisit-min-gap", type=int, default=20)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    frames = load_frames(a.valset, a.pose_json)
    n = len(frames)
    if n < 2:
        sys.exit("no frames matched between valset and pose dump")
    print(f"frames: {n} (BA output, matched to valset by timestamp)")

    rng = np.random.default_rng(0)
    perturb = [float(x) for x in a.perturb_deg.split(",") if x]
    pose_sets = ["ARKit", "BA"]
    for d in perturb:
        name = f"ARKit+{d}°"
        for fr in frames:
            fr["V"][name] = rot_perturb(fr["V"]["ARKit"], d, rng)
        pose_sets.append(name)

    sift = cv2.SIFT_create(nfeatures=4000)
    feats = []
    for fr in frames:
        kp, des = sift.detectAndCompute(gray(fr), None)
        # OpenCV puts pixel centers on integers; ours are at +0.5.
        pts = np.array([k.pt for k in kp], float) + 0.5
        feats.append((pts, des))
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Revisit pairs: frames far apart in time whose ARKit poses have nearby camera centers and similar view directions.
    # Measures consistency between segments where capture returned to the same place (global distortion invisible to local pairs).
    C = np.array([-(f["V"]["ARKit"][:3, :3].T @ f["V"]["ARKit"][:3, 3]) for f in frames])
    Z = np.array([f["V"]["ARKit"][2, :3] for f in frames])     # optical axis (world)
    ext = np.linalg.norm(C.max(0) - C.min(0))
    cand = []
    for i in range(n):
        for j in range(i + a.revisit_min_gap, n):
            d = np.linalg.norm(C[i] - C[j])
            ang = math.degrees(math.acos(max(-1, min(1, float(Z[i] @ Z[j])))))
            if d < 0.25 * ext and ang < 25:
                cand.append((d / ext + ang / 25, i, j))
    cand.sort()
    revisit = [(i, j) for _, i, j in cand[:a.revisit_pairs]]
    print(f"revisit pairs: {len(revisit)} (gap >= {a.revisit_min_gap}, close & co-oriented by ARKit)")

    rows = []  # per match: gap, pair, set errors..., radius, row, speed
    pair_stats = []
    plan = []
    for g in [int(x) for x in a.gaps.split(",")]:
        idx = list(range(0, n - g))
        if len(idx) > a.max_pairs_per_gap:
            idx = list(np.linspace(0, n - g - 1, a.max_pairs_per_gap).astype(int))
        plan += [(g, i, i + g) for i in idx]
    plan += [(0, i, j) for i, j in revisit]
    for g, i, j in plan:
        if True:
            (p1, d1), (p2, d2) = feats[i], feats[j]
            if d1 is None or d2 is None or len(p1) < 8 or len(p2) < 8:
                continue
            m12 = bf.knnMatch(d1, d2, k=2)
            m21 = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(d2, d1, k=2) if len(m) == 2}
            good = [m[0] for m in m12 if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
                    and m21.get(m[0].trainIdx) == m[0].queryIdx]
            if len(good) < a.min_inliers:
                continue
            x1 = np.array([p1[m.queryIdx] for m in good])
            x2 = np.array([p2[m.trainIdx] for m in good])
            Fd, mask = cv2.findFundamentalMat(x1, x2, cv2.USAC_MAGSAC, 1.0, 0.9999, 20000)
            if Fd is None or mask is None:
                continue
            inl = mask.ravel().astype(bool)
            if inl.sum() < a.min_inliers:
                continue
            x1, x2 = x1[inl], x2[inl]
            fi, fj = frames[i], frames[j]
            K1, K2 = kmat(fi["K"]), kmat(fj["K"])
            errs = {"data": sym_epi(Fd[:3], x1, x2)}
            for s in pose_sets:
                errs[s] = sym_epi(fundamental(fi["V"][s], fj["V"][s], K1, K2), x1, x2)
            # diagnostics: radius from principal point (normalized by half-diagonal), row, speed
            c = np.array([fi["K"][2], fi["K"][3]])
            rad = np.linalg.norm(x1 - c, axis=1) / (0.5 * math.hypot(fi["w"], fi["h"]))
            row = x1[:, 1] / fi["h"]
            Ci = -fi["V"]["ARKit"][:3, :3].T @ fi["V"]["ARKit"][:3, 3]
            Cj = -fj["V"]["ARKit"][:3, :3].T @ fj["V"]["ARKit"][:3, 3]
            speed = np.linalg.norm(Ci - Cj) / max(fj["ts"] - fi["ts"], 1e-3)
            pair_stats.append({"gap": g, "i": i, "j": j, "n": int(inl.sum()), "speed": speed,
                               **{s: float(np.median(errs[s])) for s in errs}})
            for k in range(len(x1)):
                rows.append((g, len(pair_stats) - 1, rad[k], row[k], speed,
                             *[errs[s][k] for s in ["data"] + pose_sets]))

    if not rows:
        sys.exit("no valid pairs")
    R = np.array(rows)
    cols = ["data"] + pose_sets
    E = {s: R[:, 5 + k] for k, s in enumerate(cols)}

    def med(v):
        return f"{np.median(v):6.3f}" if len(v) else "     -"

    print(f"\npairs: {len(pair_stats)}, inlier matches: {len(R)}")
    print("\n== median symmetric epipolar error [px] by frame gap ==")
    print(f"{'gap':>4} {'pairs':>6} " + " ".join(f"{s:>12}" for s in cols))
    for g in sorted(set(R[:, 0].astype(int))):
        m = R[:, 0] == g
        npair = len({int(p) for p in R[m, 1]})
        print(f"{g:>4} {npair:>6} " + " ".join(f"{np.median(E[s][m]):12.3f}" for s in cols))
    print(f"{'all':>4} {len(pair_stats):>6} " + " ".join(f"{np.median(E[s]):12.3f}" for s in cols))
    print("\n== mean / p90 [px] (all matches) ==")
    for s in cols:
        print(f"  {s:>12}: mean {np.mean(E[s]):.3f}  p90 {np.percentile(E[s], 90):.3f}")

    ps = pair_stats
    better = sum(1 for p in ps if p["BA"] < p["ARKit"])
    print(f"\npairs where BA < ARKit (per-pair median): {better}/{len(ps)} ({100.0 * better / len(ps):.1f}%)")
    rv = [p for p in ps if p["gap"] == 0]
    if rv:
        rb = sum(1 for p in rv if p["BA"] < p["ARKit"])
        print(f"revisit pairs where BA < ARKit: {rb}/{len(rv)} ({100.0 * rb / len(rv):.1f}%), "
              f"median ARKit {np.median([p['ARKit'] for p in rv]):.3f} / BA {np.median([p['BA'] for p in rv]):.3f} px")

    print("\n== by radius from principal point (normalized; 1 = image corner) ==")
    for lo, hi in [(0, .3), (.3, .6), (.6, .8), (.8, 1.1)]:
        m = (R[:, 2] >= lo) & (R[:, 2] < hi)
        print(f"  r {lo:.1f}-{hi:.1f} n={m.sum():6d} " + " ".join(f"{s}={med(E[s][m])}" for s in ["data", "ARKit", "BA"]))
    print("\n== by image row band x camera speed tertile (ARKit / BA) ==")
    sp = R[:, 4]
    t1, t2 = np.percentile(sp, [33.3, 66.7])
    for name, ms in [("slow", sp < t1), ("mid", (sp >= t1) & (sp < t2)), ("fast", sp >= t2)]:
        cells = []
        for lo, hi, rn in [(0, 1 / 3, "top"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "bottom")]:
            m = ms & (R[:, 3] >= lo) & (R[:, 3] < hi)
            cells.append(f"{rn}: {med(E['ARKit'][m])}/{med(E['BA'][m])}")
        print(f"  {name:>4} (speed ~{np.median(sp[ms]):.2f} m/s): " + "  ".join(cells))

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("gap,i,j,n,speed," + ",".join(cols) + "\n")
            for p in ps:
                f.write(f"{p['gap']},{p['i']},{p['j']},{p['n']},{p['speed']:.4f}," +
                        ",".join(f"{p[s]:.4f}" for s in cols) + "\n")


if __name__ == "__main__":
    main()
