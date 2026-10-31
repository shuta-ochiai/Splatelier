#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Splatelier capture-guide loops. The English title and captions are drawn into the frames.

  slow   : move slowly  (fast pass = sparse, blurred captures / slow = dense)
  turn   : avoid sudden turns (jerk = tracking lost / smooth orbit = locked)
  angles : capture from many heights (low -> mid -> high orbit)
  cool   : keep the phone cool for training (close other apps / unplug / take the case off)

usage: python3 render_guides.py OUT_DIR {slow,turn,angles,pivot,loop,light,subjects,cool} [--only N]
"""
import argparse, math, os, random
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("scene")
ap.add_argument("--fps", type=int, default=60)
ap.add_argument("--only", type=float, default=-1, help="render a single time (s)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

W = H = 1080; SS = 2
BG = (22, 27, 70); IVORY = (245, 236, 221); PURPLE = (124, 92, 230); LAV = (176, 156, 255)
CORAL = (232, 89, 74); ORANGE = (255, 164, 31); DARK = (15, 18, 48)
RED = (205, 62, 58); RED_HI = (232, 96, 84); GOLD = (232, 184, 74); WHITE = (255, 255, 255)
CX, CY = 540, 650
TARGET = (540, 455)
RINGS = {"low": (380, 128, 0), "mid": (305, 96, 178), "high": (185, 56, 340)}


def clamp(x): return max(0.0, min(1.0, x))
def smooth(x): x = clamp(x); return x * x * (3 - 2 * x)
def eob(x, s=1.6): x = clamp(x) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2
def seg(t, a, b): return clamp((t - a) / (b - a))
def lerp(a, b, u): return a + (b - a) * u
def fade(t, a, b, c, d):
    """0 before a, ramps to 1 by b, holds, ramps to 0 from c to d"""
    return smooth(seg(t, a, b)) * (1 - smooth(seg(t, c, d)))

def ring_xy(name, phi):
    rx, ry, h = RINGS[name]
    r = math.radians(phi)
    return CX + rx * math.cos(r), CY - h + ry * math.sin(r)

OY = 62  # global vertical offset to centre the composition
def S(p): return (p[0] * SS, (p[1] + OY) * SS)

from PIL import ImageFont
# Fonts live next to this script (fonts/, fetched by fetch_fonts.sh, SIL OFL 1.1): Lora for titles, Noto Sans JP for
# captions and labels.
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
def _vf(name, size, weight):
    f = ImageFont.truetype(os.path.join(FONT_DIR, name), size * SS)
    f.set_variation_by_axes([weight]); return f
F_TITLE = _vf("Lora-VF.ttf", 64, 600)
F_CAP = _vf("NotoSansJP-VF.ttf", 37, 500)
F_SUB = _vf("NotoSansJP-VF.ttf", 30, 500)
F_LAB = _vf("NotoSansJP-VF.ttf", 25, 500)

def text_abs(d, x, y, txt, font, c, a):
    """y in final-frame pixels (no OY offset)"""
    if a <= 0: return
    d.text((x * SS, y * SS), txt, font=font, fill=col(c, a), anchor="mm")
def col(c, a): return tuple(c) + (int(255 * clamp(a)),)


# ---------------------------------------------------------------- drawing
def dashed_arc(d, name, a0, a1, alpha, width=3, dash=7, gap=6):
    n = int(abs(a1 - a0) / 1.0) + 2
    pts = [ring_xy(name, lerp(a0, a1, i / (n - 1))) for i in range(n)]
    on = True; acc = 0; cur = [pts[0]]
    for p, q in zip(pts, pts[1:]):
        acc += 1
        cur.append(q)
        if acc >= (dash if on else gap):
            if on and len(cur) > 1:
                d.line([S(x) for x in cur], fill=col(IVORY, alpha), width=width * SS)
            on = not on; acc = 0; cur = [q]
    if on and len(cur) > 1:
        d.line([S(x) for x in cur], fill=col(IVORY, alpha), width=width * SS)

def ring(d, name, alpha, front):
    if front: dashed_arc(d, name, 0, 180, alpha)
    else:     dashed_arc(d, name, 180, 360, alpha)

def ground(d):
    rx, ry, _ = RINGS["low"]
    d.ellipse([S((CX - rx - 40, CY - ry - 22)), S((CX + rx + 40, CY + ry + 22))], fill=(255, 255, 255, 9))

# ---- subject: potted kikyo painted with real gaussians (same as the icon)
FC = (540, 418)   # flower centre in scene space
FS = 0.52         # icon -> scene scale for the flower
def _hex(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
P_PETAL = [(0, 1), (.45, .8), (.8, .22), (1, 0)]
P_TIP = [(0, 1), (.45, .75), (.8, .2), (1, 0)]
P_VEIN = [(0, .85), (.6, .35), (1, 0)]
P_WHITE = [(0, .95), (.5, .55), (1, 0)]
P_LEAF = [(0, 1), (.45, .7), (.8, .18), (1, 0)]

def _flower_splats():
    out = []
    def ring_(r, rx, ry, c, prof):
        for a in (-90, -18, 54, 126, 198):
            ra = math.radians(a)
            out.append((FC[0] + r * FS * math.cos(ra), FC[1] + r * FS * math.sin(ra), rx * FS, ry * FS, a, c, prof))
    ring_(92, 100, 62, _hex("#7C5CE6"), P_PETAL)
    ring_(158, 48, 22, _hex("#5B3FD0"), P_TIP)
    ring_(100, 88, 8, _hex("#3A2290"), P_VEIN)
    ring_(38, 36, 18, _hex("#FFFFFF"), P_WHITE)
    ring_(14, 18, 6, _hex("#F4E7A8"), P_WHITE)
    return out

def build_object():
    X0, Y0, X1, Y1 = 380, 300, 700, 640
    w, h = (X1 - X0) * SS, (Y1 - Y0) * SS
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(base, "RGBA")
    L = lambda p: ((p[0] - X0) * SS, (p[1] - Y0) * SS)
    d.ellipse([L((452, 596)), L((628, 620))], fill=(0, 0, 0, 80))                       # contact shadow
    d.line([L((540, 552)), L((538, 500)), L((540, 430))], fill=(62, 158, 104, 255), width=7 * SS, joint="curve")
    d.polygon([L((480, 552)), L((600, 552)), L((588, 606)), L((492, 606))], fill=IVORY + (255,))   # pot
    d.rounded_rectangle([L((470, 540)), L((610, 558))], radius=6 * SS, fill=(230, 218, 196, 255))
    d.line([L((485, 578)), L((595, 578))], fill=PURPLE + (170,), width=4 * SS)
    arr = np.asarray(base, np.float32) / 255
    a = arr[..., 3:4]; c = arr[..., :3] * a                                              # premultiplied
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    leaves = [(506, 526, 42, 14, -28, _hex("#3E9E68"), P_LEAF), (576, 520, 40, 13, 24, _hex("#3E9E68"), P_LEAF),
              (522, 500, 26, 9, -50, _hex("#56B87E"), P_LEAF)]
    for cx, cy, rx, ry, rot, colr, prof in leaves + _flower_splats():
        px, py = (cx - X0) * SS, (cy - Y0) * SS
        rx, ry = rx * SS, ry * SS
        R = max(rx, ry) + 2
        x0, x1 = int(max(0, px - R)), int(min(w, px + R + 1))
        y0, y1 = int(max(0, py - R)), int(min(h, py + R + 1))
        X = xx[y0:y1, x0:x1] - px; Y = yy[y0:y1, x0:x1] - py
        cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        u = (X * cr + Y * sr) / rx; v = (-X * sr + Y * cr) / ry
        r = np.sqrt(u * u + v * v)
        o, op = zip(*prof)
        al = np.interp(r, o, op).astype(np.float32); al[r > 1] = 0; al = al[..., None]
        c[y0:y1, x0:x1] = colr * al + c[y0:y1, x0:x1] * (1 - al)
        a[y0:y1, x0:x1] = al + a[y0:y1, x0:x1] * (1 - al)
    rgb = np.where(a > 1e-4, c / np.maximum(a, 1e-4), 0)
    out = np.concatenate([rgb, a], -1)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA"), (X0 * SS, (Y0 + OY) * SS)

OBJ, OBJ_AT = build_object()

def draw_object(im):
    im.paste(OBJ, OBJ_AT, OBJ)

def phone(d, x, y, s, a, aim_off=0.0, frustum=True):
    if a <= 0: return
    if frustum:
        tx, ty = TARGET
        ang = math.atan2(ty - y, tx - x) + math.radians(aim_off)
        L = math.hypot(tx - x, ty - y) * 1.05
        ex, ey = x + math.cos(ang) * L, y + math.sin(ang) * L
        nx, ny = -math.sin(ang), math.cos(ang)
        hw = 92
        tri = [(x, y), (ex + nx * hw, ey + ny * hw), (ex - nx * hw, ey - ny * hw)]
        d.polygon([S(p) for p in tri], fill=col(PURPLE, 0.20 * a))
        d.line([S(tri[0]), S(tri[1])], fill=col(LAV, 0.55 * a), width=2 * SS)
        d.line([S(tri[0]), S(tri[2])], fill=col(LAV, 0.55 * a), width=2 * SS)
    w, h = 56 * s, 104 * s
    d.rounded_rectangle([S((x - w / 2, y - h / 2)), S((x + w / 2, y + h / 2))], radius=int(12 * s * SS),
                        fill=col(IVORY, a), outline=col(DARK, a), width=int(3 * SS))
    d.ellipse([S((x - w / 2 + 9 * s, y - h / 2 + 9 * s)), S((x - w / 2 + 21 * s, y - h / 2 + 21 * s))], fill=col(DARK, a))

def capture(d, x, y, a, good=True, blur_dir=None):
    c = LAV if good else ORANGE
    r = 9
    if blur_dir is None:
        d.rounded_rectangle([S((x - r, y - r)), S((x + r, y + r))], radius=4 * SS, fill=col(c, a))
    else:
        bx, by = blur_dir
        for k in range(-3, 4):
            ox, oy = bx * k * 6, by * k * 6
            d.rounded_rectangle([S((x + ox - r, y + oy - r)), S((x + ox + r, y + oy + r))], radius=4 * SS,
                                fill=col(c, a * 0.28))

def badge(d, kind, t, t_in, t_out, cx=540, cy=150):
    if t < t_in or t > t_out + 0.25: return
    sc = eob(seg(t, t_in, t_in + 0.35))
    a = 1 - smooth(seg(t, t_out, t_out + 0.25))
    if sc <= 0 or a <= 0: return
    r = 46 * sc
    c = {"ok": PURPLE, "ng": CORAL, "warn": ORANGE}[kind]
    d.ellipse([S((cx - r, cy - r)), S((cx + r, cy + r))], fill=col(c, a))
    k = sc
    if kind == "ok":
        pts = [(cx - 19 * k, cy + 1 * k), (cx - 5 * k, cy + 15 * k), (cx + 21 * k, cy - 14 * k)]
        d.line([S(p) for p in pts], fill=col(WHITE, a), width=int(9 * SS * k), joint="curve")
    elif kind == "ng":
        for sx in (1, -1):
            d.line([S((cx - 15 * k, cy - 15 * k * sx)), S((cx + 15 * k, cy + 15 * k * sx))], fill=col(WHITE, a),
                   width=int(9 * SS * k))
    else:
        d.line([S((cx, cy - 20 * k)), S((cx, cy + 5 * k))], fill=col(DARK, a), width=int(9 * SS * k))
        rr = 5.5 * k
        d.ellipse([S((cx - rr, cy + 16 * k - rr)), S((cx + rr, cy + 16 * k + rr))], fill=col(DARK, a))

def depth_scale(phi, name="low"):
    return (0.84 + 0.16 * math.sin(math.radians(phi))) * (1.0 if name == "low" else 0.94 if name == "mid" else 0.88)


# ---------------------------------------------------------------- scenes
class Scene:
    """Collects back/front draw calls around the daruma."""
    def __init__(self): self.back = []; self.front = []
    def add(self, front, fn): (self.front if front else self.back).append(fn)


def scene_slow(t, sc):
    D = 6.0
    ring_a = 0.35
    sc.add(False, lambda d: ring(d, "low", ring_a, False)); sc.add(True, lambda d: ring(d, "low", ring_a, True))
    # phone path
    def phi_at(tt):
        if tt < 0.3: return 160
        if tt < 1.0: return lerp(160, 20, smooth(seg(tt, 0.3, 1.0)))
        if tt < 2.8: return 20
        if tt < 3.1: return 160
        if tt < 5.3: return lerp(160, 20, smooth(seg(tt, 3.1, 5.3)))
        return 20
    alpha = fade(t, 0.0, 0.3, 2.5, 2.8) + fade(t, 2.8, 3.1, 5.6, 6.0)
    # captures, every 0.1s while moving
    for tc in np.arange(0.3, 1.0001, 0.1):
        if t >= tc:
            p = phi_at(tc); x, y = ring_xy("low", p)
            a = 1 - smooth(seg(t, 2.5, 2.8))
            r = math.radians(p); bd = (math.sin(r), -math.cos(r) * 0.35)
            sc.add(math.sin(r) >= 0, lambda d, x=x, y=y, a=a, bd=bd: capture(d, x, y - 70, a, False, bd))
    for tc in np.arange(3.1, 5.3001, 0.1):
        if t >= tc:
            p = phi_at(tc); x, y = ring_xy("low", p)
            a = smooth(seg(t, tc, tc + 0.12)) * (1 - smooth(seg(t, 5.6, 6.0)))
            sc.add(math.sin(math.radians(p)) >= 0, lambda d, x=x, y=y, a=a: capture(d, x, y - 70, a, True))
    p = phi_at(t); x, y = ring_xy("low", p); s = depth_scale(p)
    fast = 0.3 < t < 1.0
    if fast:
        for k in range(1, 6):
            pk = phi_at(t - k * 0.025); xk, yk = ring_xy("low", pk)
            sc.add(math.sin(math.radians(pk)) >= 0,
                   lambda d, xk=xk, yk=yk, k=k, pk=pk: phone(d, xk, yk, depth_scale(pk), alpha * 0.22 * (1 - k / 6), 0, False))
    sc.add(math.sin(math.radians(p)) >= 0, lambda d: phone(d, x, y, s, alpha))
    sc.badge = [("ng", 1.1, 2.4), ("ok", 4.6, 5.65)]
    sc.title = "Move slowly"
    sc.caps = [("Moving fast leaves blurry gaps", 0.35, 2.55), ("Slow passes overlap every frame", 3.1, 5.75)]
    return D


TRACK = [(540, 336), (618, 392), (590, 478), (490, 478), (462, 392), (540, 418), (510, 582), (572, 586), (578, 520)]
rng = random.Random(3)
TRACK_DIR = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in TRACK]

def scene_turn(t, sc):
    D = 6.0
    sc.add(False, lambda d: ring(d, "low", 0.35, False)); sc.add(True, lambda d: ring(d, "low", 0.35, True))
    if t < 0.3:   p = 150
    elif t < 1.0: p = lerp(150, 128, smooth(seg(t, 0.3, 1.0)))
    elif t < 2.8: p = 128
    elif t < 3.1: p = 150
    elif t < 5.4: p = lerp(150, 30, smooth(seg(t, 3.1, 5.4)))
    else:         p = 30
    aim = 0.0; jit = 0.0
    if 1.0 <= t < 1.7:
        u = seg(t, 1.0, 1.7)
        aim = 78 * math.sin(math.pi * u) * math.sin(2.6 * math.pi * u)
        jit = 5 * math.sin(t * 90) * math.sin(math.pi * u)
    p2 = p + jit
    x, y = ring_xy("low", p2); s = depth_scale(p2)
    alpha = fade(t, 0.0, 0.3, 2.5, 2.8) + fade(t, 2.8, 3.1, 5.6, 6.0)
    lost = smooth(seg(t, 1.05, 1.35)) * (1 - smooth(seg(t, 2.5, 2.8)))
    if 1.0 <= t < 1.7:
        for k in range(1, 5):
            tk = t - k * 0.02; u = seg(tk, 1.0, 1.7)
            ak = 78 * math.sin(math.pi * u) * math.sin(2.6 * math.pi * u)
            sc.add(True, lambda d, k=k, ak=ak: phone(d, x, y, s, alpha * 0.18 * (1 - k / 5), ak, True))
    sc.add(math.sin(math.radians(p2)) >= 0, lambda d: phone(d, x, y, s, alpha, aim))
    # tracking features drawn on top of the daruma
    show = fade(t, 0.15, 0.45, 5.6, 6.0)
    def feats(d):
        for (fx, fy), (dx, dy) in zip(TRACK, TRACK_DIR):
            ox = dx * 46 * lost + math.sin(t * 40 + fx) * 4 * lost
            oy = dy * 46 * lost + math.cos(t * 37 + fy) * 4 * lost
            c = tuple(int(lerp(a, b, lost)) for a, b in zip(LAV, ORANGE))
            a = show * (1 - 0.65 * lost)
            r = 8 + 2 * math.sin(t * 6 + fx) * (1 - lost)
            X, Y = fx + ox, fy + oy
            d.ellipse([S((X - r, Y - r)), S((X + r, Y + r))], outline=col(c, a), width=3 * SS)
            d.ellipse([S((X - 2.5, Y - 2.5)), S((X + 2.5, Y + 2.5))], fill=col(c, a))
    sc.add(True, feats)
    sc.badge = [("ng", 1.35, 2.45), ("ok", 4.5, 5.65)]
    sc.title = "No sudden turns"
    sc.caps = [("A quick swing makes tracking lose its place", 1.05, 2.6), ("Keep the subject centered as you circle", 3.1, 5.75)]
    return D


# ---- consistent 3D projection for the angles scene ----------------------
SINA, COSA = 0.33, 0.944
FZ = 246.0                       # flower centre height (3D units)
TZ = 232.0                       # where the phone aims

def proj(r, phi, z):
    ra = math.radians(phi)
    return CX + r * math.cos(ra), CY - z * COSA + r * math.sin(ra) * SINA

def rrect_poly(w, h, rad, n=6):
    pts = []
    for cx, cy, a0 in ((w / 2 - rad, -h / 2 + rad, -90), (w / 2 - rad, h / 2 - rad, 0),
                       (-w / 2 + rad, h / 2 - rad, 90), (-w / 2 + rad, -h / 2 + rad, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return pts

def phone_tilted(d, x, y, s, a, rot, hs, tx, ty):
    """phone with pitch shown as in-plane tilt + foreshortening; frustum to (tx,ty)."""
    if a <= 0: return
    ang = math.atan2(ty - y, tx - x)
    L = math.hypot(tx - x, ty - y) * 1.05
    ex, ey = x + math.cos(ang) * L, y + math.sin(ang) * L
    nx, ny = -math.sin(ang), math.cos(ang)
    hw = 88
    tri = [(x, y), (ex + nx * hw, ey + ny * hw), (ex - nx * hw, ey - ny * hw)]
    d.polygon([S(p) for p in tri], fill=col(PURPLE, 0.20 * a))
    d.line([S(tri[0]), S(tri[1])], fill=col(LAV, 0.55 * a), width=2 * SS)
    d.line([S(tri[0]), S(tri[2])], fill=col(LAV, 0.55 * a), width=2 * SS)
    w, h = 56 * s, 104 * s * hs
    cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    R = lambda px, py: (x + px * cr - py * sr, y + px * sr + py * cr)
    body = [R(px, py) for px, py in rrect_poly(w, h, min(12 * s, h / 2 - 1))]
    d.polygon([S(p) for p in body], fill=col(IVORY, a), outline=col(DARK, a), width=int(3 * SS))
    lx, ly = R(-w / 2 + 15 * s, -h / 2 + 15 * s * hs)
    lr = 6 * s
    d.ellipse([S((lx - lr, ly - lr)), S((lx + lr, ly + lr))], fill=col(DARK, a))

# camera path: (t0, t1, r0, r1, z0, z1, phi0, phi1)
A_PATH = [
    (0.30, 2.00, 380, 380, 40, 40, 160, 520),      # 見上げる : low, looking up
    (2.00, 2.35, 380, 330, 40, 246, 160, 160),
    (2.35, 4.05, 330, 330, 246, 246, 160, -200),   # 正面    : eye level
    (4.05, 4.40, 330, 210, 246, 420, -200, -200),
    (4.40, 6.10, 210, 210, 420, 420, 160, 520),    # 覗き込む : high, looking down
    (6.10, 7.00, 210, 45, 420, 440, 160, 340),     # 真上    : spiral to the top
]
A_END = 7.0

def a_state(tt):
    if tt <= A_PATH[0][0]:
        _, _, r, _, z, _, p, _ = A_PATH[0]; return r, z, p
    for t0, t1, r0, r1, z0, z1, p0, p1 in A_PATH:
        if tt <= t1:
            u = seg(tt, t0, t1)
            e = smooth(u) if (r0 != r1 or z0 != z1) else lerp(u, smooth(u), 0.5)
            return lerp(r0, r1, e), lerp(z0, z1, e), lerp(p0, p1, e)
    _, _, _, r, _, z, _, p = A_PATH[-1]; return r, z, p

# coverage dome around the flower
DOME_R = 178
DOME = []
for lat in (-24, 0, 30, 56, 82):
    n = {-24: 12, 0: 14, 30: 12, 56: 9, 82: 4}[lat]
    for k in range(n):
        lon = k * 360 / n + (15 if lat in (30, 82) else 0)
        la, lo = math.radians(lat), math.radians(lon)
        v = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
        DOME.append(dict(lat=lat, lon=lon, v=v))
_ts = np.arange(0, A_END, 1 / 240)
for dot in DOME:
    dot["t"] = None
for tt in _ts:
    r, z, p = a_state(tt)
    pv = np.array([r * math.cos(math.radians(p)), r * math.sin(math.radians(p)), z - FZ])
    pv /= np.linalg.norm(pv)
    for dot in DOME:
        if dot["t"] is None and tt > 0.3 and float(pv @ dot["v"]) > math.cos(math.radians(27)):
            dot["t"] = float(tt)

def dashed3d(d, r, z, a0, a1, alpha):
    n = int(abs(a1 - a0)) + 2
    pts = [proj(r, lerp(a0, a1, i / (n - 1)), z) for i in range(n)]
    for i in range(0, n - 7, 13):
        d.line([S(q) for q in pts[i:i + 7]], fill=col(IVORY, alpha), width=3 * SS)

def scene_angles(t, sc):
    D = 8.4
    for r, z in ((380, 40), (330, 246), (210, 420)):
        sc.add(False, lambda d, r=r, z=z: dashed3d(d, r, z, 180, 360, 0.22))
        sc.add(True, lambda d, r=r, z=z: dashed3d(d, r, z, 0, 180, 0.22))
    alpha = fade(t, 0.0, 0.3, 7.8, 8.4)
    # coverage dome
    for dot in DOME:
        v = dot["v"]
        X = CX + DOME_R * v[0]
        Y = CY - (FZ + DOME_R * v[2]) * COSA + DOME_R * v[1] * SINA
        front = v[1] >= 0
        lit = 0.0 if dot["t"] is None else smooth(seg(t, dot["t"], dot["t"] + 0.18))
        pop = 0.0 if dot["t"] is None else math.sin(math.pi * seg(t, dot["t"], dot["t"] + 0.35))
        dim = 0.55 if front else 0.35
        def f(d, X=X, Y=Y, lit=lit, pop=pop, dim=dim):
            a = fade(t, 0.0, 0.4, 7.8, 8.4)
            r0 = 5
            d.ellipse([S((X - r0, Y - r0)), S((X + r0, Y + r0))], fill=col(IVORY, 0.32 * dim * a * (1 - lit)))
            if lit > 0:
                r1 = 6 + 3 * pop
                d.ellipse([S((X - r1, Y - r1)), S((X + r1, Y + r1))], fill=col(LAV, (0.55 + 0.45 * dim) * lit * a))
                if pop > 0:
                    r2 = 8 + 10 * pop
                    d.ellipse([S((X - r2, Y - r2)), S((X + r2, Y + r2))], outline=col(LAV, 0.5 * pop * a), width=2 * SS)
        sc.add(front, f)
    # travelled path (faint)
    tr = 1 - smooth(seg(t, 7.8, 8.4))
    ts = [tt for tt in np.arange(0.3, min(t, A_END), 1 / 60)]
    for t0_, t1_ in zip(ts, ts[1:]):
        r0, z0, p0 = a_state(t0_); r1, z1, p1 = a_state(t1_)
        front = math.sin(math.radians(p0)) >= 0 or z0 > 430
        P0, P1 = proj(r0, p0, z0), proj(r1, p1, z1)
        sc.add(front, lambda d, P0=P0, P1=P1: d.line([S(P0), S(P1)], fill=col(LAV, 0.25 * tr), width=3 * SS))
    # phone
    r, z, p = a_state(t)
    x, y = proj(r, p, z)
    tx, ty = proj(0, 0, TZ)
    pitch = math.degrees(math.atan2(TZ - z, max(r, 1)))
    cphi, sphi = math.cos(math.radians(p)), math.sin(math.radians(p))
    rot = pitch * cphi
    hs = 1 - 0.55 * abs(math.sin(math.radians(pitch))) * abs(sphi) - 0.35 * max(0, (z - 430) / 20) * 0
    if r < 120:                                  # overhead: phone lies almost flat
        hs = min(hs, 0.45 + 0.5 * (r - 45) / 75)
    s = 0.84 + 0.16 * sphi if r > 120 else 0.9
    front = sphi >= 0 or z > 430
    sc.add(front, lambda d: phone_tilted(d, x, y, s, alpha, rot, hs, tx, ty))
    # pose indicator: look-up / eye-level / look-down / overhead
    PH = [(0.30, 2.00, -38), (2.35, 4.05, 0), (4.40, 6.10, 38), (6.10, 7.00, 90)]
    LABELS = ["Look up", "Eye level", "Look down", "Overhead"]
    def poses(d):
        a = fade(t, 0.0, 0.4, 7.8, 8.4)
        for k, (p0, p1, th) in enumerate(PH):
            cx, cy = 540 + (k - 1.5) * 160, 858
            active = p0 <= t < p1
            done = t >= p1
            on = smooth(seg(t, p0, p0 + 0.2)) if t >= p0 else 0.0
            ring_c = LAV if (active or done) else IVORY
            d.ellipse([S((cx - 32, cy - 32)), S((cx + 32, cy + 32))],
                      fill=col(PURPLE, (0.35 if done else 0.9 * on if active else 0) * a),
                      outline=col(ring_c, (0.9 if (active or done) else 0.3) * a), width=2 * SS)
            r = math.radians(th); ux, uy = math.cos(r), math.sin(r)
            nx, ny = -uy, ux
            bx, by = cx - 11 * ux, cy - 11 * uy
            gc = WHITE if (active or done) else IVORY
            ga = (1.0 if (active or done) else 0.45) * a
            cone = [(bx + 3 * ux, by + 3 * uy), (cx + 17 * ux + 10 * nx, cy + 17 * uy + 10 * ny),
                    (cx + 17 * ux - 10 * nx, cy + 17 * uy - 10 * ny)]
            d.polygon([S(q) for q in cone], fill=col(gc, 0.45 * ga))
            hw, hh = 4.5, 12
            body = [(bx + ux * px + nx * py, by + uy * px + ny * py) for px, py in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            d.polygon([S(q) for q in body], fill=col(gc, ga))
            text_abs(d, cx, cy + OY + 54, LABELS[k], F_LAB, gc if (active or done) else IVORY,
                     (1.0 if active else 0.8 if done else 0.45) * a)
    sc.add(True, poses)
    sc.badge = [("ok", 7.05, 7.85)]
    sc.title = "Cover every angle"
    sc.sub = ("Light up every dot around the subject", 0.0, 99)
    return D


# ======================================================================
def scene_pivot(t, sc):
    D = 6.0
    sc.add(False, lambda d: ring(d, "low", 0.35, False)); sc.add(True, lambda d: ring(d, "low", 0.35, True))
    alpha = fade(t, 0.0, 0.3, 2.5, 2.8) + fade(t, 2.8, 3.1, 5.6, 6.0)
    if t < 2.8:
        p = 150
        u = seg(t, 0.3, 2.3)
        aim = 52 * math.sin(2 * math.pi * u) if 0 < u < 1 else 0.0
        x, y = ring_xy("low", p); s_ = depth_scale(p)
        for k, tc in enumerate(np.arange(0.35, 2.3, 0.14)):
            if t >= tc:
                a = smooth(seg(t, tc, tc + 0.1)) * (1 - smooth(seg(t, 2.5, 2.8)))
                sc.add(True, lambda d, k=k, a=a: capture(d, x - 74 + k * 5, y - 60 - k * 5, a, False))
        sc.add(True, lambda d: phone(d, x, y, s_, alpha, aim))
    else:
        p = lerp(150, 30, smooth(seg(t, 3.1, 5.3)))
        for tc in np.arange(3.1, 5.3001, 0.12):
            if t >= tc:
                pc = lerp(150, 30, smooth(seg(tc, 3.1, 5.3))); cx_, cy_ = ring_xy("low", pc)
                a = smooth(seg(t, tc, tc + 0.12)) * (1 - smooth(seg(t, 5.6, 6.0)))
                sc.add(True, lambda d, cx_=cx_, cy_=cy_, a=a: capture(d, cx_, cy_ - 70, a, True))
        x, y = ring_xy("low", p); s_ = depth_scale(p)
        sc.add(True, lambda d: phone(d, x, y, s_, alpha))
    sc.badge = [("ng", 1.0, 2.4), ("ok", 4.6, 5.65)]
    sc.title = "Walk around, don't pivot"
    sc.caps = [("Turning in place adds no depth", 0.35, 2.55), ("Walk around so each shot sees a new side", 3.1, 5.75)]
    return D


def scene_loop(t, sc):
    D = 7.4
    sc.add(False, lambda d: ring(d, "low", 0.35, False)); sc.add(True, lambda d: ring(d, "low", 0.35, True))
    T0, T1 = 0.4, 4.4
    p = lerp(160, 520, smooth(seg(t, T0, T1)) * 0.15 + seg(t, T0, T1) * 0.85)
    snap = smooth(seg(t, 4.45, 5.0))
    show = fade(t, 0.0, 0.3, 6.9, 7.4)
    def est(pp, k):
        dr = k * (pp - 160) / 360
        rx, ry, _ = RINGS["low"]; r = math.radians(pp)
        return CX + rx * (1 + 0.15 * dr) * math.cos(r), CY + ry * (1 + 0.15 * dr) * math.sin(r) + 34 * dr
    kd = 1 - snap
    if t > T0:
        pts = np.linspace(160, p, max(2, int(p - 160) + 2))
        for a0, a1 in zip(pts, pts[1:]):
            P0, P1 = est(a0, kd), est(a1, kd)
            sc.add(math.sin(math.radians(a0)) >= 0,
                   lambda d, P0=P0, P1=P1: d.line([S(P0), S(P1)], fill=col(LAV, 0.85 * show), width=4 * SS))
    mx, my = ring_xy("low", 160)
    def marker(d):
        pul = math.sin(math.pi * seg(t, 4.45, 5.2))
        d.ellipse([S((mx - 16, my - 16)), S((mx + 16, my + 16))], outline=col(IVORY, 0.85 * show), width=3 * SS)
        d.ellipse([S((mx - 6, my - 6)), S((mx + 6, my + 6))], fill=col(LAV, show))
        if pul > 0:
            r = 18 + 40 * seg(t, 4.45, 5.2)
            d.ellipse([S((mx - r, my - r)), S((mx + r, my + r))], outline=col(LAV, 0.8 * pul * show), width=4 * SS)
        text_abs(d, mx - 70, my + OY + 2, "Start", F_LAB, IVORY, 0.85 * show)
    sc.add(True, marker)
    x, y = ring_xy("low", p); s_ = depth_scale(p)
    alpha = fade(t, 0.0, 0.3, 6.9, 7.4)
    sc.add(math.sin(math.radians(p)) >= 0, lambda d: phone(d, x, y, s_, alpha))
    sc.badge = [("ok", 4.7, 6.7)]
    sc.title = "Finish where you started"
    sc.caps = [("Small errors add up as you go around", 0.6, 4.35), ("Closing the loop snaps errors back into place", 4.5, 6.95)]
    return D


_oa = np.asarray(OBJ, np.float32) / 255
_u = np.linspace(0, 1, _oa.shape[1], dtype=np.float32)[None, :]
SHADE_A = _oa[..., 3] * np.clip((0.56 - _u) / 0.56, 0, 1) * 0.62
HI_A = _oa[..., 3] * np.clip((_u - 0.56) / 0.44, 0, 1) * 0.22

def scene_light(t, sc):
    D = 6.4
    sc.add(False, lambda d: ring(d, "low", 0.3, False)); sc.add(True, lambda d: ring(d, "low", 0.3, True))
    hf = 1 - smooth(seg(t, 2.75, 3.2))
    def obj(im):
        draw_object(im)
        if hf > 0:
            sh = np.zeros(_oa.shape, np.float32); sh[..., 3] = SHADE_A * hf
            im.paste(Image.fromarray((sh * 255).astype(np.uint8), "RGBA"), OBJ_AT, Image.fromarray((sh[..., 3] * 255).astype(np.uint8)))
            hi = np.ones(_oa.shape, np.float32); hi[..., 3] = HI_A * hf
            hi8 = (hi * 255).astype(np.uint8)
            im.paste(Image.fromarray(hi8, "RGBA"), OBJ_AT, Image.fromarray(hi8[..., 3]))
    sc.obj = obj
    # hard cast shadow on the ground
    def cast(d):
        if hf <= 0: return
        d.polygon([S(q) for q in [(486, 600), (596, 600), (880, 688), (808, 722)]], fill=col(DARK, 0.55 * hf))
        d.ellipse([S((770, 672)), S((910, 724))], fill=col(DARK, 0.45 * hf))
    sc.add(False, cast)
    # the photographer's own shadow sweeping over the subject
    def person(d):
        u = seg(t, 0.9, 2.5)
        if u <= 0 or u >= 1 or hf <= 0: return
        x = lerp(180, 900, u)
        a = math.sin(math.pi * u) * hf
        for k in range(9):
            f = 1 - k / 9
            d.ellipse([S((x - 95 * f, 300 + 60 * (1 - f))), S((x + 95 * f, 700 - 30 * (1 - f)))], fill=col(DARK, 0.095 * a))
            d.ellipse([S((x - 52 * f, 200 + 30 * (1 - f))), S((x + 52 * f, 310))], fill=col(DARK, 0.08 * a))
    sc.add(True, person)
    # sky icon: hard sun -> soft cloud
    def sky(d):
        cx, cy = 190, 230
        a = fade(t, 0.0, 0.3, 6.0, 6.4)
        if hf > 0:
            sa = hf * a
            d.ellipse([S((cx - 30, cy - 30)), S((cx + 30, cy + 30))], fill=col(ORANGE, sa))
            for k in range(8):
                r = math.radians(k * 45 + t * 20)
                d.line([S((cx + 42 * math.cos(r), cy + 42 * math.sin(r))), S((cx + 60 * math.cos(r), cy + 60 * math.sin(r)))],
                       fill=col(ORANGE, sa), width=5 * SS)
        ca = (1 - hf) * a
        if ca > 0:
            for k in range(6):
                g = 70 - k * 8
                d.ellipse([S((cx - g * 1.4, cy - g * 0.8)), S((cx + g * 1.4, cy + g * 0.8))], fill=col(IVORY, 0.025 * ca))
            for ox, oy, r in ((-30, 8, 26), (0, -8, 34), (32, 6, 26), (0, 14, 26)):
                d.ellipse([S((cx + ox - r, cy + oy - r)), S((cx + ox + r, cy + oy + r))], fill=col(IVORY, 0.92 * ca))
    sc.add(True, sky)
    alpha = fade(t, 0.0, 0.3, 6.0, 6.4)
    p = 150 if t < 3.2 else lerp(150, 40, smooth(seg(t, 3.2, 5.6)))
    x, y = ring_xy("low", p); s_ = depth_scale(p)
    sc.add(math.sin(math.radians(p)) >= 0, lambda d: phone(d, x, y, s_, alpha))
    sc.badge = [("ng", 1.3, 2.6), ("ok", 4.4, 5.75)]
    sc.title = "Soft, even light"
    sc.caps = [("Harsh sun and your own shadow leave blotches", 0.35, 2.85), ("Overcast or indirect light keeps colors even", 3.2, 5.95)]
    return D


SBY = 660                                    # base line of the three subjects
SUBX = {"glass": 250, "plain": 540, "tex": 830}
_k = 0.8
OBJ_SMALL = OBJ.resize((int(OBJ.width * _k), int(OBJ.height * _k)), Image.LANCZOS)
OBJ_SMALL_AT = (int((SUBX["tex"] - (540 - 380) * _k) * SS), int((SBY - (606 - 300) * _k + OY) * SS))
SUB_DOTS = {
    "glass": [(-50, -70), (40, -95), (-20, -150), (55, -30), (-60, -20), (10, -230), (30, -170)],
    "plain": [(-40, -90), (35, -60), (0, -130), (-20, -40), (45, -110)],
    "tex":   [((x - 540) * _k, (y - 606) * _k) for x, y in TRACK],
}

def vase(d, cx, by, a):
    d.ellipse([S((cx - 95, by - 12)), S((cx + 95, by + 12))], fill=col((0, 0, 0), 0.3 * a))
    body = []
    for k in range(0, 361, 6):
        r = math.radians(k)
        body.append((cx + 88 * math.cos(r), by - 92 + 88 * math.sin(r)))
    neck = [(cx - 28, by - 250), (cx + 28, by - 250), (cx + 24, by - 170), (cx - 24, by - 170)]
    d.polygon([S(q) for q in neck], fill=col((200, 225, 255), 0.13 * a), outline=col((220, 236, 255), 0.75 * a), width=3 * SS)
    d.polygon([S(q) for q in body], fill=col((200, 225, 255), 0.13 * a), outline=col((220, 236, 255), 0.75 * a), width=3 * SS)
    d.ellipse([S((cx - 30, by - 258)), S((cx + 30, by - 244))], outline=col((220, 236, 255), 0.75 * a), width=3 * SS)
    d.arc([S((cx - 70, by - 160)), S((cx + 40, by - 30))], 150, 230, fill=col(WHITE, 0.8 * a), width=6 * SS)
    d.arc([S((cx - 20, by - 150)), S((cx + 74, by - 40))], 300, 340, fill=col(WHITE, 0.6 * a), width=4 * SS)
    d.line([S((cx - 12, by - 240)), S((cx - 12, by - 185))], fill=col(WHITE, 0.6 * a), width=4 * SS)
    # refracted, flipped hint of the dashed ground ring behind it
    for k in range(5):
        y = by - 70 + k * 9
        d.line([S((cx - 50 + k * 6, y)), S((cx - 25 + k * 6, y + 3))], fill=col(IVORY, 0.18 * a), width=2 * SS)

def ball(d, cx, by, a):
    d.ellipse([S((cx - 92, by - 12)), S((cx + 92, by + 12))], fill=col((0, 0, 0), 0.32 * a))
    R = 86; ccy = by - R
    for k in range(24):
        f = 1 - k / 24
        c = tuple(int(lerp(150, 247, (k / 23) ** 0.8)) for _ in range(3))
        c = (int(c[0] * 1.0), int(c[1] * 0.98), int(c[2] * 0.95))
        r = R * f
        ox, oy = -26 * (1 - f), -30 * (1 - f)
        d.ellipse([S((cx + ox - r, ccy + oy - r)), S((cx + ox + r, ccy + oy + r))], fill=col(c, a))

def scene_subjects(t, sc):
    D = 7.6
    sc.no_ground = True
    A = fade(t, 0.0, 0.3, 7.1, 7.6)
    PH = {"glass": (0.3, 2.5), "plain": (2.6, 4.8), "tex": (4.9, 7.0)}
    def floor(d):
        for name, x in SUBX.items():
            p0, p1 = PH[name]
            act = fade(t, p0, p0 + 0.25, p1 - 0.1, p1 + 0.15)
            d.ellipse([S((x - 120, SBY - 26)), S((x + 120, SBY + 26))], fill=col(IVORY, (0.05 + 0.08 * act) * A),
                      outline=col(LAV, 0.55 * act * A), width=2 * SS)
        vase(d, SUBX["glass"], SBY, A)
        ball(d, SUBX["plain"], SBY, A)
    sc.add(False, floor)
    sc.obj = lambda im: im.paste(OBJ_SMALL, OBJ_SMALL_AT, OBJ_SMALL) if A > 0.99 else _paste_fade(im, A)
    def dots(d):
        for name, x in SUBX.items():
            p0, p1 = PH[name]
            act = fade(t, p0 + 0.2, p0 + 0.45, p1 - 0.2, p1)
            if act <= 0: continue
            for i, (dx, dy) in enumerate(SUB_DOTS[name]):
                if name == "glass":
                    jx = 18 * math.sin(t * 11 + i * 2.1); jy = 14 * math.cos(t * 9 + i * 1.3)
                    fl = clamp(0.5 + 0.9 * math.sin(t * 17 + i * 2.7))
                    c, a = ORANGE, act * fl
                elif name == "plain":
                    jx = jy = 0
                    ph = (t * 1.6 + i * 0.37) % 1.0
                    fl = math.sin(math.pi * clamp(ph / 0.35)) if ph < 0.35 else 0
                    c, a = ORANGE, act * fl * 0.9
                else:
                    jx = jy = 0
                    c, a = LAV, act
                X, Y = x + dx + jx, SBY + dy + jy
                r = 8 + (1.5 * math.sin(t * 6 + i) if name == "tex" else 0)
                d.ellipse([S((X - r, Y - r)), S((X + r, Y + r))], outline=col(c, a), width=3 * SS)
                d.ellipse([S((X - 2.5, Y - 2.5)), S((X + 2.5, Y + 2.5))], fill=col(c, a))
        for name, lab in (("glass", "Transparent"), ("plain", "Plain"), ("tex", "Textured")):
            p0, _ = PH[name]
            on = t >= p0
            text_abs(d, SUBX[name], SBY + OY + 62, lab, F_LAB, LAV if on else IVORY, (0.95 if on else 0.5) * A)
    sc.add(True, dots)
    sc.badge = [("warn", 1.0, 7.05, SUBX["glass"], SBY - 330), ("warn", 3.3, 7.05, SUBX["plain"], SBY - 330),
                ("ok", 5.6, 7.05, SUBX["tex"], SBY - 330)]
    sc.title = "Some subjects are harder"
    sc.caps = [("Glass and mirrors confuse tracking", 0.35, 2.55), ("Plain, featureless surfaces give nothing to track", 2.65, 4.85),
               ("Pick subjects with texture and detail", 4.95, 7.1)]
    return D

def _paste_fade(im, a):
    o = np.asarray(OBJ_SMALL).copy(); o[..., 3] = (o[..., 3] * a).astype(np.uint8)
    oi = Image.fromarray(o, "RGBA"); im.paste(oi, OBJ_SMALL_AT, oi)


def scene_cool(t, sc):
    """Three tips for training without the phone throttling: each one is a hot phone (red meter,
    heat waves) that turns cool (lavender meter) once the cause is taken away."""
    D = 7.9
    sc.obj = lambda im: None; sc.no_ground = True
    PX, PY, PW, PH = 500, 520, 168, 312      # the phone
    MX = 900                                 # the heat meter
    T0, PANEL = 0.2, 2.5
    CHANGE = (1.25, 1.75)                    # when the cause goes away, within a panel

    def rrect(d, x, y, w, h, r, **kw):
        d.rounded_rectangle([S((x - w / 2, y - h / 2)), S((x + w / 2, y + h / 2))], radius=int(r * SS), **kw)

    def phone_body(d, a, x=PX, y=PY):
        rrect(d, x, y, PW, PH, 28, fill=col(IVORY, a), outline=col(DARK, a), width=3 * SS)
        rrect(d, x, y, PW - 24, PH - 28, 18, fill=col(DARK, a))
        # training progress on the screen
        rrect(d, x, y + 70, 100, 10, 5, fill=col(LAV, 0.25 * a))
        rrect(d, x - 20, y + 70, 60, 10, 5, fill=col(LAV, 0.9 * a))
        d.ellipse([S((x - 14, y - 26)), S((x + 14, y + 2))], fill=col(LAV, 0.85 * a))

    def meter(d, a, cool):
        x, y0, y1, w = MX, 400, 680, 34
        c = tuple(int(lerp(CORAL[i], LAV[i], cool)) for i in range(3))
        rrect(d, x, (y0 + y1) / 2, w, y1 - y0, w / 2, outline=col(IVORY, 0.55 * a), width=3 * SS)
        top = y1 - (y1 - y0 - 24) * lerp(0.92, 0.28, cool)
        rrect(d, x, (top + y1) / 2, w - 12, y1 - top, (w - 12) / 2, fill=col(c, 0.95 * a))
        d.ellipse([S((x - 26, y1 - 12)), S((x + 26, y1 + 40))], fill=col(c, a))

    def waves(d, a, cool):
        a *= 1 - cool
        if a <= 0: return
        for k in (-1, 0, 1):
            pts = [(PX + k * 46 + 9 * math.sin(i / 5.0 + t * 7 + k), PY - PH / 2 - 24 - i * 2.2) for i in range(34)]
            d.line([S(p) for p in pts], fill=col(ORANGE, 0.8 * a), width=5 * SS, joint="curve")

    def panel(i, behind, front):
        t0 = T0 + i * PANEL
        a = fade(t, t0, t0 + 0.25, t0 + PANEL - 0.25, t0 + PANEL)
        cool = smooth(seg(t, t0 + CHANGE[0], t0 + CHANGE[1]))
        out = cool                                  # the cause leaves as the phone cools
        if a <= 0: return
        sc.add(True, lambda d: behind(d, a, out))
        sc.add(True, lambda d: phone_body(d, a))
        sc.add(True, lambda d: front(d, a, out))
        sc.add(True, lambda d: waves(d, a, cool))
        sc.add(True, lambda d: meter(d, a, cool))
        sc.badge += [("ng", t0 + 0.3, t0 + CHANGE[0] - 0.05), ("ok", t0 + CHANGE[1] + 0.05, t0 + PANEL - 0.1)]

    sc.badge = []
    nothing = lambda d, a, out: None

    def cards(d, a, out):                           # other apps, swiped away
        for cx in (PX - 215, PX + 215):
            y = PY - out * 300
            rrect(d, cx, y, 120, 220, 18, fill=col(IVORY, 0.42 * a * (1 - out)), outline=col(LAV, 0.5 * a * (1 - out)), width=2 * SS)
            rrect(d, cx, y - 82, 70, 12, 6, fill=col(PURPLE, 0.8 * a * (1 - out)))
    def cable(d, a, out):                           # the charger, pulled out
        dy = out * 130; ca = a * (1 - out)
        if ca <= 0: return
        yb = PY + PH / 2
        d.line([S((PX, yb + 38 + dy)), S((PX, yb + 130 + dy)), S((PX + 190, yb + 130 + dy))],
               fill=col(IVORY, 0.9 * ca), width=9 * SS, joint="curve")
        rrect(d, PX, yb + 20 + dy, 34, 44, 8, fill=col(IVORY, ca), outline=col(DARK, ca), width=3 * SS)
        bolt = [(10, -60), (-26, 8), (-2, 8), (-10, 60), (26, -12), (2, -12)]
        d.polygon([S((PX + 0.55 * x, PY - 95 + 0.55 * y)) for x, y in bolt], fill=col(GOLD, ca))
    def case(d, a, out):                            # the case, slid off
        ca = a * (1 - out)
        if ca <= 0: return
        rrect(d, PX - out * 360, PY + out * 30, PW + 40, PH + 40, 40, fill=col(PURPLE, 0.95 * ca), outline=col(LAV, 0.8 * ca), width=3 * SS)

    panel(0, cards, nothing)
    panel(1, nothing, cable)
    panel(2, case, nothing)
    sc.title = "Keep it cool"
    sc.caps = [("Close other apps before training", T0 + 0.1, T0 + PANEL - 0.05),
               ("Unplug the charger while it trains", T0 + PANEL + 0.1, T0 + 2 * PANEL - 0.05),
               ("Take the case off to shed heat", T0 + 2 * PANEL + 0.1, T0 + 3 * PANEL - 0.05)]
    return D


SCENES = {"cool": scene_cool, "slow": scene_slow, "turn": scene_turn, "angles": scene_angles,
          "pivot": scene_pivot, "loop": scene_loop, "light": scene_light, "subjects": scene_subjects}


def render(t):
    sc = Scene(); sc.badge = []; sc.title = ""; sc.sub = None; sc.caps = []; sc.obj = draw_object; sc.no_ground = False
    D = SCENES[args.scene](t, sc)
    im = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(im, "RGBA")
    if not sc.no_ground: ground(d)
    for fn in sc.back: fn(d)
    sc.obj(im)
    for fn in sc.front: fn(d)
    for b in sc.badge: badge(d, b[0], t, *b[1:])
    text_abs(d, 540, 86, sc.title, F_TITLE, IVORY, 1.0)
    if sc.sub:
        text_abs(d, 540, 146, sc.sub[0], F_SUB, LAV, 0.9)
    for txt, a0, a1 in sc.caps:
        text_abs(d, 540, 1000, txt, F_CAP, IVORY, 0.88 * fade(t, a0, a0 + 0.25, a1 - 0.25, a1))
    return im.resize((W, H), Image.LANCZOS), D


_, DUR = render(0.0)
times = [args.only] if args.only >= 0 else [i / args.fps for i in range(int(round(DUR * args.fps)))]
for i, t in enumerate(times):
    im, _ = render(t)
    im.save(os.path.join(args.out, f"f{i:04d}.png"))
print(args.scene, "frames:", len(times), "dur:", DUR)
