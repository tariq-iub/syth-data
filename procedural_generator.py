"""
CopperSpec-Synth :: procedural_generator.py
===========================================
Training-free copper specimen synthesiser (instant baseline).

Pipeline per image
------------------
1. ``generate_copper_base``  - metallic copper micrograph built from multi-
   octave Perlin-style gradient noise (rolled/brushed relief) + Worley /
   Voronoi cellular noise (crystalline grain structure & boundaries), shaded
   with a height-field Lambertian term + tight specular so grains catch light
   differently, plus anisotropic streaking and a microscope vignette.
2. ``inject_corrosion``      - corrosion spots whose colours are sampled from
   the DB-derived KMeans palette / GaussianMixture (LAB space), placed with
   Bridson Poisson-disk sampling, feathered organic pit masks, oxide rim
   darkening and granular two-tone patina texture, alpha blended.
3. ``generate_from_stats``   - ties 1+2 to models/color_stats.npz and writes
   PNGs into outputs/synthetic/.

Hardware notes: pure NumPy/SciPy on CPU; a 512x512 mixed-corrosion image
renders in well under 5 s on a laptop CPU. No GPU code paths at all.

CLI:
    python procedural_generator.py --type mixed --corrosion 30
    python procedural_generator.py --type healthy --seed 7
    python procedural_generator.py --batch 8 --out outputs/synthetic
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import color
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs", "synthetic")


# ===========================================================================
# Noise primitives (vectorised; no per-pixel Python loops)
# ===========================================================================
def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def gradient_noise(size: int, freq: float, seed: int) -> np.ndarray:
    """Classic 2-D Perlin gradient noise, lattice resolution ~freq cells,
    output normalised to [0,1]. Fully vectorised."""
    rng = np.random.default_rng(seed % (2 ** 31))
    gh = max(2, int(round(freq)))
    gw = max(2, int(round(freq)))
    ang = rng.random((gh + 1, gw + 1)).astype(np.float32) * (2 * np.pi)
    gyy, gxx = np.cos(ang), np.sin(ang)

    ys = np.linspace(0, gh, size, dtype=np.float32, endpoint=False)
    xs = np.linspace(0, gw, size, dtype=np.float32, endpoint=False)
    y0 = np.floor(ys).astype(np.int32); x0 = np.floor(xs).astype(np.int32)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    sy = _smoothstep(fy); sx = _smoothstep(fx)

    n00 = gyy[y0][:, x0] * fy + gxx[y0][:, x0] * fx
    n10 = gyy[y0 + 1][:, x0] * (fy - 1) + gxx[y0 + 1][:, x0] * fx
    n01 = gyy[y0][:, x0 + 1] * fy + gxx[y0][:, x0 + 1] * (fx - 1)
    n11 = gyy[y0 + 1][:, x0 + 1] * (fy - 1) + gxx[y0 + 1][:, x0 + 1] * (fx - 1)
    out = ((n00 * (1 - sx) + n01 * sx) * (1 - sy) +
           (n10 * (1 - sx) + n11 * sx) * sy)
    out = out.astype(np.float32)
    return (out - out.min()) / (np.ptp(out) + 1e-8)


def value_noise(size: int, freq: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth interpolated white noise on a coarse lattice, [0,1]."""
    gh = gw = max(2, int(round(freq)))
    grid = rng.random((gh + 1, gw + 1)).astype(np.float32)
    ys = np.linspace(0, gh, size, dtype=np.float32, endpoint=False)
    xs = np.linspace(0, gw, size, dtype=np.float32, endpoint=False)
    y0 = np.floor(ys).astype(np.int32); x0 = np.floor(xs).astype(np.int32)
    fy = _smoothstep((ys - y0)[:, None]); fx = _smoothstep((xs - x0)[None, :])
    g00 = grid[np.ix_(y0, x0)]; g10 = grid[np.ix_(y0 + 1, x0)]
    g01 = grid[np.ix_(y0, x0 + 1)]; g11 = grid[np.ix_(y0 + 1, x0 + 1)]
    top = g00 * (1 - fx) + g01 * fx
    bot = g10 * (1 - fx) + g11 * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def perlin_octaves(size: int, octaves: int = 5, base_freq: float = 4.0,
                   persistence: float = 0.5, lacunarity: float = 2.0,
                   seed: int = 0) -> np.ndarray:
    """Multi-octave gradient noise normalised to [0,1]."""
    total = np.zeros((size, size), dtype=np.float32)
    amp, freq = 1.0, base_freq
    for o in range(octaves):
        total += amp * gradient_noise(size, freq, seed * 97 + o * 131 + 5)
        amp *= persistence
        freq = min(freq * lacunarity, size / 2.0)
    mn, mx = float(total.min()), float(total.max())
    return ((total - mn) / (mx - mn + 1e-8)).astype(np.float32)


def worley_noise(size: int, n_points: int, seed: int = 0) -> np.ndarray:
    """Worley (cellular / Voronoi F1) noise in [0,1] via FFT distance transform.
    Equivalent to nearest-seed distance of a random Voronoi diagram."""
    rng = np.random.default_rng(seed % (2 ** 31))
    n_points = max(4, int(n_points))
    field = np.ones((size, size), dtype=np.float32)
    iy = rng.integers(0, size, n_points)
    ix = rng.integers(0, size, n_points)
    field[iy, ix] = 0.0
    dist = ndi.distance_transform_edt(field)
    return (dist / (dist.max() + 1e-8)).astype(np.float32)


def poisson_disk(width: int, height: int, min_dist: float,
                 rng: np.random.Generator, k: int = 25,
                 max_points: int = 6000) -> np.ndarray:
    """Bridson's Poisson-disk sampling -> (M,2) float32 (x,y) pixel coords."""
    min_dist = max(2.0, float(min_dist))
    cell = min_dist / np.sqrt(2.0)
    gw, gh = int(np.ceil(width / cell)), int(np.ceil(height / cell))
    grid = {}
    points: List[Tuple[float, float]] = []

    def add(x, y):
        points.append((x, y))
        grid[(int(y / cell), int(x / cell))] = len(points) - 1

    add(float(rng.uniform(0, width)), float(rng.uniform(0, height)))
    active = [0]
    md2 = min_dist * min_dist
    while active and len(points) < max_points:
        ai = int(rng.integers(len(active)))
        px, py = points[active[ai]]
        placed = False
        for _ in range(k):
            ang = rng.random() * 2 * np.pi
            rad = min_dist * (1.0 + rng.random())
            qx, qy = px + np.cos(ang) * rad, py + np.sin(ang) * rad
            if not (0 <= qx < width and 0 <= qy < height):
                continue
            gy, gx = int(qy / cell), int(qx / cell)
            ok = True
            for yy in range(max(0, gy - 2), min(gh, gy + 3)):
                rowget = grid.get
                for xx in range(max(0, gx - 2), min(gw, gx + 3)):
                    j = rowget((yy, xx), -1)
                    if j >= 0:
                        ox, oy = points[j]
                        if (ox - qx) ** 2 + (oy - qy) ** 2 < md2:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                add(qx, qy)
                active.append(len(points) - 1)
                placed = True
                break
        if not placed:
            active.pop(ai)
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


# ===========================================================================
# Reference copper colours (fallback when DB stats unavailable)
# ===========================================================================
_COPPER_BASE_RGB = np.array([0.72, 0.31, 0.20], dtype=np.float32)   # raw copper
_COPPER_HI_RGB = np.array([0.98, 0.76, 0.55], dtype=np.float32)     # polished highlight
_COPPER_SH_RGB = np.array([0.42, 0.16, 0.09], dtype=np.float32)     # deep shade

FALLBACK_PALETTES: Dict[str, dict] = {
    "healthy": dict(
        palette_rgb=np.array([[184, 121, 89], [176, 96, 67], [203, 141, 101],
                              [159, 79, 54], [220, 160, 118], [146, 66, 44]],
                             dtype=np.float32)),
    # malachite Cu2CO3(OH)2, azurite, cuprite Cu2O, tenorite CuO
    "corroded": dict(
        palette_rgb=np.array([[27, 94, 66], [42, 121, 87], [23, 68, 110],
                              [154, 61, 35], [86, 124, 96], [35, 105, 124],
                              [58, 74, 58], [120, 84, 52]], dtype=np.float32)),
    "mixed": dict(
        palette_rgb=np.array([[184, 121, 89], [27, 94, 66], [176, 96, 67],
                              [42, 121, 87], [203, 141, 101], [23, 68, 110],
                              [154, 61, 35], [159, 79, 54]], dtype=np.float32)),
}


# ===========================================================================
# Copper base texture
# ===========================================================================
def generate_copper_base(size: int = 512, seed: int = 0,
                         brushed: bool = True) -> np.ndarray:
    """
    Metallic copper micrograph, float RGB in [0,1]:
      * fine Perlin octaves        -> rolled surface relief,
      * Worley cellular (FFT EDT)  -> crystalline grain interiors/boundaries,
      * height-field shading       -> per-grain Lambertian tilt + specular,
      * anisotropic streaks        -> rolling-direction brush marks,
      * channel roll               -> subtle R/B layer decorrelation,
      * radial vignette + sensor noise -> microscope optics feel.
    """
    rng = np.random.default_rng(seed % (2 ** 31))
    t0 = perlin_octaves(size, octaves=6, base_freq=6.0, persistence=0.55, seed=seed)
    t1 = perlin_octaves(size, octaves=4, base_freq=24.0, persistence=0.5, seed=seed + 1)
    big = worley_noise(size, n_points=max(24, (size // 26) ** 2), seed=seed + 2)
    fine = worley_noise(size, n_points=max(64, (size // 9) ** 2), seed=seed + 3)

    boundary = np.clip((fine - 0.86) / 0.14, 0, 1)          # dark grain edges
    interior = 1.0 - np.clip(big * 2.2, 0, 1)               # broad grain facets

    height = (0.50 * t0 + 0.22 * t1 + 0.42 * interior - 0.55 * boundary
              + 0.10 * (1.0 - fine))
    height = ndi.gaussian_filter(height.astype(np.float32), sigma=0.8)

    # lighting from a random direction in the image plane
    light_dir = float(rng.uniform(0, 2 * np.pi))
    gy, gx = np.gradient(height)
    slope = (gx * np.cos(light_dir) + gy * np.sin(light_dir)) * float(size) * 0.10
    lambert = np.clip(0.55 + slope, 0.0, 1.6)
    spec = np.clip(lambert - 0.75, 0, 1) ** 3.0             # tight Blinn-like lobes

    hnorm = (height - height.min()) / (np.ptp(height) + 1e-8)
    base = _COPPER_BASE_RGB[None, None, :] * (0.62 + 0.72 * hnorm)[..., None]
    base *= (0.72 + 0.55 * lambert)[..., None]
    base += (_COPPER_HI_RGB - _COPPER_BASE_RGB)[None, None, :] * (0.85 * spec)[..., None]
    base -= (_COPPER_BASE_RGB - _COPPER_SH_RGB)[None, None, :] * (0.85 * boundary)[..., None]

    if brushed:
        streak = value_noise(size, 220.0, rng)
        streak = ndi.uniform_filter(streak, size=(1, max(3, size // 64)))
        base *= (0.93 + 0.14 * streak)[..., None]

    # channel decorrelation (thin-oxide interference tint)
    sh = max(1, size // 170)
    base = np.stack([np.roll(base[:, :, 0], sh, axis=0),
                     base[:, :, 1],
                     np.roll(base[:, :, 2], -sh, axis=1)], axis=-1)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (0.5 * size)
    vig = 1.0 - 0.26 * np.clip(rr - 0.55, 0, 1) ** 1.6
    base *= vig[..., None]

    base += rng.normal(0, 0.012, base.shape).astype(np.float32)
    return np.clip(base, 0, 1).astype(np.float32)


# ===========================================================================
# Corrosion injection
# ===========================================================================
def _sample_lab(stats: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n LAB colours from stored GMM (fallback: palette jitter)."""
    try:
        means = np.asarray(stats["gmm_means"], dtype=np.float64)
        covs = np.asarray(stats["gmm_covariances"], dtype=np.float64)
        w = np.asarray(stats["gmm_weights"], dtype=np.float64)
        cov_type = str(np.asarray(stats["gmm_cov_type"]).item()) \
            if "gmm_cov_type" in stats else "full"
        w = np.clip(w, 1e-6, None); w = w / w.sum()
        comps = rng.choice(len(means), size=n, p=w)
        out = np.empty((n, 3), dtype=np.float64)
        for ci in np.unique(comps):
            m = comps == ci
            k = int(m.sum())
            if cov_type == "full" and covs.ndim == 3:
                L = np.linalg.cholesky(covs[ci] + 1e-6 * np.eye(3))
                out[m] = means[ci] + rng.standard_normal((k, 3)) @ L.T
            elif cov_type == "diag" and covs.ndim == 2:
                out[m] = means[ci] + rng.standard_normal((k, 3)) * np.sqrt(covs[ci])
            else:
                out[m] = means[ci] + rng.standard_normal((k, 3)) * 4.0
        return out
    except Exception:
        pal = np.asarray(stats["palette_rgb"], dtype=np.float64)
        lab = color.rgb2lab((pal / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
        idx = rng.integers(0, len(lab), n)
        return lab[idx] + rng.normal(0, 2.5, (n, 3))


def _spot_mask(size: int, cy: int, cx: int, radius: float,
               rng: np.random.Generator, irregularity: float = 0.35
               ) -> Tuple[np.ndarray, slice, slice]:
    """Organic corroded-pit alpha mask in a local window (fast, no full-canvas ops)."""
    pad = int(radius * (1 + irregularity)) + 3
    y0, y1 = max(0, cy - pad), min(size, cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(size, cx + pad + 1)
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0), np.float32), slice(0, 0), slice(0, 0)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dy = yy - cy; dx = xx - cx
    d = np.sqrt(dy * dy + dx * dx) + 1e-6
    ang = np.arctan2(dy, dx)
    harm = np.zeros_like(ang)
    for kk, aa in ((2, 0.5), (3, 0.32), (5, 0.18), (8, 0.08)):
        harm += aa * float(rng.normal()) * np.sin(kk * ang + float(rng.uniform(0, 2 * np.pi)))
    r_edge = radius * (1.0 + irregularity * harm)
    core = np.clip((r_edge - d) / max(1.5, radius * 0.35), 0, 1) ** 1.2
    return core.astype(np.float32), slice(y0, y1), slice(x0, x1)


def inject_corrosion(base_image: np.ndarray, palette: Optional[np.ndarray],
                     gmm: Optional[dict], corrosion_pct: float = 0.2,
                     spot_size_range: Tuple[int, int] = (5, 30),
                     seed: int = 0, mode: str = "spread"
                     ) -> Tuple[np.ndarray, dict]:
    """
    Alpha-blend corrosion onto a copper base image.

    mode='single' : one dominant coalesced colony (early pitting) plus orbiting
                    satellite speckles - mimics a single corrosion site.
    mode='spread' : many independent spots via Poisson-disk sampling.

    Returns (image, info) with realised coverage statistics.
    """
    img = base_image.copy()
    size = img.shape[0]
    rng = np.random.default_rng((seed + 1) % (2 ** 31))
    pct = float(np.clip(corrosion_pct, 0.0, 1.0))
    stats = gmm if gmm else {
        "palette_rgb": (palette if palette is not None
                        else FALLBACK_PALETTES["corroded"]["palette_rgb"]) * 255.0
    }

    target_area = pct * size * size
    lo, hi = spot_size_range
    achieved = 0.0
    n_placed = 0

    centres: List[Tuple[float, float, float]] = []
    if mode == "single":
        main_r = min(float(np.sqrt(0.55 * target_area / np.pi)) if target_area > 0 else 0.0,
                     size * 0.30)
        cy, cx = (float(rng.uniform(0.32, 0.68) * size),
                  float(rng.uniform(0.32, 0.68) * size))
        centres.append((cy, cx, main_r))
        want_extra = int(40 + 160 * pct)
        satellites_min = 12
        min_d = max(6.0, hi * 1.25)
        sat = poisson_disk(size, size, min_dist=min_d, rng=rng,
                           max_points=want_extra * 6)
        # sort by distance to the colony centre -> halo of coalescing pits
        d2 = (sat[:, 0] - cx) ** 2 + (sat[:, 1] - cy) ** 2
        order = np.argsort(d2)
        for idx in order[:want_extra]:
            px, py = float(sat[idx, 0]), float(sat[idx, 1])
            if d2[idx] <= (main_r * 3.0 + hi) ** 2 or len(centres) < satellites_min:
                r = float(rng.uniform(lo, max(lo + 1, hi)))
                # satellite probability decays with distance from the colony
                if d2[idx] <= (main_r * 3.0 + hi) ** 2 and rng.random() < 0.85:
                    centres.append((py, px, r))
        satellites_min = 0  # placeholder; loop above already bounded by want_extra
    else:
        mean_r = 0.5 * (lo + hi)
        want = max(1, int(target_area / (np.pi * mean_r * mean_r * 0.55)))
        want = min(want, 900)
        min_d = max(6.0, mean_r * 1.1)
        pts = poisson_disk(size, size, min_dist=min_d, rng=rng, max_points=want)
        for (px, py) in pts:
            centres.append((float(py), float(px), float(rng.uniform(lo, hi))))
        rng.shuffle(centres)

    labs = _sample_lab(stats, rng, max(2, len(centres)) * 2)
    rgbs = np.clip(color.lab2rgb(labs.reshape(-1, 1, 3)).reshape(-1, 3), 0, 1)

    # per-pixel coverage map -> exact realised-area control (no double counting)
    cov = np.zeros((size, size), dtype=np.float32)

    for si, (cy, cx, rad) in enumerate(centres):
        if achieved >= target_area or rad < 1.0:
            break
        mask, sy, sx = _spot_mask(size, int(cy), int(cx), rad, rng)
        if mask.size == 0:
            continue
        alpha = np.clip(mask * 1.15, 0, 1)                  # feathered blend
        new_px = np.maximum(0.0, alpha - cov[sy, sx])       # uncovered fraction
        gain = float(new_px.sum())
        if gain < 1.0:
            continue
        region = img[sy, sx]
        col = rgbs[si % len(rgbs)]
        col2 = rgbs[(si + 1) % len(rgbs)]
        # granular two-tone patina inside the pit
        tex = value_noise(max(8, int(4 * rad)), max(3.0, rad * 0.55), rng)
        if tex.shape != mask.shape:
            ty = np.linspace(0, tex.shape[0] - 1, mask.shape[0]).astype(int)
            tx = np.linspace(0, tex.shape[1] - 1, mask.shape[1]).astype(int)
            tex = tex[np.ix_(ty, tx)]
        tex = ndi.gaussian_filter(tex, sigma=0.6)
        spot_col = col[None, None, :] * (1 - tex[..., None]) + \
            col2[None, None, :] * tex[..., None]
        rim = np.clip(mask / (mask.max() + 1e-8), 0, 1)
        shade = 0.55 + 0.45 * rim ** 0.6                    # dark oxide edge
        spot_col *= shade[..., None]
        eff = np.clip(new_px / np.maximum(alpha, 1e-6), 0, 1) * alpha
        img[sy, sx] = region * (1 - eff[..., None]) + spot_col * eff[..., None]
        cov[sy, sx] = np.maximum(cov[sy, sx], alpha)
        achieved += gain
        n_placed += 1

    realised = float(np.clip(achieved / target_area, 0, 1)) if target_area > 0 else 0.0
    info = dict(corrosion_requested=pct, corrosion_realised=realised,
                n_spots=n_placed, mode=mode)
    return np.clip(img, 0, 1).astype(np.float32), info


# ===========================================================================
# High-level API
# ===========================================================================
_STATS_CACHE: Optional[Dict[str, dict]] = None


def _load_color_stats() -> Optional[Dict[str, dict]]:
    global _STATS_CACHE
    if _STATS_CACHE is None:
        try:
            from data_loader import load_stats, STATS_PATH
            _STATS_CACHE = load_stats(STATS_PATH)
        except Exception:
            _STATS_CACHE = {}
    return _STATS_CACHE or None


def _label_stats(label: str) -> dict:
    """{'palette_rgb':(K,3),'gmm':dict|None} from DB stats, fallback otherwise."""
    st = _load_color_stats()
    if st and label in st:
        d = st[label]
        gmm = {"gmm_means": d["gmm_means"], "gmm_covariances": d["gmm_covariances"],
               "gmm_weights": d["gmm_weights"], "palette_rgb": d["palette_rgb"]}
        if "gmm_cov_type" in d:
            gmm["gmm_cov_type"] = d["gmm_cov_type"]
        return {"palette_rgb": d["palette_rgb"], "gmm": gmm}
    fb = FALLBACK_PALETTES.get(label, FALLBACK_PALETTES["corroded"])
    return {"palette_rgb": fb["palette_rgb"], "gmm": None}


def generate_from_stats(label: str = "mixed", corrosion_pct: float = 30.0,
                        size: int = 512, seed: int = 0,
                        out_dir: str = OUT_DIR, save: bool = True,
                        variant: str = "auto") -> Tuple[np.ndarray, dict]:
    """
    Render one synthetic specimen.

    label         : 'healthy' | 'corroded' | 'mixed'
    corrosion_pct : 0..100 (% of surface covered by corrosion products)
    variant       : 'single' | 'spread' | 'auto' (corroded->single, mixed->spread)
    Saves PNG into outputs/synthetic/ when save=True. Returns (rgb float32, info).
    """
    label = str(label).lower().strip()
    assert label in ("healthy", "corroded", "mixed"), f"unknown label {label}"
    t_start = time.time()

    img = generate_copper_base(size=size, seed=seed)
    info = {"label": label, "corrosion_pct_input": corrosion_pct,
            "size": size, "seed": seed}

    if label != "healthy" and corrosion_pct > 0:
        stats = _label_stats(label)
        mode = variant if variant != "auto" else \
            ("single" if label == "corroded" else "spread")
        img, spot_info = inject_corrosion(
            img, stats["palette_rgb"], stats["gmm"],
            corrosion_pct=corrosion_pct / 100.0,
            spot_size_range=(max(4, size // 128), max(10, size // 14)),
            seed=seed + 7, mode=mode)
        info.update(spot_info)
    else:
        info.update(corrosion_requested=0.0, corrosion_realised=0.0, n_spots=0,
                    mode="none")

    # unsharp mask emulating crisp microscope optics
    blur = ndi.gaussian_filter(img, sigma=1.0)
    img = np.clip(img + 0.45 * (img - blur), 0, 1)

    info["render_seconds"] = round(time.time() - t_start, 3)
    if save:
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir,
                             f"{label}_{int(corrosion_pct)}pct_seed{seed}.png")
        Image.fromarray((img * 255).round().astype(np.uint8)).save(fname)
        info["path"] = fname
    return img, info


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Procedural copper specimen generator")
    ap.add_argument("--type", default="mixed",
                    choices=["healthy", "corroded", "mixed"])
    ap.add_argument("--corrosion", type=float, default=30.0, help="percent 0-100")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    for i in range(args.batch):
        img, info = generate_from_stats(args.type, args.corrosion, args.size,
                                        seed=args.seed + i, out_dir=args.out)
        print(f"[procedural] {info.get('path')}  "
              f"realised={info.get('corrosion_realised', 0) * 100:.1f}%  "
              f"spots={info.get('n_spots', 0)}  time={info['render_seconds']}s")
