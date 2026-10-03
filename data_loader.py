"""
CopperSpec-Synth :: data_loader.py
==================================
Streaming access to the 8-million-row copper color SQLite database plus
LAB-space color statistics (KMeans palettes + GaussianMixture fits) per label.

Hardware notes (24 GB RAM / 2 GB VRAM target):
    * NEVER materialises the full ``colors`` table. Rows are consumed in
      chunks of ``chunk_size`` (default 100k) via a server-side cursor and
      converted to float32 LAB arrays chunk-by-chunk.
    * KMeans is run with MiniBatchKMeans on a bounded reservoir sample so the
      memory footprint stays < ~50 MB regardless of table size.
    * GMM fitting uses EM warm-started from the KMeans centroids; if the
      sklearn API ever changes, we fall back to a diagonal-covariance fit.

Expected schema (auto-detected otherwise):
    colors(id INTEGER, r INT, g INT, b INT, label TEXT,
           specimen_id INT, corrosion_pct REAL)

Run standalone for a smoke test:
    python data_loader.py --db data/copper_colors.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from skimage import color
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(ROOT, "data", "copper_colors.db")
MODELS_DIR = os.path.join(ROOT, "models")
STATS_PATH = os.path.join(MODELS_DIR, "color_stats.npz")

CHUNK_SIZE = 100_000          # rows per streamed chunk (spec requirement)
N_CLUSTERS = 24               # palette size per label
N_GMM_COMPONENTS = 8          # mixture components per label
RESERVOIR_CAP = 250_000       # max samples fed to clusterers (memory bound)


# --------------------------------------------------------------------------
# Schema auto-detection
# --------------------------------------------------------------------------
def _first_table(conn: sqlite3.Connection) -> str:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "ORDER BY sql IS NULL, name LIMIT 1"
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Database contains no tables.")
    return row[0]


def _cols(conn: sqlite3.Connection, table: str) -> Dict[str, str]:
    """Return {lower-case column name: declared type} for *table*."""
    info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {r[1].lower(): (r[2] or "").upper() for r in info}


def _pick(cols: Dict[str, str], exact: List[str], like: List[str],
          types: Tuple[str, ...] = ()) -> Optional[str]:
    """Pick a column by exact name, then fuzzy substring, then type hint."""
    for name in exact:
        if name in cols:
            return name
    for pat in like:
        for c in cols:
            if pat in c:
                return c
    for c, t in cols.items():
        if types and any(t.startswith(tt) for tt in types):
            return c
    return None


class ColorDB:
    """Thin streaming wrapper around the copper color SQLite database."""

    def __init__(self, db_path: str = DEFAULT_DB):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                "Place copper_colors.db under data/ or pass --db."
            )
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        # modest page cache; keeps big scans fast without eating RAM
        self.conn.execute("PRAGMA cache_size=-16000")   # ~16 MB
        self.table = _first_table(self.conn)
        self.cols = _cols(self.conn, self.table)

        self.col_r = _pick(self.cols, ["r", "red"], ["red"], ("INT", "REAL"))
        self.col_g = _pick(self.cols, ["g", "green"], ["green"], ("INT", "REAL"))
        self.col_b = _pick(self.cols, ["b", "blue"], ["blue"], ("INT", "REAL"))
        self.col_label = _pick(self.cols, ["label", "class"], ["label", "clas"])
        self.col_corr = _pick(self.cols, ["corrosion_pct", "corrosion"],
                              ["corros", "pct"], ("REAL", "FLOA"))
        if not all([self.col_r, self.col_g, self.col_b]):
            raise RuntimeError(f"Could not find RGB columns in {self.table}: {list(self.cols)}")

        self._label_map: Optional[Dict[int, str]] = None  # id -> label (inferred mode)
        self._ensure_index()

    # -- housekeeping ------------------------------------------------------
    def _ensure_index(self) -> None:
        """CREATE INDEX IF NOT EXISTS on the label column (spec requirement)."""
        if self.col_label:
            try:
                self.conn.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_{self.table}_label '
                    f'ON "{self.table}" ("{self.col_label}")'
                )
                self.conn.commit()
            except sqlite3.OperationalError:
                pass  # e.g. expression index restrictions - non-fatal

    @property
    def has_labels(self) -> bool:
        return self.col_label is not None

    def labels(self) -> List[str]:
        if self.has_labels:
            cur = self.conn.execute(
                f'SELECT DISTINCT "{self.col_label}" FROM "{self.table}" '
                f'WHERE "{self.col_label}" IS NOT NULL')
            return [str(r[0]) for r in cur.fetchall()]
        return list(LABEL_ALIASES.keys())  # inferred-mode canonical labels

    def close(self):
        self.conn.close()

    # -- streaming ---------------------------------------------------------
    def yield_chunk(self, label: Optional[str] = None,
                    chunk_size: int = CHUNK_SIZE,
                    include_corrosion: bool = False
                    ) -> Iterator[np.ndarray]:
        """
        Yield ``(N,3)`` uint8 RGB arrays (optionally ``(N,4)`` with a trailing
        corrosion_pct column) for rows matching *label*. ``label=None`` streams
        the whole table. Memory usage is O(chunk_size), never O(table).
        """
        sel = [f'"{self.col_r}"', f'"{self.col_g}"', f'"{self.col_b}"']
        if include_corrosion:
            sel.append(f'"{self.col_corr}"' if self.col_corr else "0.0")
        sql = f'SELECT {", ".join(sel)} FROM "{self.table}"'
        params: tuple = ()
        if label is not None and self.has_labels:
            sql += f' WHERE "{self.col_label}" = ?'
            params = (label,)
        cur = self.conn.execute(sql, params)
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            arr = np.asarray(rows, dtype=np.float32 if include_corrosion else np.uint8)
            if include_corrosion:
                arr[:, :3] = np.clip(arr[:, :3], 0, 255).astype(np.uint8)
            yield arr
        cur.close()

    # -- label inference (no `label` column) --------------------------------
    def infer_labels(self, chunk_size: int = CHUNK_SIZE) -> Dict[int, str]:
        """
        Cluster the colour space into 3 groups and map each group to
        healthy / corroded / mixed using copper-referenced heuristics:

          * hue angle in ab-plane: oxidised copper shifts toward green/blue
            (positive hb, negative a relative to metallic copper);
          * lightness L: patina (malachite/azurite) is darker than bright metal;
          * spread within a specimen => 'mixed'.

        Assignment is done per specimen_id when available, else per row-id
        block. Result cached in ``self._label_map``.
        """
        if self._label_map is not None:
            return self._label_map

        key_col = _pick(self.cols, ["specimen_id", "id"], ["specimen", "id"],
                        ("INT",))
        lab_chunks, key_chunks = [], []
        cur = self.conn.execute(
            f'SELECT "{key_col}", "{self.col_r}", "{self.col_g}", "{self.col_b}" '
            f'FROM "{self.table}"')
        n_seen, cap = 0, RESERVOIR_CAP
        rng = np.random.default_rng(0)
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            a = np.asarray(rows, dtype=np.float32)
            # reservoir-ish sampling to bound memory
            take = min(len(a), max(1, cap - n_seen))
            if take > 0:
                idx = rng.choice(len(a), take, replace=False) if take < len(a) \
                    else np.arange(take)
                lab_chunks.append(color.rgb2lab((a[idx, 1:4] / 255.0)))
                key_chunks.append(a[idx, 0].astype(np.int64))
            n_seen += len(a)
        cur.close()
        X = np.concatenate(lab_chunks).reshape(-1, 3)
        keys = np.concatenate(key_chunks)

        km = MiniBatchKMeans(n_clusters=3, random_state=0, n_init=3,
                             batch_size=10_000).fit(X)
        centers = km.cluster_centers_           # (3,3) LAB
        # rank clusters: most "metallic" (brightest, reddest a>) = healthy,
        # most shifted (lowest a, highest b-green or blue) = corroded,
        # middle = mixed.
        score = centers[:, 0] * 0.4 + centers[:, 1] * 1.0 - np.abs(centers[:, 2]) * 0.2
        order = np.argsort(score)               # ascending corrosion-ness? see below
        # higher score (bright, positive a) -> healthier
        lab_of_cluster = {}
        names = ["corroded", "mixed", "healthy"]  # order ascending health
        for rank, cid in enumerate(order):
            lab_of_cluster[int(cid)] = names[rank]

        # majority vote per specimen key
        point_labels = np.array([lab_of_cluster[c] for c in km.labels_])
        from collections import Counter, defaultdict
        votes = defaultdict(Counter)
        for k, pl in zip(keys.tolist(), point_labels.tolist()):
            votes[k][pl] += 1
        label_map = {k: v.most_common(1)[0][0] for k, v in votes.items()}
        self._label_map = label_map
        return label_map

    def _label_where(self, label: str) -> Tuple[str, tuple]:
        """SQL fragment selecting rows of *label*, using real column or cache."""
        if self.has_labels:
            return f'WHERE "{self.col_label}" = ?', (label,)
        ids = [k for k, v in self.infer_labels().items() if v == label]
        if not ids:
            return "WHERE 1=0", ()
        id_col = _pick(self.cols, ["specimen_id", "id"], ["specimen", "id"], ("INT",))
        placeholders = ",".join("?" * len(ids))
        return f'WHERE "{id_col}" IN ({placeholders})', tuple(ids)

    def iter_lab(self, label: Optional[str], chunk_size: int = CHUNK_SIZE
                 ) -> Iterator[np.ndarray]:
        """Stream LAB arrays shaped (N,3) float64 for *label* (None = all)."""
        if label is None:
            where_sql, params = "", ()
        elif self.has_labels:
            where_sql, params = f'WHERE "{self.col_label}" = ?', (label,)
        else:
            where_sql, params = self._label_where(label)
        cur = self.conn.execute(
            f'SELECT "{self.col_r}", "{self.col_g}", "{self.col_b}" '
            f'FROM "{self.table}" {where_sql}', params)
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            rgb = np.asarray(rows, dtype=np.float64) / 255.0
            yield color.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        cur.close()


# Canonical label spellings used across the project.
LABEL_ALIASES = {
    "healthy": ("healthy", "clean", "pristine", "0"),
    "corroded": ("corroded", "single_corroded", "corrosion", "1"),
    "mixed": ("mixed", "partial", "2"),
}


def canonical_label(name: str) -> str:
    n = str(name).strip().lower()
    for canon, al in LABEL_ALIASES.items():
        if n in al:
            return canon
    return n


# --------------------------------------------------------------------------
# Colour statistics: palettes + GMMs in LAB space
# --------------------------------------------------------------------------
def analyze_colors(db_path: str = DEFAULT_DB,
                   out_path: str = STATS_PATH,
                   chunk_size: int = CHUNK_SIZE,
                   verbose: bool = True) -> Dict[str, dict]:
    """
    For every label present in the DB:
      * stream its colours in chunks, converting RGB->LAB on the fly,
      * build a 24-colour palette with MiniBatchKMeans (bounded memory),
      * fit an 8-component GaussianMixture (full covariance) in LAB,
      * store everything (plus mean/std/mixing weights) in models/color_stats.npz.

    Returns the stats dict; also written to disk.
    """
    db = ColorDB(db_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    labels = db.labels()
    if verbose:
        print(f"[data_loader] table='{db.table}' labels={labels} "
              f"labelled={'yes' if db.has_labels else 'no (will infer)'}")

    result: Dict[str, dict] = {}
    for raw_label in labels:
        lab_name = canonical_label(raw_label)
        # ---- stream into a bounded float32 reservoir ---------------------
        reservoir = np.empty((RESERVOIR_CAP, 3), dtype=np.float32)
        n_seen = 0
        rng = np.random.default_rng(hash(lab_name) % (2**31))
        for chunk in db.iter_lab(raw_label, chunk_size):
            for row in range(len(chunk)):
                if n_seen < RESERVOIR_CAP:
                    reservoir[n_seen] = chunk[row]
                else:  # classic reservoir sampling keeps uniform coverage
                    j = rng.integers(0, n_seen + 1)
                    if j < RESERVOIR_CAP:
                        reservoir[j] = chunk[row]
                n_seen += 1
            # chunk-local vectorised path is faster for the first fill
            if n_seen <= RESERVOIR_CAP and len(chunk) <= RESERVOIR_CAP - (n_seen - len(chunk)):
                pass  # already handled above; kept simple & correct
        if n_seen == 0:
            continue
        X = reservoir[:min(n_seen, RESERVOIR_CAP)].astype(np.float64)
        if verbose:
            print(f"  [{lab_name}] saw {n_seen:,} rows, fitting on {len(X):,}")

        # ---- palette: MiniBatchKMeans (streaming-friendly) ---------------
        k = min(N_CLUSTERS, max(1, len(X)))
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=2,
                             batch_size=min(10_000, len(X)))
        km.fit(X)
        palette_lab = km.cluster_centers_                       # (k,3)
        counts = np.bincount(km.labels_, minlength=k).astype(np.float64)
        palette_rgb = (np.clip(color.lab2rgb(palette_lab.reshape(-1, 1, 3))
                               .reshape(-1, 3), 0, 1) * 255).round()

        # ---- GMM: 8 components, warm start from palette subset -----------
        n_comp = min(N_GMM_COMPONENTS, max(1, len(X)))
        gm = GaussianMixture(n_components=n_comp, covariance_type="full",
                             random_state=0, reg_covar=1e-4, max_iter=120,
                             init_params="k-means++")
        gm.means_init = palette_lab[np.linspace(0, len(palette_lab) - 1,
                                                n_comp).astype(int)]
        gm.weights_init = (counts / counts.sum())[
            np.linspace(0, len(counts) - 1, n_comp).astype(int)]
        gm.covariances_init = np.tile(np.cov(X.T) + 1e-3 * np.eye(3),
                                      (n_comp, 1, 1))
        gm.init_params = "random_from_data" if False else gm.init_params
        try:
            gm.set_params(warm_start=False)
            gm = GaussianMixture(
                n_components=n_comp, covariance_type="full", random_state=0,
                reg_covar=1e-4, max_iter=120, means_init=gm.means_init,
                weights_init=gm.weights_init, covariances_init=gm.covariances_init)
            gm.fit(X)
        except Exception:
            gm = GaussianMixture(n_components=n_comp, covariance_type="diag",
                                 random_state=0, reg_covar=1e-3).fit(X)

        result[lab_name] = dict(
            palette_lab=palette_lab,                     # (24,3)
            palette_rgb=palette_rgb,                     # (24,3) uint8-ish
            palette_weights=counts / counts.sum(),       # (24,)
            gmm_means=gm.means_,                         # (8,3)
            gmm_covariances=gm.covariances_,             # (8,3,3) or (8,3)
            gmm_weights=gm.weights_,                     # (8,)
            gmm_cov_type=np.array(gm.covariance_type),   # string scalar
            lab_mean=X.mean(axis=0),
            lab_std=X.std(axis=0),
            n_rows=np.array(n_seen),
        )

    db.close()
    if not result:
        raise RuntimeError("No colour data found - is the database populated?")

    # ---- persist: one array per key "label__field" ------------------------
    flat: Dict[str, np.ndarray] = {"labels": np.array(sorted(result))}
    for lab, d in result.items():
        for field, val in d.items():
            flat[f"{lab}__{field}"] = np.asarray(val)
    np.savez_compressed(out_path, **flat)
    if verbose:
        size_kb = os.path.getsize(out_path) / 1024
        print(f"[data_loader] wrote {out_path} ({size_kb:.1f} KB)")
    return result


def load_stats(path: str = STATS_PATH) -> Dict[str, dict]:
    """Load models/color_stats.npz back into {label: {field: ndarray}}."""
    z = np.load(path, allow_pickle=False)
    labels = [str(x) for x in z["labels"]]
    out: Dict[str, dict] = {}
    for lab in labels:
        d = {}
        pre = f"{lab}__"
        for k in z.files:
            if k.startswith(pre):
                d[k[len(pre):]] = z[k]
        out[lab] = d
    return out


def get_label_stats(label: str, path: str = STATS_PATH) -> dict:
    stats = load_stats(path)
    lab = canonical_label(label)
    if lab not in stats:
        # nearest available fallback keeps the UI alive on partial data
        lab = sorted(stats)[0]
    return stats[lab]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CopperSpec-Synth data loader")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=STATS_PATH)
    ap.add_argument("--chunk", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    db = ColorDB(args.db)
    total = db.conn.execute(f'SELECT COUNT(*) FROM "{db.table}"').fetchone()[0]
    print(f"[data_loader] {args.db}\n  table={db.table}  rows={total:,}")
    # demonstrate streaming (does not accumulate)
    shown = 0
    for chunk in db.yield_chunk(chunk_size=args.chunk):
        shown += len(chunk)
        if shown >= 3 * args.chunk:
            break
    print(f"  streaming OK (sampled {shown:,} rows in {max(1, shown // args.chunk)} chunks)")
    db.close()
    analyze_colors(args.db, args.out, args.chunk)
