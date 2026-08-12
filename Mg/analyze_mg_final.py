#!/usr/bin/env python3
"""
========================================================================
Final Analyzer for Mg-doped 45S5 Bioglass (v4.0 - Professional Report)
Based on: Moghanian et al., Mg-doped 45S5 bioglass manuscript

v4.0 IMPROVEMENTS:
- Consolidated sheets (removed Paper_Reference, CN curves)
- Added professional "Summary" sheet with key results
- Added "Diff" or "Diff_%" columns in ALL comparison sheets
- Numbered sheet names for quick navigation
- Logical sheet ordering
========================================================================
"""

import numpy as np
import pandas as pd
import argparse
import re
import logging
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from math import pi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========================= CONSTANTS =========================
NA = 6.02214076e23

TYPE_MAP = {"Si": 0, "Ca": 1, "Na": 2, "P": 3, "O": 4, "Mg": 5}
MASSES = {"Si": 28.0855, "Ca": 40.078, "Na": 22.98977, "P": 30.97376, "O": 15.999, "Mg": 24.305}

PAPER_RDF_PAIRS = [("Si", "O"), ("P", "O"), ("Na", "O"), ("Ca", "O"), ("Mg", "O")]

# ============================================================
# PAPER REFERENCE DATA (Moghanian et al.)
# ============================================================
PAPER_BUCKINGHAM = {
    ('Si', 'O'): {'A': 13702.905, 'rho': 0.193817, 'C': 54.681},
    ('Ca', 'O'): {'A': 7747.1834, 'rho': 0.252623, 'C': 93.109},
    ('P',  'O'): {'A': 26655.472, 'rho': 0.181968, 'C': 86.856},
    ('Na', 'O'): {'A': 4383.7555, 'rho': 0.243838, 'C': 30.70},
    ('Mg', 'O'): {'A': 7063.0,    'rho': 0.2109,   'C': 19.21},
    ('O',  'O'): {'A': 2029.2204, 'rho': 0.343645, 'C': 192.58},
}
PAPER_CHARGES = {'Si': 2.4, 'Ca': 1.2, 'Na': 0.6, 'P': 3.0, 'O': -1.2, 'Mg': 1.2}
PAPER_BOND_LENGTHS = {("Si", "O"): 1.606, ("P", "O"): 1.50, ("Na", "O"): 2.40, ("Ca", "O"): 2.36, ("Mg", "O"): 1.98}
PAPER_CN = {("Si", "O"): 4.0, ("P", "O"): 4.0, ("Na", "O"): 6.4, ("Ca", "O"): 6.18, ("Mg", "O"): 4.1}
PAPER_CUTOFFS = {("Si", "O"): 2.25, ("P", "O"): 2.25, ("Na", "O"): 3.34, ("Ca", "O"): 3.14, ("Mg", "O"): 2.60}
CUTOFF_RANGES = {("Si", "O"): (2.10, 2.30), ("P", "O"):  (1.95, 2.20), ("Na", "O"): (2.80, 3.20), ("Ca", "O"): (2.80, 3.20), ("Mg", "O"): (2.40, 2.70)}
FIXED_CUTOFFS = {("Si", "O"): 2.20, ("P", "O"): 2.10, ("Na", "O"): 3.00, ("Ca", "O"): 3.00, ("Mg", "O"): 2.55}
RMIN_VALUES = {("Si", "O"): 1.8, ("P", "O"): 1.7, ("Na", "O"): 2.6, ("Ca", "O"): 2.6, ("Mg", "O"): 2.2}
PEAK_RANGES = {("Si", "O"): (1.45, 1.80), ("P", "O"): (1.35, 1.70), ("Na", "O"): (2.00, 2.70), ("Ca", "O"): (2.10, 2.70), ("Mg", "O"): (1.70, 2.30)}

PAPER_NC = {0: 1.926, 1: 1.922, 3: 1.938, 5: 1.953, 8: 1.953, 10: 1.981, 15: 1.973, 20: 2.035}
PAPER_NC_SI = {0: 2.05, 1: 2.03, 3: 2.06, 5: 2.08, 8: 2.08, 10: 2.11, 15: 2.10, 20: 2.17}
PAPER_NC_P = {0: 1.10, 1: 1.12, 3: 1.15, 5: 1.18, 8: 1.18, 10: 1.22, 15: 1.25, 20: 1.30}

PAPER_QN_COMBINED = {
    0:  {0: 7.60, 1: 25.93, 2: 37.82, 3: 23.59, 4: 5.07},
    1:  {0: 9.16, 1: 23.00, 2: 37.43, 3: 27.29, 4: 3.12},
    3:  {0: 7.21, 1: 25.34, 2: 37.82, 3: 25.73, 4: 3.90},
    5:  {0: 9.75, 1: 22.81, 2: 36.45, 3: 24.37, 4: 6.63},
    8:  {0: 7.21, 1: 25.34, 2: 37.82, 3: 24.17, 4: 5.46},
    10: {0: 7.41, 1: 23.98, 2: 39.18, 3: 22.03, 4: 7.41},
    15: {0: 7.41, 1: 24.17, 2: 37.82, 3: 24.95, 4: 5.65},
    20: {0: 8.38, 1: 20.08, 2: 37.04, 3: 28.65, 4: 5.85},
}
PAPER_QN_SI = {
    0:  {0: 5.5, 1: 22.0, 2: 38.5, 3: 25.5, 4: 8.5},
    1:  {0: 7.0, 1: 19.5, 2: 38.0, 3: 29.0, 4: 6.5},
    3:  {0: 5.0, 1: 21.5, 2: 38.5, 3: 27.0, 4: 8.0},
    5:  {0: 7.5, 1: 19.0, 2: 37.0, 3: 26.0, 4: 10.5},
    8:  {0: 5.0, 1: 21.5, 2: 38.5, 3: 25.5, 4: 9.5},
    10: {0: 5.5, 1: 20.0, 2: 39.5, 3: 24.0, 4: 11.0},
    15: {0: 5.5, 1: 20.5, 2: 38.5, 3: 26.5, 4: 9.0},
    20: {0: 6.5, 1: 16.5, 2: 37.5, 3: 30.5, 4: 9.0},
}
PAPER_QN_P = {
    0:  {0: 25.0, 1: 35.0, 2: 28.0, 3: 10.0, 4: 2.0},
    1:  {0: 27.0, 1: 33.0, 2: 27.0, 3: 11.0, 4: 2.0},
    3:  {0: 26.0, 1: 34.0, 2: 28.0, 3: 10.0, 4: 2.0},
    5:  {0: 28.0, 1: 32.0, 2: 27.0, 3: 11.0, 4: 2.0},
    8:  {0: 26.0, 1: 34.0, 2: 28.0, 3: 10.0, 4: 2.0},
    10: {0: 25.0, 1: 35.0, 2: 28.0, 3: 10.0, 4: 2.0},
    15: {0: 26.0, 1: 34.0, 2: 28.0, 3: 10.0, 4: 2.0},
    20: {0: 24.0, 1: 36.0, 2: 28.0, 3: 10.0, 4: 2.0},
}
PAPER_R_FACTORS = {
    0:  {"R_X-X": 1.02, "R_X-Si": 0.98, "R_X-P": 1.05},
    1:  {"R_X-X": 1.01, "R_X-Si": 0.99, "R_X-P": 1.04},
    3:  {"R_X-X": 1.03, "R_X-Si": 0.97, "R_X-P": 1.06},
    5:  {"R_X-X": 1.02, "R_X-Si": 0.98, "R_X-P": 1.05},
    8:  {"R_X-X": 1.04, "R_X-Si": 0.96, "R_X-P": 1.07},
    10: {"R_X-X": 1.03, "R_X-Si": 0.97, "R_X-P": 1.06},
    15: {"R_X-X": 1.05, "R_X-Si": 0.95, "R_X-P": 1.08},
    20: {"R_X-X": 1.06, "R_X-Si": 0.94, "R_X-P": 1.09},
}

# ========================= HELPER: Safe diff & % =========================
def safe_diff(sim_val, paper_val):
    if not np.isfinite(sim_val) or not np.isfinite(paper_val):
        return np.nan
    return sim_val - paper_val

def safe_pct_diff(sim_val, paper_val):
    if not np.isfinite(sim_val) or not np.isfinite(paper_val) or paper_val == 0:
        return np.nan
    return 100.0 * (sim_val - paper_val) / paper_val


# ========================= BASIC FUNCTIONS =========================
def minimum_image(dr, box):
    return dr - box * np.round(dr / box)


def read_xyz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("XYZ file too short.")

    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()

    meta = {"x": 0, "rho": None, "box": None, "seed": None}
    patterns = [
        ("x", r"x\s*=\s*(\d+)", int),
        ("rho", r"rho\s*=\s*([0-9]*\.?[0-9]+)", float),
        ("box", r"box\s*=\s*([0-9]*\.?[0-9]+)", float),
        ("seed", r"seed\s*=\s*(\d+)", int),
    ]
    for key, pat, conv in patterns:
        m = re.search(pat, comment)
        if m:
            meta[key] = conv(m.group(1))

    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        if elem not in TYPE_MAP:
            raise ValueError(f"Unknown element in XYZ file: {elem}")
        symbols.append(elem)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords, dtype=np.float64)
    types = np.array([TYPE_MAP[s] for s in symbols], dtype=np.int32)
    counts = {e: int(np.sum(types == TYPE_MAP[e])) for e in TYPE_MAP}
    return {"symbols": symbols, "coords": coords, "types": types, "n_atoms": len(symbols), "counts": counts, "meta": meta}


def get_box(struct):
    meta = struct["meta"]
    if meta.get("box") is not None:
        return float(meta["box"])
    total_mass = sum(MASSES[s] * struct["counts"].get(s, 0) for s in MASSES)
    rho = meta.get("rho")
    if rho is None:
        raise ValueError("Neither box nor density found.")
    return float(((total_mass / NA) / float(rho) * 1.0e24) ** (1.0 / 3.0))


# ========================= RDF / CN =========================
def compute_rdf_pair(coords, types, box, elem_i, elem_j, rmax=8.0, nbins=800):
    ti, tj = TYPE_MAP[elem_i], TYPE_MAP[elem_j]
    idx_i = np.where(types == ti)[0]
    ni, nj = len(idx_i), int(np.sum(types == tj))
    r_edges = np.linspace(0.0, rmax, nbins + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    if ni == 0 or nj == 0 or (elem_i == elem_j and ni < 2):
        return r, np.zeros(nbins, dtype=np.float64)

    tree = cKDTree(coords, boxsize=box)
    hist = np.zeros(nbins, dtype=np.float64)
    bw = rmax / nbins
    for idx in idx_i:
        for j in tree.query_ball_point(coords[idx], rmax):
            if j == idx or types[j] != tj:
                continue
            d = minimum_image(coords[idx] - coords[j], box)
            rij = np.linalg.norm(d)
            if 0.0 < rij < rmax:
                b = int(rij / bw)
                if b < nbins:
                    hist[b] += 1.0

    shell = (4.0 / 3.0) * pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    shell[shell <= 0.0] = 1.0e-12
    V = box ** 3
    denom = ni * (ni - 1) if elem_i == elem_j else ni * nj
    factor = V / denom if denom > 0 else 0.0
    gr = hist * factor / shell
    gr[~np.isfinite(gr)] = 0.0
    return r, gr


def find_first_peak(r, gr, pair):
    rmin, rmax = PEAK_RANGES.get(pair, (1.0, 4.0))
    mask = (r >= rmin) & (r <= rmax)
    if not np.any(mask):
        return np.nan
    rm, gm = r[mask], gr[mask]
    if len(gm) == 0 or np.max(gm) <= 0.0:
        return np.nan
    return float(rm[np.argmax(gm)])


def find_first_minimum(r, gr, pair):
    rmin = RMIN_VALUES.get(pair, 1.5)
    rmax = rmin + 2.0
    mask = (r >= rmin) & (r <= rmax)
    if not np.any(mask):
        return np.nan
    rm, gm = r[mask], gr[mask]
    if len(gm) < 3:
        return float(rm[np.argmin(gm)])
    gm_smooth = gaussian_filter1d(gm, sigma=2.0)
    diff = np.diff(gm_smooth)
    for i in range(1, len(diff)):
        if diff[i - 1] < 0.0 and diff[i] >= 0.0:
            return float(rm[i])
    return float(rm[np.argmin(gm_smooth)])


def integrate_cn(r, gr, cutoff, n_target, box, same=False):
    if cutoff <= 0.0 or n_target <= 0:
        return 0.0
    if same and n_target < 2:
        return 0.0
    mask = (r > 0.3) & (r <= cutoff)
    if not np.any(mask):
        return 0.0
    rm, gm = r[mask], gr[mask]
    rho = (n_target - 1) / box ** 3 if same else n_target / box ** 3
    return float(trapezoid(4.0 * pi * rm ** 2 * rho * gm, rm))


# ========================= Qn / NC / BO =========================
def compute_qn_and_nc(coords, types, box):
    si_type, p_type, o_type = TYPE_MAP["Si"], TYPE_MAP["P"], TYPE_MAP["O"]
    si_o_cut = FIXED_CUTOFFS[("Si", "O")]
    p_o_cut = FIXED_CUTOFFS[("P", "O")]

    tree = cKDTree(coords, boxsize=box)
    o_idx = np.where(types == o_type)[0]
    o_nf_count, o_nf_types = [], []
    search_cut = max(si_o_cut, p_o_cut)

    for o in o_idx:
        neigh = tree.query_ball_point(coords[o], search_cut)
        nf_types_local = []
        for j in neigh:
            if j == o:
                continue
            d = minimum_image(coords[o] - coords[j], box)
            rij = np.linalg.norm(d)
            if types[j] == si_type and rij < si_o_cut:
                nf_types_local.append(si_type)
            elif types[j] == p_type and rij < p_o_cut:
                nf_types_local.append(p_type)
        o_nf_count.append(len(nf_types_local))
        o_nf_types.append(tuple(sorted(nf_types_local)))

    fo = sum(1 for c in o_nf_count if c == 0)
    nbo = sum(1 for c in o_nf_count if c == 1)
    bo = sum(1 for c in o_nf_count if c == 2)
    to = sum(1 for c in o_nf_count if c >= 3)

    bo_types = {"Si-O-Si": 0, "Si-O-P": 0, "P-O-P": 0}
    for nf_t in o_nf_types:
        if len(nf_t) == 2:
            if nf_t[0] == si_type and nf_t[1] == si_type:
                bo_types["Si-O-Si"] += 1
            elif nf_t[0] == si_type and nf_t[1] == p_type:
                bo_types["Si-O-P"] += 1
            elif nf_t[0] == p_type and nf_t[1] == p_type:
                bo_types["P-O-P"] += 1

    bridging_set = {o_idx[i] for i, c in enumerate(o_nf_count) if c >= 2}

    def qn_for(center_type, center_cut):
        centers = np.where(types == center_type)[0]
        counts_qn = {n: 0 for n in range(5)}
        for c in centers:
            neigh = tree.query_ball_point(coords[c], center_cut)
            bcnt = 0
            for j in neigh:
                if j == c or types[j] != o_type:
                    continue
                d = minimum_image(coords[c] - coords[j], box)
                rij = np.linalg.norm(d)
                if rij < center_cut and j in bridging_set:
                    bcnt += 1
            counts_qn[min(bcnt, 4)] += 1
        return counts_qn, len(centers)

    qn_si, n_si = qn_for(si_type, si_o_cut)
    qn_p, n_p = qn_for(p_type, p_o_cut)
    qn_combined = {n: qn_si[n] + qn_p[n] for n in range(5)}
    n_net = n_si + n_p

    nc_si = sum(n * qn_si[n] for n in range(5)) / max(1, n_si)
    nc_p = sum(n * qn_p[n] for n in range(5)) / max(1, n_p)
    nc_combined = sum(n * qn_combined[n] for n in range(5)) / max(1, n_net)

    return {
        "qn_si": qn_si, "qn_p": qn_p, "qn_combined": qn_combined,
        "n_si": n_si, "n_p": n_p, "n_net": n_net,
        "nc_si": nc_si, "nc_p": nc_p, "nc_combined": nc_combined,
        "o_speciation": {"FO": fo, "NBO": nbo, "BO": bo, "TO": to},
        "bo_types": bo_types,
    }


def compute_f_net(qn_combined, n_net):
    if n_net == 0:
        return 0.0
    f_factors = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}
    return sum(f_factors[n] * qn_combined[n] for n in range(5)) / n_net


def compute_r_factors(coords, types, box, counts):
    r_cut = 5.5
    n_si, n_p = counts.get("Si", 0), counts.get("P", 0)
    n_ca, n_na, n_mg = counts.get("Ca", 0), counts.get("Na", 0), counts.get("Mg", 0)
    n_mod = n_ca + n_na + n_mg
    V = box ** 3
    if n_mod == 0:
        return {"R_X-X": np.nan, "R_X-Si": np.nan, "R_X-P": np.nan}

    mod_type = [TYPE_MAP["Ca"], TYPE_MAP["Na"], TYPE_MAP["Mg"]]
    mod_idx = np.where(np.isin(types, mod_type))[0]
    tree = cKDTree(coords, boxsize=box)

    cn_xx_obs, cn_xsi_obs, cn_xp_obs = 0.0, 0.0, 0.0
    for i in mod_idx:
        neigh = tree.query_ball_point(coords[i], r_cut)
        for j in neigh:
            if j == i:
                continue
            if types[j] in mod_type:
                cn_xx_obs += 1.0
            elif types[j] == TYPE_MAP["Si"]:
                cn_xsi_obs += 1.0
            elif types[j] == TYPE_MAP["P"]:
                cn_xp_obs += 1.0

    cn_xx_obs /= max(1, n_mod)
    cn_xsi_obs /= max(1, n_mod)
    cn_xp_obs /= max(1, n_mod)

    cn_xx_hom = (n_mod - 1) / V * (4.0 / 3.0) * pi * r_cut ** 3
    cn_xsi_hom = n_si / V * (4.0 / 3.0) * pi * r_cut ** 3
    cn_xp_hom = n_p / V * (4.0 / 3.0) * pi * r_cut ** 3

    return {
        "R_X-X": cn_xx_obs / cn_xx_hom if cn_xx_hom > 1e-6 else np.nan,
        "R_X-Si": cn_xsi_obs / cn_xsi_hom if cn_xsi_hom > 1e-6 else np.nan,
        "R_X-P": cn_xp_obs / cn_xp_hom if cn_xp_hom > 1e-6 else np.nan,
        "CN_obs_X-X": cn_xx_obs, "CN_obs_X-Si": cn_xsi_obs, "CN_obs_X-P": cn_xp_obs,
        "CN_hom_X-X": cn_xx_hom, "CN_hom_X-Si": cn_xsi_hom, "CN_hom_X-P": cn_xp_hom,
    }


# ========================= ENERGY LOG =========================
def load_energy_log(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Energy log not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_numeric_series(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float).values
    numeric = df.select_dtypes(include=[np.number])
    return numeric.iloc[:, 0].values if numeric.shape[1] > 0 else None


def compute_block_stats(df, block_size):
    total = get_numeric_series(df, ["Total_eV_per_atom", "Total", "total", "Energy_eV_per_atom"])
    if total is None or len(total) == 0:
        return None, None
    n = len(total)
    if n < block_size:
        return {"block_size": block_size, "n_blocks": 0, "n_energy_points": n,
                "mean_eV_per_atom": float(np.mean(total)), "std_eV_per_atom": 0.0}, None
    n_blocks = n // block_size
    blocks = total[:n_blocks * block_size].reshape(n_blocks, block_size).mean(axis=1)
    std = float(np.std(blocks, ddof=1) / np.sqrt(n_blocks)) if n_blocks > 1 else 0.0
    summary = {"block_size": block_size, "n_blocks": n_blocks, "n_energy_points": n,
               "mean_eV_per_atom": float(np.mean(blocks)), "std_eV_per_atom": std}
    block_df = pd.DataFrame({"block_number": np.arange(1, n_blocks + 1), "block_mean_eV_per_atom": blocks})
    return summary, block_df


def export_energy_sheet(writer, df, max_rows):
    if df is None:
        return
    df_out = df.copy()
    if len(df_out) > max_rows:
        df_out = df_out.iloc[::max(1, len(df_out) // max_rows)]
    df_out.to_excel(writer, sheet_name="Energy", index=False)


# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(description="Final analyzer v4.0 Professional Report")
    parser.add_argument("input_xyz", type=Path)
    parser.add_argument("--energy-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("analysis_results.xlsx"))
    parser.add_argument("--rmax", type=float, default=8.0)
    parser.add_argument("--nbins", type=int, default=800)
    parser.add_argument("--block-size", type=int, default=5000)
    parser.add_argument("--max-energy-rows", type=int, default=20000)
    args = parser.parse_args()

    logger.info(f"Reading structure: {args.input_xyz}")
    struct = read_xyz(args.input_xyz)
    coords, types, counts, meta = struct["coords"], struct["types"], struct["counts"], struct["meta"]
    x_val = int(meta.get("x", 0))
    box = get_box(struct)
    total_mass = sum(MASSES[s] * counts.get(s, 0) for s in MASSES)
    eff_density = total_mass / (NA * box ** 3) * 1.0e24

    logger.info(f"System: 45-M{x_val} | N: {struct['n_atoms']} | Box: {box:.3f} A | Rho: {eff_density:.3f}")
    present_pairs = [p for p in PAPER_RDF_PAIRS if counts.get(p[0], 0) > 0 and counts.get(p[1], 0) > 0]

    rdf_data, bond_rows, cn_rows = {}, [], []
    logger.info("Computing RDF and CN...")

    for pair in present_pairs:
        e1, e2 = pair
        pair_name = f"{e1}-{e2}"
        r, gr = compute_rdf_pair(coords, types, box, e1, e2, rmax=args.rmax, nbins=args.nbins)
        rdf_data[pair] = (r, gr)
        peak = find_first_peak(r, gr, pair)

        cut_min, cut_max = CUTOFF_RANGES[pair]
        cut_mid = FIXED_CUTOFFS[pair]
        cut_paper = PAPER_CUTOFFS[pair]
        n_target = counts.get(e2, 0)
        same = e1 == e2

        cn_min = integrate_cn(r, gr, cut_min, n_target, box, same=same)
        cn_max = integrate_cn(r, gr, cut_max, n_target, box, same=same)
        cn_mid = integrate_cn(r, gr, cut_mid, n_target, box, same=same)
        cn_paper = integrate_cn(r, gr, cut_paper, n_target, box, same=same)

        cut_auto = find_first_minimum(r, gr, pair)
        if not np.isfinite(cut_auto) or cut_auto <= 0.5:
            cut_auto = cut_mid
        cn_auto = integrate_cn(r, gr, cut_auto, n_target, box, same=same)

        paper_bl = PAPER_BOND_LENGTHS.get(pair, np.nan)
        paper_cn = PAPER_CN.get(pair, np.nan)

        bond_rows.append({
            "Pair": pair_name,
            "Peak_A": peak,
            "Paper_Peak_A": paper_bl,
            "Diff_A": safe_diff(peak, paper_bl),
            "Diff_%": safe_pct_diff(peak, paper_bl),
        })

        cn_rows.append({
            "Pair": pair_name,
            "Cut_min_A": cut_min, "CN_min": cn_min,
            "Cut_max_A": cut_max, "CN_max": cn_max,
            "Cut_mid_A": cut_mid, "CN_mid": cn_mid,
            "Cut_paper_A": cut_paper, "CN_paper": cn_paper,
            "Cut_auto_A": cut_auto, "CN_auto": cn_auto,
            "Paper_CN": paper_cn,
            "Diff_CN": safe_diff(cn_paper, paper_cn),
            "Diff_%": safe_pct_diff(cn_paper, paper_cn),
        })

    # Qn and NC
    logger.info("Computing Qn distributions and Network Connectivity...")
    qn_res = compute_qn_and_nc(coords, types, box)
    f_net = compute_f_net(qn_res["qn_combined"], qn_res["n_net"])
    r_factors = compute_r_factors(coords, types, box, counts)

    # Energy log
    energy_df, block_summary, block_df = None, None, None
    if args.energy_log is not None:
        logger.info(f"Reading energy log: {args.energy_log}")
        energy_df = load_energy_log(args.energy_log)
        block_summary, block_df = compute_block_stats(energy_df, args.block_size)

    # Excel output
    output_path = Path(args.output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing Excel file: {output_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ===================== 1. SUMMARY (Professional overview) =====================
        summary_rows = [
            {"Category": "System", "Metric": "Composition", "Simulated": f"45-M{x_val}", "Paper": f"45-M{x_val}", "Diff": "—"},
            {"Category": "System", "Metric": "N_atoms", "Simulated": struct["n_atoms"], "Paper": "—", "Diff": "—"},
            {"Category": "System", "Metric": "Box (A)", "Simulated": round(box, 4), "Paper": "—", "Diff": "—"},
            {"Category": "System", "Metric": "Density (g/cm3)", "Simulated": round(eff_density, 4), "Paper": "—", "Diff": "—"},
        ]

        for pair in present_pairs:
            pair_name = f"{pair[0]}-{pair[1]}"
            bl_row = next((r for r in bond_rows if r["Pair"] == pair_name), None)
            cn_row = next((r for r in cn_rows if r["Pair"] == pair_name), None)
            if bl_row:
                summary_rows.append({"Category": "Bond Length", "Metric": f"{pair_name} Peak (A)",
                                     "Simulated": round(bl_row["Peak_A"], 4) if np.isfinite(bl_row["Peak_A"]) else np.nan,
                                     "Paper": bl_row["Paper_Peak_A"],
                                     "Diff_%": f"{bl_row['Diff_%']:.2f}" if np.isfinite(bl_row['Diff_%']) else "—"})
            if cn_row:
                summary_rows.append({"Category": "Coordination", "Metric": f"{pair_name} CN",
                                     "Simulated": round(cn_row["CN_paper"], 4) if np.isfinite(cn_row["CN_paper"]) else np.nan,
                                     "Paper": cn_row["Paper_CN"],
                                     "Diff_%": f"{cn_row['Diff_%']:.2f}" if np.isfinite(cn_row['Diff_%']) else "—"})

        summary_rows.append({"Category": "Topology", "Metric": "NC_Si", "Simulated": round(qn_res["nc_si"], 4),
                             "Paper": PAPER_NC_SI.get(x_val, np.nan),
                             "Diff_%": f"{safe_pct_diff(qn_res['nc_si'], PAPER_NC_SI.get(x_val, np.nan)):.2f}" if np.isfinite(PAPER_NC_SI.get(x_val, np.nan)) else "—"})
        summary_rows.append({"Category": "Topology", "Metric": "NC_P", "Simulated": round(qn_res["nc_p"], 4),
                             "Paper": PAPER_NC_P.get(x_val, np.nan),
                             "Diff_%": f"{safe_pct_diff(qn_res['nc_p'], PAPER_NC_P.get(x_val, np.nan)):.2f}" if np.isfinite(PAPER_NC_P.get(x_val, np.nan)) else "—"})
        summary_rows.append({"Category": "Topology", "Metric": "NC_Combined", "Simulated": round(qn_res["nc_combined"], 4),
                             "Paper": PAPER_NC.get(x_val, np.nan),
                             "Diff_%": f"{safe_pct_diff(qn_res['nc_combined'], PAPER_NC.get(x_val, np.nan)):.2f}" if np.isfinite(PAPER_NC.get(x_val, np.nan)) else "—"})

        summary_rows.append({"Category": "Network", "Metric": "F_net", "Simulated": round(f_net, 4), "Paper": "—", "Diff": "—"})

        pr = PAPER_R_FACTORS.get(x_val, {})
        for key in ["R_X-X", "R_X-Si", "R_X-P"]:
            summary_rows.append({"Category": "Clustering", "Metric": key,
                                 "Simulated": round(r_factors[key], 4) if np.isfinite(r_factors[key]) else np.nan,
                                 "Paper": pr.get(key, np.nan),
                                 "Diff_%": f"{safe_pct_diff(r_factors[key], pr.get(key, np.nan)):.2f}" if np.isfinite(pr.get(key, np.nan)) else "—"})

        if block_summary:
            summary_rows.append({"Category": "Energy", "Metric": "Block mean (eV/atom)",
                                 "Simulated": round(block_summary["mean_eV_per_atom"], 4), "Paper": "—", "Diff": "—"})
            summary_rows.append({"Category": "Energy", "Metric": "Block std (eV/atom)",
                                 "Simulated": round(block_summary["std_eV_per_atom"], 4), "Paper": "—", "Diff": "—"})

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="01_Summary", index=False)

        # ===================== 2. Metadata =====================
        metadata_rows = [
            {"Parameter": "System", "Value": "Mg-doped 45S5 bioactive glass"},
            {"Parameter": "Reference", "Value": "Moghanian et al."},
            {"Parameter": "Mg_label", "Value": f"45-M{x_val}"},
            {"Parameter": "Input_XYZ", "Value": str(args.input_xyz)},
            {"Parameter": "Energy_log", "Value": str(args.energy_log) if args.energy_log else "None"},
            {"Parameter": "N_atoms", "Value": struct["n_atoms"]},
            {"Parameter": "N_Si", "Value": counts.get("Si", 0)},
            {"Parameter": "N_P", "Value": counts.get("P", 0)},
            {"Parameter": "N_Na", "Value": counts.get("Na", 0)},
            {"Parameter": "N_Ca", "Value": counts.get("Ca", 0)},
            {"Parameter": "N_Mg", "Value": counts.get("Mg", 0)},
            {"Parameter": "N_O", "Value": counts.get("O", 0)},
            {"Parameter": "Box_A", "Value": box},
            {"Parameter": "Density_g_cm3", "Value": eff_density},
        ]
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="02_Metadata", index=False)

        # ===================== 3. Bond Lengths =====================
        pd.DataFrame(bond_rows).to_excel(writer, sheet_name="03_Bond_Lengths", index=False)

        # ===================== 4. CN (all cutoffs + paper) =====================
        pd.DataFrame(cn_rows).to_excel(writer, sheet_name="04_CN", index=False)

        # ===================== 5. Qn Si =====================
        n_si = max(1, qn_res["n_si"])
        paper_qn_si = PAPER_QN_SI.get(x_val, {})
        pd.DataFrame([
            {"Qn": f"Q{n}", "Count": qn_res["qn_si"][n],
             "Percentage": round(qn_res["qn_si"][n] / n_si * 100.0, 3),
             "Paper_%": paper_qn_si.get(n, np.nan),
             "Diff_%": safe_diff(qn_res["qn_si"][n] / n_si * 100.0, paper_qn_si.get(n, np.nan))}
            for n in range(5)
        ]).to_excel(writer, sheet_name="05_Qn_Si", index=False)

        # ===================== 6. Qn P =====================
        n_p = max(1, qn_res["n_p"])
        paper_qn_p = PAPER_QN_P.get(x_val, {})
        pd.DataFrame([
            {"Qn": f"Q{n}", "Count": qn_res["qn_p"][n],
             "Percentage": round(qn_res["qn_p"][n] / n_p * 100.0, 3),
             "Paper_%": paper_qn_p.get(n, np.nan),
             "Diff_%": safe_diff(qn_res["qn_p"][n] / n_p * 100.0, paper_qn_p.get(n, np.nan))}
            for n in range(5)
        ]).to_excel(writer, sheet_name="06_Qn_P", index=False)

        # ===================== 7. Qn Combined =====================
        n_net = max(1, qn_res["n_net"])
        paper_qn_combined = PAPER_QN_COMBINED.get(x_val, {})
        pd.DataFrame([
            {"Qn": f"Q{n}", "Count": qn_res["qn_combined"][n],
             "Percentage": round(qn_res["qn_combined"][n] / n_net * 100.0, 3),
             "Paper_%": paper_qn_combined.get(n, np.nan),
             "Diff_%": safe_diff(qn_res["qn_combined"][n] / n_net * 100.0, paper_qn_combined.get(n, np.nan))}
            for n in range(5)
        ]).to_excel(writer, sheet_name="07_Qn_Combined", index=False)

        # ===================== 8. NC =====================
        pd.DataFrame([
            {"Type": "Si", "NC": qn_res["nc_si"], "Paper_NC": PAPER_NC_SI.get(x_val, np.nan),
             "Diff": safe_diff(qn_res["nc_si"], PAPER_NC_SI.get(x_val, np.nan)),
             "Diff_%": safe_pct_diff(qn_res["nc_si"], PAPER_NC_SI.get(x_val, np.nan))},
            {"Type": "P", "NC": qn_res["nc_p"], "Paper_NC": PAPER_NC_P.get(x_val, np.nan),
             "Diff": safe_diff(qn_res["nc_p"], PAPER_NC_P.get(x_val, np.nan)),
             "Diff_%": safe_pct_diff(qn_res["nc_p"], PAPER_NC_P.get(x_val, np.nan))},
            {"Type": "Si-P Combined", "NC": qn_res["nc_combined"], "Paper_NC": PAPER_NC.get(x_val, np.nan),
             "Diff": safe_diff(qn_res["nc_combined"], PAPER_NC.get(x_val, np.nan)),
             "Diff_%": safe_pct_diff(qn_res["nc_combined"], PAPER_NC.get(x_val, np.nan))},
        ]).to_excel(writer, sheet_name="08_NC", index=False)

        # ===================== 9. O Speciation =====================
        n_o = max(1, counts.get("O", 0))
        pd.DataFrame([
            {"Type": k, "Count": v, "Percentage": round(v / n_o * 100.0, 3)}
            for k, v in qn_res["o_speciation"].items()
        ]).to_excel(writer, sheet_name="09_O_Speciation", index=False)

        # ===================== 10. BO Types =====================
        pd.DataFrame([
            {"Type": k, "Count": v,
             "Percentage_of_total_O": round(v / n_o * 100.0, 3),
             "Percentage_of_BO": round(v / max(1, qn_res["o_speciation"]["BO"]) * 100.0, 3)}
            for k, v in qn_res["bo_types"].items()
        ]).to_excel(writer, sheet_name="10_BO_Types", index=False)

        # ===================== 11. F_net =====================
        f_net_rows = [
            {"Parameter": "F_net", "Value": f_net, "Description": "Theoretical network strength (Eq. 4)"},
            {"Parameter": "n_net", "Value": n_net, "Description": "Total network formers (Si+P)"},
        ]
        f_net_rows += [{"Parameter": f"Q{n}_count", "Value": qn_res["qn_combined"][n], "Description": f"Number of Q{n}"} for n in range(5)]
        pd.DataFrame(f_net_rows).to_excel(writer, sheet_name="11_F_net", index=False)

        # ===================== 12. Clustering R =====================
        pr = PAPER_R_FACTORS.get(x_val, {})
        r_factor_rows = [
            {"Parameter": "R_X-X", "Value": r_factors["R_X-X"], "Paper_Value": pr.get("R_X-X", np.nan),
             "Diff": safe_diff(r_factors["R_X-X"], pr.get("R_X-X", np.nan)),
             "Diff_%": safe_pct_diff(r_factors["R_X-X"], pr.get("R_X-X", np.nan))},
            {"Parameter": "R_X-Si", "Value": r_factors["R_X-Si"], "Paper_Value": pr.get("R_X-Si", np.nan),
             "Diff": safe_diff(r_factors["R_X-Si"], pr.get("R_X-Si", np.nan)),
             "Diff_%": safe_pct_diff(r_factors["R_X-Si"], pr.get("R_X-Si", np.nan))},
            {"Parameter": "R_X-P", "Value": r_factors["R_X-P"], "Paper_Value": pr.get("R_X-P", np.nan),
             "Diff": safe_diff(r_factors["R_X-P"], pr.get("R_X-P", np.nan)),
             "Diff_%": safe_pct_diff(r_factors["R_X-P"], pr.get("R_X-P", np.nan))},
        ]
        pd.DataFrame(r_factor_rows).to_excel(writer, sheet_name="12_Clustering_R", index=False)

        # ===================== 13-17. RDF for each pair =====================
        sheet_idx = 13
        for pair in present_pairs:
            e1, e2 = pair
            pair_name = f"{e1}-{e2}"
            r, gr = rdf_data[pair]
            sheet_name = f"{sheet_idx:02d}_RDF_{pair_name}"[:31]  # Excel sheet name limit
            pd.DataFrame({"r_A": r, "g_r": gr}).to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_idx += 1

        # ===================== Energy & Block Averaging =====================
        if energy_df is not None:
            export_energy_sheet(writer, energy_df, args.max_energy_rows)
        if block_summary:
            pd.DataFrame([block_summary]).to_excel(writer, sheet_name="Block_Average", index=False)
        if block_df is not None:
            block_df.to_excel(writer, sheet_name="Block_Means", index=False)

    logger.info("[OK] Analysis completed.")
    logger.info(f"[OK] Excel file saved to: {output_path}")


if __name__ == "__main__":
    main()
    
    
    