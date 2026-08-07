#!/usr/bin/env python3
"""
========================================================================
Final Standalone Structural Analysis for 45S5/Mg Bioglass
Based on: Moghanian et al., Mg-doped 45S5 bioglass manuscript

This analyzer computes:
- RDF, bond lengths, coordination numbers
- Qn for Si, P, and Si-P combined
- NC for Si, P, and Si-P combined
- Oxygen speciation: FO, NBO, BO, TO
- BO types: Si-O-Si, Si-O-P, P-O-P
- Modifier clustering R_XX
- Modifier preference
- Fnet
- Si-O-P linkages
- Bond angle distributions
- Excel export and plots

Usage:
python analyze_mg_final.py Bioglass_Mg5_N11340_seed42/final_structure.xyz
python analyze_mg_final.py final_structure.xyz --energy-log energy_log.csv
python analyze_mg_final.py final_structure.xyz --output-dir analysis_Mg5
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import re
import sys
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
KB_J = 1.380649e-23
EV_TO_J = 1.602176634e-19

TYPE_MAP = {
    "Si": 0,
    "Ca": 1,
    "Na": 2,
    "P": 3,
    "O": 4,
    "Mg": 5,
}

ELEM_MAP = {
    0: "Si",
    1: "Ca",
    2: "Na",
    3: "P",
    4: "O",
    5: "Mg",
}

MASSES = {
    "Si": 28.0855,
    "Ca": 40.078,
    "Na": 22.98977,
    "P": 30.97376,
    "O": 15.999,
    "Mg": 24.305,
}

# Analysis cutoffs adapted for Mg-doped 45S5
CUTOFFS = {
    ("Si", "O"): 2.25,
    ("P", "O"): 2.25,
    ("Na", "O"): 3.34,
    ("Ca", "O"): 3.14,
    ("Mg", "O"): 2.60,
    ("O", "O"): 2.91,

    ("P", "Ca"): 4.44,
    ("P", "Na"): 4.44,
    ("P", "Mg"): 3.90,

    ("Si", "Ca"): 4.37,
    ("Si", "Na"): 4.42,
    ("Si", "Mg"): 3.85,

    ("Ca", "Ca"): 4.85,
    ("Na", "Na"): 4.15,
    ("Mg", "Mg"): 4.20,

    ("Ca", "Na"): 4.94,
    ("Ca", "Mg"): 4.30,
    ("Na", "Mg"): 3.80,
}

PEAK_RANGES = {
    ("Si", "O"): (1.45, 1.80),
    ("P", "O"): (1.35, 1.70),
    ("Na", "O"): (2.00, 2.70),
    ("Ca", "O"): (2.10, 2.70),
    ("Mg", "O"): (1.70, 2.30),
    ("O", "O"): (2.00, 3.20),

    ("P", "Ca"): (3.0, 5.0),
    ("P", "Na"): (3.0, 5.0),
    ("P", "Mg"): (3.0, 4.8),

    ("Si", "Ca"): (3.0, 5.0),
    ("Si", "Na"): (2.8, 5.0),
    ("Si", "Mg"): (3.0, 4.8),

    ("Ca", "Ca"): (3.0, 5.5),
    ("Na", "Na"): (2.5, 5.0),
    ("Mg", "Mg"): (3.0, 5.2),

    ("Ca", "Na"): (3.0, 5.5),
    ("Ca", "Mg"): (3.2, 5.3),
    ("Na", "Mg"): (3.0, 4.8),
}

RMIN_VALUES = {
    ("Si", "O"): 1.8,
    ("P", "O"): 1.7,
    ("Na", "O"): 2.7,
    ("Ca", "O"): 2.7,
    ("Mg", "O"): 2.2,
    ("O", "O"): 2.2,

    ("P", "Ca"): 3.5,
    ("P", "Na"): 3.5,
    ("P", "Mg"): 3.0,

    ("Si", "Ca"): 3.5,
    ("Si", "Na"): 3.0,
    ("Si", "Mg"): 3.0,

    ("Ca", "Ca"): 3.5,
    ("Na", "Na"): 3.0,
    ("Mg", "Mg"): 3.2,

    ("Ca", "Na"): 3.5,
    ("Ca", "Mg"): 3.3,
    ("Na", "Mg"): 3.0,
}

PAPER_BOND_LENGTHS = {
    ("Si", "O"): 1.606,
    ("P", "O"): 1.50,
    ("Ca", "O"): 2.36,
    ("Na", "O"): 2.40,
    ("Mg", "O"): 1.98,
}

PAPER_CN = {
    "Si": 4.0,
    "P": 4.0,
    "Ca": 6.18,
    "Na": 6.4,
    "Mg": 4.1,
}

PAPER_NC_BY_X = {
    0: 1.926,
    1: 1.922,
    3: 1.938,
    5: 1.953,
    8: 1.953,
    10: 1.981,
    15: 1.973,
    20: 2.035,
}

SBS_X_O = {
    "Ca": 32.0,
    "Na": 20.0,
    "Mg": 32.0,
}

NV_MODIFIER = {
    "Ca": 2,
    "Na": 1,
    "Mg": 2,
}


# ========================= UTILITIES =========================
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

    meta = {
        "x": 0,
        "rho": None,
        "box": None,
        "seed": None,
    }

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

    symbols = []
    coords = []

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

    counts = {
        e: int(np.sum(types == TYPE_MAP[e]))
        for e in TYPE_MAP
    }

    return {
        "symbols": symbols,
        "coords": coords,
        "types": types,
        "n_atoms": len(symbols),
        "counts": counts,
        "meta": meta,
        "comment": comment,
    }


# ========================= RDF =========================
def build_rdf_histograms(coords, types, counts, box, present_pairs, rmax=8.0, nbins=800):
    tree = cKDTree(coords, boxsize=box)
    pairs = tree.query_pairs(rmax, output_type="ndarray")

    hists = {p: np.zeros(nbins, dtype=np.float64) for p in present_pairs}

    pair_lookup = {}
    for p in present_pairs:
        pair_lookup[p] = p
        if p[0] != p[1]:
            pair_lookup[(p[1], p[0])] = p

    bw = rmax / nbins

    for i, j in pairs:
        ei = ELEM_MAP[types[i]]
        ej = ELEM_MAP[types[j]]

        key = pair_lookup.get((ei, ej))
        if key is None:
            continue

        d = coords[i] - coords[j]
        d = minimum_image(d, box)
        r = np.linalg.norm(d)

        if r <= 0.0 or r >= rmax:
            continue

        b = int(r / bw)
        if b < nbins:
            hists[key][b] += 1.0

    r_edges = np.linspace(0.0, rmax, nbins + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    shell = (4.0 / 3.0) * pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    shell[shell <= 0.0] = 1.0e-12

    V = box ** 3
    rdf_data = {}

    for p in present_pairs:
        e1, e2 = p
        n1 = counts.get(e1, 0)
        n2 = counts.get(e2, 0)

        if e1 == e2:
            denom = n1 * (n1 - 1)
            factor = (2.0 * V / denom) if denom > 0 else 0.0
        else:
            denom = n1 * n2
            factor = (V / denom) if denom > 0 else 0.0

        gr = hists[p] * factor / shell
        gr[~np.isfinite(gr)] = 0.0
        rdf_data[p] = (r, gr)

    return r, rdf_data


def find_first_peak(r, gr, pair):
    key = pair if pair in PEAK_RANGES else (pair[1], pair[0])
    rmin, rmax = PEAK_RANGES.get(key, (1.0, 4.0))

    mask = (r >= rmin) & (r <= rmax)
    if not np.any(mask):
        return np.nan

    rm = r[mask]
    gm = gr[mask]

    if len(gm) < 3 or np.max(gm) <= 0:
        return np.nan

    gm_smooth = gaussian_filter1d(gm, sigma=1.0)
    return float(rm[np.argmax(gm_smooth)])


def find_first_minimum(r, gr, pair):
    key = pair if pair in RMIN_VALUES else (pair[1], pair[0])
    rmin = RMIN_VALUES.get(key, 1.5)
    rmax = rmin + 1.5

    mask = (r >= rmin) & (r <= rmax)
    if not np.any(mask):
        return np.nan

    rm = r[mask]
    gm = gr[mask]

    if len(gm) < 3:
        return np.nan

    gm_smooth = gaussian_filter1d(gm, sigma=2.0)
    diff = np.diff(gm_smooth)

    for i in range(1, len(diff)):
        if diff[i - 1] < 0.0 and diff[i] >= 0.0:
            return float(rm[i])

    return float(rm[np.argmin(gm_smooth)])


def coordination_number(r, gr, cutoff, n_target, box, same):
    if n_target <= 0:
        return 0.0

    if same and n_target < 2:
        return 0.0

    mask = (r > 0.3) & (r <= cutoff)
    if not np.any(mask):
        return 0.0

    rm = r[mask]
    gm = gr[mask]

    rho = (n_target - 1) / box ** 3 if same else n_target / box ** 3

    cn = trapezoid(4.0 * pi * rm ** 2 * rho * gm, rm)
    return float(cn)


# ========================= ANGLES =========================
def angle_x_y_x(coords, types, box, center_elem, ligand_elem, cutoff):
    center_type = TYPE_MAP[center_elem]
    ligand_type = TYPE_MAP[ligand_elem]

    centers = np.where(types == center_type)[0]
    if len(centers) == 0:
        return np.array([])

    tree = cKDTree(coords, boxsize=box)
    angles = []

    for c in centers:
        neigh = tree.query_ball_point(coords[c], cutoff)
        ligands = []

        for j in neigh:
            if j == c or types[j] != ligand_type:
                continue

            d = minimum_image(coords[c] - coords[j], box)
            r = np.linalg.norm(d)

            if r < cutoff:
                ligands.append(j)

        for a in range(len(ligands)):
            for b in range(a + 1, len(ligands)):
                v1 = minimum_image(coords[ligands[a]] - coords[c], box)
                v2 = minimum_image(coords[ligands[b]] - coords[c], box)

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 1.0e-8 and n2 > 1.0e-8:
                    cosang = np.dot(v1, v2) / (n1 * n2)
                    cosang = np.clip(cosang, -1.0, 1.0)
                    angles.append(np.degrees(np.arccos(cosang)))

    return np.array(angles)


def angle_y_x_y(coords, types, box, bridge_elem, ligand_elem, cutoff):
    bridge_type = TYPE_MAP[bridge_elem]
    ligand_type = TYPE_MAP[ligand_elem]

    bridges = np.where(types == bridge_type)[0]
    if len(bridges) == 0:
        return np.array([])

    tree = cKDTree(coords, boxsize=box)
    angles = []

    for b in bridges:
        neigh = tree.query_ball_point(coords[b], cutoff)
        ligands = []

        for j in neigh:
            if j == b or types[j] != ligand_type:
                continue

            d = minimum_image(coords[b] - coords[j], box)
            r = np.linalg.norm(d)

            if r < cutoff:
                ligands.append(j)

        for a in range(len(ligands)):
            for c in range(a + 1, len(ligands)):
                v1 = minimum_image(coords[ligands[a]] - coords[b], box)
                v2 = minimum_image(coords[ligands[c]] - coords[b], box)

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 1.0e-8 and n2 > 1.0e-8:
                    cosang = np.dot(v1, v2) / (n1 * n2)
                    cosang = np.clip(cosang, -1.0, 1.0)
                    angles.append(np.degrees(np.arccos(cosang)))

    return np.array(angles)


def angle_si_o_p(coords, types, box):
    o_type = TYPE_MAP["O"]
    si_type = TYPE_MAP["Si"]
    p_type = TYPE_MAP["P"]

    si_o_cut = CUTOFFS[("Si", "O")]
    p_o_cut = CUTOFFS[("P", "O")]

    o_idx = np.where(types == o_type)[0]
    if len(o_idx) == 0:
        return np.array([])

    tree = cKDTree(coords, boxsize=box)
    angles = []
    search_cut = max(si_o_cut, p_o_cut)

    for o in o_idx:
        neigh = tree.query_ball_point(coords[o], search_cut)
        si_list = []
        p_list = []

        for j in neigh:
            if j == o:
                continue

            d = minimum_image(coords[o] - coords[j], box)
            r = np.linalg.norm(d)

            if types[j] == si_type and r < si_o_cut:
                si_list.append(j)
            elif types[j] == p_type and r < p_o_cut:
                p_list.append(j)

        for si in si_list:
            for p in p_list:
                v1 = minimum_image(coords[si] - coords[o], box)
                v2 = minimum_image(coords[p] - coords[o], box)

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 1.0e-8 and n2 > 1.0e-8:
                    cosang = np.dot(v1, v2) / (n1 * n2)
                    cosang = np.clip(cosang, -1.0, 1.0)
                    angles.append(np.degrees(np.arccos(cosang)))

    return np.array(angles)


def compute_angles(coords, types, box):
    logger.info("Computing bond angle distributions...")

    angles = {}

    angles["O-Si-O"] = angle_x_y_x(
        coords, types, box,
        "Si", "O",
        CUTOFFS[("Si", "O")]
    )

    angles["O-P-O"] = angle_x_y_x(
        coords, types, box,
        "P", "O",
        CUTOFFS[("P", "O")]
    )

    angles["Si-O-Si"] = angle_y_x_y(
        coords, types, box,
        "O", "Si",
        CUTOFFS[("Si", "O")]
    )

    angles["Si-O-P"] = angle_si_o_p(coords, types, box)

    return angles


# ========================= Qn / O SPECIATION =========================
def compute_qn_and_speciation(coords, types, box):
    logger.info("Computing Qn distributions and oxygen speciation...")

    si_type = TYPE_MAP["Si"]
    p_type = TYPE_MAP["P"]
    o_type = TYPE_MAP["O"]

    si_o_cut = CUTOFFS[("Si", "O")]
    p_o_cut = CUTOFFS[("P", "O")]

    tree = cKDTree(coords, boxsize=box)

    o_idx = np.where(types == o_type)[0]
    si_idx = np.where(types == si_type)[0]
    p_idx = np.where(types == p_type)[0]

    o_nf_count = []
    o_nf_types = []

    search_cut = max(si_o_cut, p_o_cut)

    for o in o_idx:
        neigh = tree.query_ball_point(coords[o], search_cut)
        nf_types = []

        for j in neigh:
            if j == o:
                continue

            d = minimum_image(coords[o] - coords[j], box)
            r = np.linalg.norm(d)

            if types[j] == si_type and r < si_o_cut:
                nf_types.append(si_type)
            elif types[j] == p_type and r < p_o_cut:
                nf_types.append(p_type)

        o_nf_count.append(len(nf_types))
        o_nf_types.append(tuple(sorted(nf_types)))

    fo = sum(1 for c in o_nf_count if c == 0)
    nbo = sum(1 for c in o_nf_count if c == 1)
    bo = sum(1 for c in o_nf_count if c == 2)
    to = sum(1 for c in o_nf_count if c >= 3)

    bo_types = {
        "Si-O-Si": 0,
        "Si-O-P": 0,
        "P-O-P": 0,
    }

    for nf_t in o_nf_types:
        if len(nf_t) == 2:
            if nf_t[0] == si_type and nf_t[1] == si_type:
                bo_types["Si-O-Si"] += 1
            elif nf_t[0] == si_type and nf_t[1] == p_type:
                bo_types["Si-O-P"] += 1
            elif nf_t[0] == p_type and nf_t[1] == p_type:
                bo_types["P-O-P"] += 1

    bridging_set = {
        o_idx[i] for i, c in enumerate(o_nf_count) if c >= 2
    }

    def qn_for(center_idx, center_cut):
        qn_values = []

        for c in center_idx:
            neigh = tree.query_ball_point(coords[c], center_cut)
            bcnt = 0

            for j in neigh:
                if j == c or types[j] != o_type:
                    continue

                d = minimum_image(coords[c] - coords[j], box)
                r = np.linalg.norm(d)

                if r < center_cut and j in bridging_set:
                    bcnt += 1

            qn_values.append(min(bcnt, 4))

        qn_values = np.array(qn_values, dtype=np.int32)

        counts_qn = {
            n: int(np.sum(qn_values == n))
            for n in range(5)
        }

        return counts_qn

    qn_si = qn_for(si_idx, si_o_cut)
    qn_p = qn_for(p_idx, p_o_cut)

    qn_combined = {
        n: qn_si[n] + qn_p[n]
        for n in range(5)
    }

    oxygen_speciation = {
        "FO": fo,
        "NBO": nbo,
        "BO": bo,
        "TO": to,
    }

    return qn_si, qn_p, qn_combined, oxygen_speciation, bo_types


def nc_from_qn_counts(qn_counts, total_centers):
    if total_centers <= 0:
        return 0.0

    return sum(n * qn_counts[n] for n in range(5)) / total_centers


# ========================= CLUSTERING / PREFERENCE =========================
def compute_rxx(cn_obs, cutoff, n_elem, box):
    if n_elem < 2:
        return 0.0, cn_obs, 0.0

    V = box ** 3
    number_density = (n_elem - 1) / V
    cn_hom = (4.0 / 3.0) * pi * cutoff ** 3 * number_density

    if cn_hom <= 0.0:
        return 0.0, cn_obs, cn_hom

    rxx = cn_obs / cn_hom
    return rxx, cn_obs, cn_hom


def compute_modifier_preference(coords, types, counts, box, rdf_data, cn_fixed):
    logger.info("Computing modifier preference ratios...")

    preferences = {}

    for A in ["Si", "P"]:
        if counts.get(A, 0) == 0:
            continue

        for B, C in [("Mg", "Ca"), ("Ca", "Na"), ("Mg", "Na")]:
            if counts.get(B, 0) == 0 or counts.get(C, 0) == 0:
                continue

            pair_ab = (A, B) if (A, B) in CUTOFFS else (B, A)
            pair_ac = (A, C) if (A, C) in CUTOFFS else (C, A)

            cn_ab = cn_fixed.get(pair_ab, 0.0)
            cn_ac = cn_fixed.get(pair_ac, 0.0)

            key = f"{A}_{B}_vs_{C}"

            if cn_ac > 0.0:
                preferences[key] = (cn_ab / cn_ac) * (counts[C] / counts[B])
            else:
                preferences[key] = 0.0

    return preferences


def compute_fnet(counts, cn_fixed, nc_combined):
    n_net = counts.get("Si", 0) + counts.get("P", 0)

    if n_net <= 0:
        return 0.0

    s = 0.0

    for elem in ["Ca", "Na", "Mg"]:
        cx = counts.get(elem, 0)
        if cx == 0:
            continue

        pair = (elem, "O")
        cn_o = cn_fixed.get(pair, 0.0)

        s += cx * NV_MODIFIER[elem] * SBS_X_O[elem] * cn_o * nc_combined

    return s / n_net


def compute_si_o_p_links(coords, types, box):
    logger.info("Computing Si-O-P linkages...")

    si_type = TYPE_MAP["Si"]
    p_type = TYPE_MAP["P"]
    o_type = TYPE_MAP["O"]

    si_o_cut = CUTOFFS[("Si", "O")]
    p_o_cut = CUTOFFS[("P", "O")]

    p_idx = np.where(types == p_type)[0]

    if len(p_idx) == 0:
        return 0, 0, 0.0

    tree = cKDTree(coords, boxsize=box)

    p_si_links = 0
    total_p_o = 0

    for p in p_idx:
        neigh_p = tree.query_ball_point(coords[p], p_o_cut)
        o_list = []

        for j in neigh_p:
            if j == p or types[j] != o_type:
                continue

            d = minimum_image(coords[p] - coords[j], box)
            r = np.linalg.norm(d)

            if r < p_o_cut:
                o_list.append(j)

        total_p_o += len(o_list)

        for o in o_list:
            neigh_o = tree.query_ball_point(coords[o], si_o_cut)
            found_si = False

            for k in neigh_o:
                if k == o or types[k] != si_type:
                    continue

                d = minimum_image(coords[o] - coords[k], box)
                r = np.linalg.norm(d)

                if r < si_o_cut:
                    found_si = True
                    break

            if found_si:
                p_si_links += 1

    frac = p_si_links / total_p_o if total_p_o > 0 else 0.0
    return p_si_links, total_p_o, frac


# ========================= ENERGY LOG =========================
def analyze_energy_log(path, n_atoms):
    path = Path(path)
    if not path.exists():
        return None

    logger.info(f"Reading energy log: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"Could not read energy log: {e}")
        return None

    if "Total_eV_per_atom" not in df.columns:
        logger.warning("Energy log does not have Total_eV_per_atom column.")
        return None

    total = df["Total_eV_per_atom"].values
    if len(total) == 0:
        return None

    stats = {
        "n_points": len(total),
        "mean_eV_per_atom": float(np.mean(total)),
        "final_eV_per_atom": float(total[-1]),
        "std_eV_per_atom": float(np.std(total, ddof=1)) if len(total) > 1 else 0.0,
    }

    block_size = 5000
    if len(total) >= 3 * block_size:
        n_blocks = len(total) // block_size
        block_avgs = [
            np.mean(total[i * block_size:(i + 1) * block_size])
            for i in range(n_blocks)
        ]
        stats["block_mean_eV_per_atom"] = float(np.mean(block_avgs))
        stats["block_std_eV_per_atom"] = float(np.std(block_avgs, ddof=1) / np.sqrt(n_blocks))
    else:
        stats["block_mean_eV_per_atom"] = stats["mean_eV_per_atom"]
        stats["block_std_eV_per_atom"] = 0.0

    # Heat capacity from energy fluctuations at 300 K
    energies_j = total * n_atoms * EV_TO_J
    var_e = np.var(energies_j, ddof=1) if len(energies_j) > 1 else 0.0
    cv_system = var_e / (KB_J * 300.0 ** 2)
    cv_mol = cv_system * NA / n_atoms

    stats["Cv_J_per_mol_K"] = float(cv_mol)

    return stats


# ========================= EXCEL EXPORT =========================
def export_excel(results, output_path):
    logger.info(f"Exporting Excel file: {output_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Metadata
        md_rows = [
            {"Parameter": k, "Value": v}
            for k, v in results["metadata"].items()
        ]
        pd.DataFrame(md_rows).to_excel(writer, sheet_name="Metadata", index=False)

        # Bond lengths
        bl_rows = []
        for pair, bl in results["bond_lengths"].items():
            paper = PAPER_BOND_LENGTHS.get(pair, PAPER_BOND_LENGTHS.get((pair[1], pair[0]), np.nan))
            bl_rows.append({
                "Pair": f"{pair[0]}-{pair[1]}",
                "Bond_length_A": bl,
                "Paper_value_A": paper,
            })
        pd.DataFrame(bl_rows).to_excel(writer, sheet_name="Bond_Lengths", index=False)

        # CN
        cn_rows = []
        for pair, cn in results["cn_fixed"].items():
            cn_auto = results["cn_auto"].get(pair, cn)
            cutoff_fixed = CUTOFFS.get(pair, np.nan)
            cutoff_auto = results["auto_cutoff"].get(pair, cutoff_fixed)

            cn_rows.append({
                "Pair": f"{pair[0]}-{pair[1]}",
                "CN_fixed": cn,
                "Cutoff_fixed_A": cutoff_fixed,
                "CN_auto": cn_auto,
                "Cutoff_auto_A": cutoff_auto,
            })
        pd.DataFrame(cn_rows).to_excel(writer, sheet_name="CN", index=False)

        # Qn Si
        n_si = results["metadata"].get("N_Si", 0)
        qn_si_rows = []
        for n in range(5):
            cnt = results["qn_si_counts"][n]
            qn_si_rows.append({
                "Qn": f"Q{n}",
                "Count": cnt,
                "Percentage": cnt / n_si * 100.0 if n_si > 0 else 0.0,
            })
        pd.DataFrame(qn_si_rows).to_excel(writer, sheet_name="Qn_Si", index=False)

        # Qn P
        n_p = results["metadata"].get("N_P", 0)
        qn_p_rows = []
        for n in range(5):
            cnt = results["qn_p_counts"][n]
            qn_p_rows.append({
                "Qn": f"Q{n}",
                "Count": cnt,
                "Percentage": cnt / n_p * 100.0 if n_p > 0 else 0.0,
            })
        pd.DataFrame(qn_p_rows).to_excel(writer, sheet_name="Qn_P", index=False)

        # Qn Combined
        n_net = n_si + n_p
        qn_c_rows = []
        for n in range(5):
            cnt = results["qn_combined_counts"][n]
            qn_c_rows.append({
                "Qn": f"Q{n}",
                "Count": cnt,
                "Percentage": cnt / n_net * 100.0 if n_net > 0 else 0.0,
            })
        pd.DataFrame(qn_c_rows).to_excel(writer, sheet_name="Qn_Combined", index=False)

        # NC: Si, P, Combined
        x_val = results["metadata"].get("Mg_label_x", 0)
        paper_nc_combined = PAPER_NC_BY_X.get(int(x_val), np.nan)

        nc_rows = [
            {
                "Type": "Si",
                "NC": results["nc"]["Si"],
                "Paper_NC": np.nan,
            },
            {
                "Type": "P",
                "NC": results["nc"]["P"],
                "Paper_NC": np.nan,
            },
            {
                "Type": "Si-P Combined",
                "NC": results["nc"]["Combined"],
                "Paper_NC": paper_nc_combined,
            },
        ]
        pd.DataFrame(nc_rows).to_excel(writer, sheet_name="NC", index=False)

        # Oxygen speciation
        n_o = results["metadata"].get("N_O", 0)
        os_rows = []
        for k, v in results["oxygen_speciation"].items():
            os_rows.append({
                "Type": k,
                "Count": v,
                "Percentage": v / n_o * 100.0 if n_o > 0 else 0.0,
            })
        pd.DataFrame(os_rows).to_excel(writer, sheet_name="O_speciation", index=False)

        # BO types
        bo_rows = []
        for k, v in results["bo_types"].items():
            bo_rows.append({
                "Type": k,
                "Count": v,
                "Percentage_of_total_O": v / n_o * 100.0 if n_o > 0 else 0.0,
            })
        pd.DataFrame(bo_rows).to_excel(writer, sheet_name="BO_types", index=False)

        # R_XX clustering
        rxx_rows = []
        for elem, vals in results["rxx"].items():
            rxx_rows.append({
                "Element": elem,
                "R_XX": vals[0],
                "CN_obs": vals[1],
                "CN_hom": vals[2],
            })
        pd.DataFrame(rxx_rows).to_excel(writer, sheet_name="R_XX", index=False)

        # Modifier preference
        pref_rows = [
            {"Ratio": k, "Value": v}
            for k, v in results["modifier_preference"].items()
        ]
        if pref_rows:
            pd.DataFrame(pref_rows).to_excel(writer, sheet_name="Modifier_Preference", index=False)

        # Fnet
        pd.DataFrame({
            "Fnet": [results["fnet"]],
            "SBS_Ca": [SBS_X_O["Ca"]],
            "SBS_Na": [SBS_X_O["Na"]],
            "SBS_Mg": [SBS_X_O["Mg"]],
        }).to_excel(writer, sheet_name="Fnet", index=False)

        # Si-O-P
        pd.DataFrame({
            "Total_P-O_bonds": [results["total_p_o_bonds"]],
            "P-O-Si_bonds": [results["p_si_links"]],
            "Fraction_P-O-Si": [results["frac_p_si"]],
        }).to_excel(writer, sheet_name="Si-O-P", index=False)

        # Angles summary
        ang_rows = []
        for name, arr in results["angles"].items():
            ang_rows.append({
                "Angle": name,
                "Mean_deg": float(np.mean(arr)) if len(arr) > 0 else np.nan,
                "Count": len(arr),
            })
        pd.DataFrame(ang_rows).to_excel(writer, sheet_name="Angles_Summary", index=False)

        # Energy stats
        if results.get("energy_stats") is not None:
            energy_rows = [
                {"Parameter": k, "Value": v}
                for k, v in results["energy_stats"].items()
            ]
            pd.DataFrame(energy_rows).to_excel(writer, sheet_name="Energy_Stats", index=False)

        # RDF sheets
        for pair, (r, gr) in results["rdf_data"].items():
            df = pd.DataFrame({
                "r_A": r,
                "g_r": gr,
            })
            sheet_name = f"RDF_{pair[0]}-{pair[1]}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Excel export completed.")


# ========================= PLOTS =========================
def make_plots(results, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # RDF plot
    rdf_items = list(results["rdf_data"].items())
    n_pairs = min(len(rdf_items), 20)

    if n_pairs > 0:
        ncols = 4
        nrows = int(np.ceil(n_pairs / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for idx in range(n_pairs):
            pair, (r, gr) = rdf_items[idx]
            ax = axes[idx]

            ax.plot(r, gr, lw=1.0, color="#2980b9")

            cut = CUTOFFS.get(pair)
            if cut is not None:
                ax.axvline(cut, color="#e74c3c", ls="--", lw=0.8,
                           label=f"Cutoff={cut:.2f} A")
                ax.legend(fontsize=7)

            ax.set_xlabel("r (A)")
            ax.set_ylabel("g(r)")
            ax.set_title(f"{pair[0]}-{pair[1]}")
            ax.set_xlim(0, 8)
            ax.grid(alpha=0.25)

        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(plots_dir / "rdf_all.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Qn plot
    labels = [f"Q{n}" for n in range(5)]
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    n_si = max(1, results["metadata"].get("N_Si", 1))
    n_p = max(1, results["metadata"].get("N_P", 1))
    n_net = max(1, results["metadata"].get("N_Si", 0) + results["metadata"].get("N_P", 0))

    si_pct = [results["qn_si_counts"][n] / n_si * 100.0 for n in range(5)]
    p_pct = [results["qn_p_counts"][n] / n_p * 100.0 for n in range(5)]
    c_pct = [results["qn_combined_counts"][n] / n_net * 100.0 for n in range(5)]

    axes[0].bar(labels, si_pct, color=colors, edgecolor="white")
    axes[0].set_title("Si Qn Distribution")
    axes[0].set_ylabel("%")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(labels, p_pct, color=colors, edgecolor="white")
    axes[1].set_title("P Qn Distribution")
    axes[1].set_ylabel("%")
    axes[1].grid(alpha=0.3, axis="y")

    axes[2].bar(labels, c_pct, color=colors, edgecolor="white")
    axes[2].set_title("Si-P Combined Qn Distribution")
    axes[2].set_ylabel("%")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "qn_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # NC plot
    nc_types = ["Si", "P", "Si-P Combined"]
    nc_vals = [
        results["nc"]["Si"],
        results["nc"]["P"],
        results["nc"]["Combined"],
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(nc_types, nc_vals, color=["#3498db", "#9b59b6", "#e67e22"], edgecolor="white")

    x_val = results["metadata"].get("Mg_label_x", 0)
    paper_nc = PAPER_NC_BY_X.get(int(x_val))
    if paper_nc is not None:
        ax.axhline(paper_nc, color="red", ls="--", label=f"Paper combined NC={paper_nc:.3f}")
        ax.legend()

    ax.set_ylabel("Network Connectivity")
    ax.set_title("NC: Si, P, Si-P Combined")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "nc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Oxygen speciation plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    os_keys = list(results["oxygen_speciation"].keys())
    os_vals = list(results["oxygen_speciation"].values())
    axes[0].bar(os_keys, os_vals, color=["#95a5a6", "#f39c12", "#2ecc71", "#e74c3c"], edgecolor="white")
    axes[0].set_title("Oxygen Speciation")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3, axis="y")

    bo_keys = list(results["bo_types"].keys())
    bo_vals = list(results["bo_types"].values())
    axes[1].bar(bo_keys, bo_vals, color=["#3498db", "#2ecc71", "#9b59b6"], edgecolor="white")
    axes[1].set_title("Bridging Oxygen Types")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "oxygen_speciation.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots saved to: {plots_dir}")


# ========================= MAIN ANALYSIS =========================
def run_analysis(input_xyz, output_dir=None, energy_log=None):
    input_xyz = Path(input_xyz)

    logger.info(f"Reading structure: {input_xyz}")
    struct = read_xyz(input_xyz)

    coords = struct["coords"]
    types = struct["types"]
    counts = struct["counts"]
    meta = struct["meta"]
    x_val = meta.get("x", 0)

    total_mass = sum(MASSES[s] for s in struct["symbols"])

    if meta.get("box") is not None:
        box = float(meta["box"])
    elif meta.get("rho") is not None:
        rho = float(meta["rho"])
        box = ((total_mass / NA) / rho * 1.0e24) ** (1.0 / 3.0)
    else:
        raise ValueError("Neither box nor density could be determined from XYZ header.")

    eff_density = total_mass / (NA * box ** 3) * 1.0e24

    logger.info(f"Box = {box:.4f} A, N = {struct['n_atoms']}, Mg label x = {x_val}")
    logger.info(f"Composition: {counts}")

    if output_dir is None:
        output_dir = Path(f"analysis_Mg{x_val}")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Present pairs
    present_pairs = [
        p for p in CUTOFFS
        if counts.get(p[0], 0) > 0
        and counts.get(p[1], 0) > 0
        and (p[0] != p[1] or counts.get(p[0], 0) > 1)
    ]

    # RDF
    logger.info("Computing RDFs...")
    r_common, rdf_data = build_rdf_histograms(
        coords, types, counts, box,
        present_pairs,
        rmax=8.0,
        nbins=800
    )

    # Bond lengths and CN
    logger.info("Computing bond lengths and coordination numbers...")
    bond_lengths = {}
    cn_fixed = {}
    cn_auto = {}
    auto_cutoff = {}

    for pair, (r, gr) in rdf_data.items():
        bl = find_first_peak(r, gr, pair)
        if not np.isnan(bl):
            bond_lengths[pair] = bl

        cutoff = CUTOFFS[pair]
        n_target = counts.get(pair[1], 0)
        same = pair[0] == pair[1]

        cn_val = coordination_number(r, gr, cutoff, n_target, box, same)
        cn_fixed[pair] = cn_val

        first_min = find_first_minimum(r, gr, pair)

        if not np.isnan(first_min) and first_min > 0.5:
            cn_auto_val = coordination_number(r, gr, first_min, n_target, box, same)
            cn_auto[pair] = cn_auto_val
            auto_cutoff[pair] = first_min
        else:
            cn_auto[pair] = cn_val
            auto_cutoff[pair] = cutoff

    # Angles
    angles = compute_angles(coords, types, box)

    # Qn and oxygen speciation
    qn_si, qn_p, qn_combined, oxygen_speciation, bo_types = compute_qn_and_speciation(
        coords, types, box
    )

    n_si = counts.get("Si", 0)
    n_p = counts.get("P", 0)
    n_net = n_si + n_p

    nc_si = nc_from_qn_counts(qn_si, n_si)
    nc_p = nc_from_qn_counts(qn_p, n_p)
    nc_combined = nc_from_qn_counts(qn_combined, n_net)

    logger.info(f"NC Si = {nc_si:.4f}")
    logger.info(f"NC P = {nc_p:.4f}")
    logger.info(f"NC Si-P Combined = {nc_combined:.4f}")

    # Clustering R_XX
    logger.info("Computing clustering R_XX...")
    rxx = {}

    for elem in ["Na", "Ca", "Mg"]:
        if counts.get(elem, 0) == 0:
            continue

        pair = (elem, elem)
        if pair not in cn_fixed:
            continue

        cutoff = CUTOFFS[pair]
        cn_obs = cn_fixed[pair]
        rxx_val, cn_obs, cn_hom = compute_rxx(cn_obs, cutoff, counts[elem], box)
        rxx[elem] = (rxx_val, cn_obs, cn_hom)

    # Modifier preference
    modifier_preference = compute_modifier_preference(
        coords, types, counts, box, rdf_data, cn_fixed
    )

    # Fnet
    fnet = compute_fnet(counts, cn_fixed, nc_combined)

    # Si-O-P links
    p_si_links, total_p_o, frac_p_si = compute_si_o_p_links(coords, types, box)

    # Energy log
    energy_stats = None
    if energy_log is not None:
        energy_stats = analyze_energy_log(energy_log, struct["n_atoms"])

    # Metadata
    metadata = {
        "System": "45S5 Mg-doped bioactive glass",
        "Reference": "Moghanian et al. manuscript",
        "Mg_label": f"45-M{x_val}",
        "Mg_label_x": x_val,
        "N_atoms": struct["n_atoms"],
        "N_Si": counts.get("Si", 0),
        "N_P": counts.get("P", 0),
        "N_Na": counts.get("Na", 0),
        "N_Ca": counts.get("Ca", 0),
        "N_Mg": counts.get("Mg", 0),
        "N_O": counts.get("O", 0),
        "Box_A": box,
        "Density_g_cm3": eff_density,
        "Seed": meta.get("seed", None),
        "NC_Si": nc_si,
        "NC_P": nc_p,
        "NC_Si_P_Combined": nc_combined,
        "Paper_NC_Combined": PAPER_NC_BY_X.get(int(x_val), np.nan),
    }

    if energy_stats is not None:
        metadata["Energy_mean_eV_per_atom"] = energy_stats.get("mean_eV_per_atom", np.nan)
        metadata["Energy_block_mean_eV_per_atom"] = energy_stats.get("block_mean_eV_per_atom", np.nan)
        metadata["Energy_block_std_eV_per_atom"] = energy_stats.get("block_std_eV_per_atom", np.nan)
        metadata["Heat_capacity_Cv_J_per_mol_K"] = energy_stats.get("Cv_J_per_mol_K", np.nan)

    results = {
        "metadata": metadata,
        "bond_lengths": bond_lengths,
        "cn_fixed": cn_fixed,
        "cn_auto": cn_auto,
        "auto_cutoff": auto_cutoff,
        "rdf_data": rdf_data,
        "angles": angles,
        "qn_si_counts": qn_si,
        "qn_p_counts": qn_p,
        "qn_combined_counts": qn_combined,
        "nc": {
            "Si": nc_si,
            "P": nc_p,
            "Combined": nc_combined,
        },
        "oxygen_speciation": oxygen_speciation,
        "bo_types": bo_types,
        "rxx": rxx,
        "modifier_preference": modifier_preference,
        "fnet": fnet,
        "p_si_links": p_si_links,
        "total_p_o_bonds": total_p_o,
        "frac_p_si": frac_p_si,
        "energy_stats": energy_stats,
    }

    # Export
    excel_path = output_dir / "analysis_results.xlsx"
    export_excel(results, excel_path)

    # Plots
    make_plots(results, plots_dir)

    # Console summary
    print("\n" + "=" * 70)
    print(f"  ANALYSIS SUMMARY - 45S5 + Mg label 45-M{x_val}")
    print("=" * 70)

    print("\n--- Network Connectivity ---")
    print(f"  NC Si            : {nc_si:.4f}")
    print(f"  NC P             : {nc_p:.4f}")
    print(f"  NC Si-P Combined : {nc_combined:.4f}")
    print(f"  Paper NC Combined: {PAPER_NC_BY_X.get(int(x_val), 'N/A')}")

    print("\n--- Qn Distribution (%) ---")
    print("  Si:  " + "  ".join(
        f"Q{n}:{qn_si[n] / n_si * 100.0:.1f}" if n_si > 0 else f"Q{n}:0.0"
        for n in range(5)
    ))
    print("  P:   " + "  ".join(
        f"Q{n}:{qn_p[n] / n_p * 100.0:.1f}" if n_p > 0 else f"Q{n}:0.0"
        for n in range(5)
    ))
    print("  Comb:" + "  ".join(
        f"Q{n}:{qn_combined[n] / n_net * 100.0:.1f}" if n_net > 0 else f"Q{n}:0.0"
        for n in range(5)
    ))

    print("\n--- Oxygen Speciation ---")
    n_o = counts.get("O", 0)
    for k, v in oxygen_speciation.items():
        print(f"  {k:4s}: {v:6d}  ({v / n_o * 100.0:5.2f}%)" if n_o > 0 else f"  {k}: 0")

    print("\n--- BO Types ---")
    for k, v in bo_types.items():
        print(f"  {k:8s}: {v:6d}  ({v / n_o * 100.0:5.2f}%)" if n_o > 0 else f"  {k}: 0")

    print("\n--- Clustering R_XX ---")
    for elem, vals in rxx.items():
        print(f"  {elem}: R={vals[0]:.3f}  CN_obs={vals[1]:.2f}  CN_hom={vals[2]:.2f}")

    print("\n--- Modifier Preference ---")
    for k, v in modifier_preference.items():
        print(f"  {k}: {v:.3f}")

    print("\n--- Fnet ---")
    print(f"  Fnet = {fnet:.3f}")

    print("\n--- Si-O-P ---")
    print(f"  Total P-O bonds : {total_p_o}")
    print(f"  P-O-Si bonds    : {p_si_links}")
    print(f"  Fraction P-O-Si : {frac_p_si:.3f}")

    print("=" * 70)
    print(f"\n[OK] Results saved to: {output_dir}")


# ========================= CLI =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final standalone structural analysis for 45S5/Mg bioglass"
    )

    parser.add_argument(
        "input",
        help="Path to final_structure.xyz"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: analysis_Mg{x}"
    )

    parser.add_argument(
        "--energy-log",
        type=str,
        default=None,
        help="Optional path to energy_log.csv"
    )

    args = parser.parse_args()

    try:
        run_analysis(
            args.input,
            output_dir=args.output_dir,
            energy_log=args.energy_log
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)