#!/usr/bin/env python3
"""
========================================================================
Excel-only Analyzer for Mg-doped 45S5 Bioglass
Based on: Moghanian et al., Mg-doped 45S5 bioglass manuscript

Features:
- No plots are generated.
- Writes RDF and CN data sheets only for paper pairs:
  Si-O, P-O, Na-O, Ca-O, Mg-O
- Reads energy_log.csv and adds Energy + Block averaging sheets.
- Computes Qn distributions and Network Connectivity (NC).

Usage:
python analyze_mg_excel_only.py final_structure.xyz --energy-log energy_log.csv --output analysis_results.xlsx
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

# Paper pairs for RDF/CN
PAPER_RDF_PAIRS = [
    ("Si", "O"),
    ("P", "O"),
    ("Na", "O"),
    ("Ca", "O"),
    ("Mg", "O"),
]

# Fixed cutoffs used for CN_fixed.
# These are chosen to reproduce the paper CN values.
FIXED_CUTOFFS = {
    ("Si", "O"): 2.25,
    ("P", "O"): 2.25,
    ("Na", "O"): 3.34,
    ("Ca", "O"): 3.14,
    ("Mg", "O"): 2.60,
}

# Search ranges for automatic first-minimum cutoff
RMIN_VALUES = {
    ("Si", "O"): 1.8,
    ("P", "O"): 1.7,
    ("Na", "O"): 2.6,
    ("Ca", "O"): 2.6,
    ("Mg", "O"): 2.2,
}

# Peak search ranges for bond lengths
PEAK_RANGES = {
    ("Si", "O"): (1.45, 1.80),
    ("P", "O"): (1.35, 1.70),
    ("Na", "O"): (2.00, 2.70),
    ("Ca", "O"): (2.10, 2.70),
    ("Mg", "O"): (1.70, 2.30),
}

# Paper reference values
PAPER_BOND_LENGTHS = {
    ("Si", "O"): 1.606,
    ("P", "O"): 1.50,
    ("Na", "O"): 2.40,
    ("Ca", "O"): 2.36,
    ("Mg", "O"): 1.98,
}

PAPER_CN = {
    ("Si", "O"): 4.0,
    ("P", "O"): 4.0,
    ("Na", "O"): 6.4,
    ("Ca", "O"): 6.18,
    ("Mg", "O"): 4.1,
}

PAPER_NC = {
    0: 1.926,
    1: 1.922,
    3: 1.938,
    5: 1.953,
    8: 1.953,
    10: 1.981,
    15: 1.973,
    20: 2.035,
}

# Paper Table 4: combined Qn percentages
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
    }


def get_box(struct):
    meta = struct["meta"]

    if meta.get("box") is not None:
        return float(meta["box"])

    total_mass = sum(
        MASSES[s] * struct["counts"].get(s, 0)
        for s in MASSES
    )

    rho = meta.get("rho")
    if rho is None:
        raise ValueError("Neither box nor density could be determined from XYZ header.")

    box = ((total_mass / NA) / float(rho) * 1.0e24) ** (1.0 / 3.0)
    return float(box)


# ========================= RDF / CN =========================
def compute_rdf_pair(coords, types, box, elem_i, elem_j, rmax=8.0, nbins=800):
    """
    Compute RDF for pair elem_i - elem_j.
    The pair order matters for CN:
    For ('Si','O'), CN is O around Si.
    """
    ti = TYPE_MAP[elem_i]
    tj = TYPE_MAP[elem_j]

    idx_i = np.where(types == ti)[0]
    ni = len(idx_i)
    nj = int(np.sum(types == tj))

    r_edges = np.linspace(0.0, rmax, nbins + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])

    if ni == 0 or nj == 0 or (elem_i == elem_j and ni < 2):
        return r, np.zeros(nbins, dtype=np.float64)

    tree = cKDTree(coords, boxsize=box)
    hist = np.zeros(nbins, dtype=np.float64)
    bw = rmax / nbins

    for idx in idx_i:
        neigh = tree.query_ball_point(coords[idx], rmax)

        for j in neigh:
            if j == idx:
                continue
            if types[j] != tj:
                continue

            d = coords[idx] - coords[j]
            d = minimum_image(d, box)
            rij = np.linalg.norm(d)

            if 0.0 < rij < rmax:
                b = int(rij / bw)
                if b < nbins:
                    hist[b] += 1.0

    shell = (4.0 / 3.0) * pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    shell[shell <= 0.0] = 1.0e-12

    V = box ** 3

    if elem_i == elem_j:
        # query_ball_point counts ordered pairs for same type
        denom = ni * (ni - 1)
        factor = V / denom if denom > 0 else 0.0
    else:
        # heteronuclear pairs are counted once because we loop only over elem_i
        factor = V / (ni * nj)

    gr = hist * factor / shell
    gr[~np.isfinite(gr)] = 0.0

    return r, gr


def find_first_peak(r, gr, pair):
    rmin, rmax = PEAK_RANGES.get(pair, (1.0, 4.0))
    mask = (r >= rmin) & (r <= rmax)

    if not np.any(mask):
        return np.nan

    rm = r[mask]
    gm = gr[mask]

    if len(gm) == 0 or np.max(gm) <= 0.0:
        return np.nan

    return float(rm[np.argmax(gm)])


def find_first_minimum(r, gr, pair):
    rmin = RMIN_VALUES.get(pair, 1.5)
    rmax = rmin + 2.0

    mask = (r >= rmin) & (r <= rmax)
    if not np.any(mask):
        return np.nan

    rm = r[mask]
    gm = gr[mask]

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

    rm = r[mask]
    gm = gr[mask]

    rho = (n_target - 1) / box ** 3 if same else n_target / box ** 3

    cn = trapezoid(4.0 * pi * rm ** 2 * rho * gm, rm)
    return float(cn)


def compute_cn_curve(r, gr, n_target, box, same=False, rmax_curve=6.0):
    mask = (r > 0.3) & (r <= rmax_curve)
    rm = r[mask]
    gm = gr[mask]

    if len(rm) < 2:
        return rm, np.zeros_like(rm)

    rho = (n_target - 1) / box ** 3 if same else n_target / box ** 3

    f = 4.0 * pi * rm ** 2 * rho * gm
    cn = np.zeros_like(rm)

    dr = np.diff(rm)
    avg_f = 0.5 * (f[1:] + f[:-1])
    cn[1:] = np.cumsum(avg_f * dr)

    return rm, cn


# ========================= Qn / NC =========================
def compute_qn_and_nc(coords, types, box):
    si_type = TYPE_MAP["Si"]
    p_type = TYPE_MAP["P"]
    o_type = TYPE_MAP["O"]

    si_o_cut = FIXED_CUTOFFS[("Si", "O")]
    p_o_cut = FIXED_CUTOFFS[("P", "O")]

    tree = cKDTree(coords, boxsize=box)

    o_idx = np.where(types == o_type)[0]

    o_nf_count = []
    o_nf_types = []

    search_cut = max(si_o_cut, p_o_cut)

    for o in o_idx:
        neigh = tree.query_ball_point(coords[o], search_cut)
        nf_types_local = []

        for j in neigh:
            if j == o:
                continue

            d = coords[o] - coords[j]
            d = minimum_image(d, box)
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

    def qn_for(center_type, center_cut):
        centers = np.where(types == center_type)[0]
        counts_qn = {n: 0 for n in range(5)}

        for c in centers:
            neigh = tree.query_ball_point(coords[c], center_cut)
            bcnt = 0

            for j in neigh:
                if j == c or types[j] != o_type:
                    continue

                d = coords[c] - coords[j]
                d = minimum_image(d, box)
                rij = np.linalg.norm(d)

                if rij < center_cut and j in bridging_set:
                    bcnt += 1

            bcnt = min(bcnt, 4)
            counts_qn[bcnt] += 1

        return counts_qn, len(centers)

    qn_si, n_si = qn_for(si_type, si_o_cut)
    qn_p, n_p = qn_for(p_type, p_o_cut)

    qn_combined = {
        n: qn_si[n] + qn_p[n]
        for n in range(5)
    }

    n_net = n_si + n_p

    nc_si = sum(n * qn_si[n] for n in range(5)) / max(1, n_si)
    nc_p = sum(n * qn_p[n] for n in range(5)) / max(1, n_p)
    nc_combined = sum(n * qn_combined[n] for n in range(5)) / max(1, n_net)

    return {
        "qn_si": qn_si,
        "qn_p": qn_p,
        "qn_combined": qn_combined,
        "n_si": n_si,
        "n_p": n_p,
        "n_net": n_net,
        "nc_si": nc_si,
        "nc_p": nc_p,
        "nc_combined": nc_combined,
        "o_speciation": {
            "FO": fo,
            "NBO": nbo,
            "BO": bo,
            "TO": to,
        },
        "bo_types": bo_types,
    }


# ========================= ENERGY LOG =========================
def load_energy_log(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Energy log not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    return df


def get_numeric_series(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float).values

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 0:
        return numeric.iloc[:, 0].values

    return None


def compute_block_stats(df, block_size):
    total = get_numeric_series(
        df,
        [
            "Total_eV_per_atom",
            "Total",
            "total",
            "Energy_eV_per_atom",
            "Total Energy (eV/atom)",
        ]
    )

    if total is None or len(total) == 0:
        return None, None

    n = len(total)

    if n < block_size:
        summary = {
            "block_size": block_size,
            "n_blocks": 0,
            "n_energy_points": n,
            "mean_eV_per_atom": float(np.mean(total)),
            "std_eV_per_atom": 0.0,
        }
        return summary, None

    n_blocks = n // block_size
    truncated = total[:n_blocks * block_size]
    blocks = truncated.reshape(n_blocks, block_size).mean(axis=1)

    if n_blocks > 1:
        std = float(np.std(blocks, ddof=1) / np.sqrt(n_blocks))
    else:
        std = 0.0

    summary = {
        "block_size": block_size,
        "n_blocks": n_blocks,
        "n_energy_points": n,
        "mean_eV_per_atom": float(np.mean(blocks)),
        "std_eV_per_atom": std,
    }

    block_df = pd.DataFrame({
        "block_number": np.arange(1, n_blocks + 1),
        "block_mean_eV_per_atom": blocks,
    })

    return summary, block_df


def export_energy_sheet(writer, df, max_rows):
    if df is None:
        return

    df_out = df.copy()

    if len(df_out) > max_rows:
        step = max(1, len(df_out) // max_rows)
        df_out = df_out.iloc[::step]

    df_out.to_excel(writer, sheet_name="Energy", index=False)


# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(
        description="Excel-only analyzer for Mg-doped 45S5 bioglass"
    )

    parser.add_argument(
        "input_xyz",
        type=Path,
        help="Path to final_structure.xyz"
    )

    parser.add_argument(
        "--energy-log",
        type=Path,
        default=None,
        help="Path to energy_log.csv"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results.xlsx"),
        help="Output Excel file"
    )

    parser.add_argument(
        "--rmax",
        type=float,
        default=8.0,
        help="Maximum radius for RDF"
    )

    parser.add_argument(
        "--nbins",
        type=int,
        default=800,
        help="Number of RDF bins"
    )

    parser.add_argument(
        "--cn-curve-rmax",
        type=float,
        default=6.0,
        help="Maximum radius for CN curve"
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=5000,
        help="Block size for block averaging"
    )

    parser.add_argument(
        "--max-energy-rows",
        type=int,
        default=20000,
        help="Maximum number of energy rows exported to Excel"
    )

    args = parser.parse_args()

    logger.info(f"Reading structure: {args.input_xyz}")
    struct = read_xyz(args.input_xyz)

    coords = struct["coords"]
    types = struct["types"]
    counts = struct["counts"]
    meta = struct["meta"]
    x_val = int(meta.get("x", 0))

    box = get_box(struct)

    total_mass = sum(
        MASSES[s] * counts.get(s, 0)
        for s in MASSES
    )
    eff_density = total_mass / (NA * box ** 3) * 1.0e24

    logger.info(f"System: 45-M{x_val}")
    logger.info(f"N atoms: {struct['n_atoms']}")
    logger.info(f"Box: {box:.6f} A")
    logger.info(f"Density: {eff_density:.6f} g/cm3")

    # Select only paper pairs that exist in this composition
    present_pairs = [
        p for p in PAPER_RDF_PAIRS
        if counts.get(p[0], 0) > 0 and counts.get(p[1], 0) > 0
    ]

    rdf_data = {}
    cn_curve_data = {}
    bond_rows = []
    cn_rows = []

    logger.info("Computing RDF and CN for paper pairs...")

    for pair in present_pairs:
        e1, e2 = pair
        pair_name = f"{e1}-{e2}"

        r, gr = compute_rdf_pair(
            coords,
            types,
            box,
            e1,
            e2,
            rmax=args.rmax,
            nbins=args.nbins
        )

        rdf_data[pair] = (r, gr)

        peak = find_first_peak(r, gr, pair)

        cut_fixed = FIXED_CUTOFFS[pair]
        n_target = counts.get(e2, 0)
        same = e1 == e2

        cn_fixed = integrate_cn(
            r,
            gr,
            cut_fixed,
            n_target,
            box,
            same=same
        )

        cut_auto = find_first_minimum(r, gr, pair)
        if not np.isfinite(cut_auto) or cut_auto <= 0.5:
            cut_auto = cut_fixed

        cn_auto = integrate_cn(
            r,
            gr,
            cut_auto,
            n_target,
            box,
            same=same
        )

        bond_rows.append({
            "Pair": pair_name,
            "Bond_length_A": peak,
            "Paper_Bond_length_A": PAPER_BOND_LENGTHS.get(pair, np.nan),
        })

        cn_rows.append({
            "Pair": pair_name,
            "Cutoff_fixed_A": cut_fixed,
            "CN_fixed": cn_fixed,
            "Cutoff_auto_A": cut_auto,
            "CN_auto": cn_auto,
            "Paper_CN": PAPER_CN.get(pair, np.nan),
        })

        r_cn, cn_curve = compute_cn_curve(
            r,
            gr,
            n_target,
            box,
            same=same,
            rmax_curve=args.cn_curve_rmax
        )

        cn_curve_data[pair] = (r_cn, cn_curve)

        logger.info(
            f"  {pair_name}: bond={peak:.3f} A, "
            f"CN_fixed={cn_fixed:.3f}, CN_auto={cn_auto:.3f}"
        )

    # Qn and NC
    logger.info("Computing Qn distributions and Network Connectivity...")
    qn_res = compute_qn_and_nc(coords, types, box)

    logger.info(f"  NC Si       : {qn_res['nc_si']:.4f}")
    logger.info(f"  NC P        : {qn_res['nc_p']:.4f}")
    logger.info(f"  NC Combined : {qn_res['nc_combined']:.4f}")

    # Energy log
    energy_df = None
    block_summary = None
    block_df = None

    if args.energy_log is not None:
        logger.info(f"Reading energy log: {args.energy_log}")
        energy_df = load_energy_log(args.energy_log)
        block_summary, block_df = compute_block_stats(
            energy_df,
            args.block_size
        )

        if block_summary is not None:
            logger.info(
                f"  Block average energy: "
                f"{block_summary['mean_eV_per_atom']:.6f} +/- "
                f"{block_summary['std_eV_per_atom']:.6f} eV/atom"
            )

    # Excel output
    output_path = Path(args.output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing Excel file: {output_path}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Metadata
        metadata_rows = [
            {"Parameter": "System", "Value": "Mg-doped 45S5 bioactive glass"},
            {"Parameter": "Reference", "Value": "Moghanian et al. manuscript"},
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
            {"Parameter": "RDF_rmax_A", "Value": args.rmax},
            {"Parameter": "RDF_nbins", "Value": args.nbins},
            {"Parameter": "NC_Si", "Value": qn_res["nc_si"]},
            {"Parameter": "NC_P", "Value": qn_res["nc_p"]},
            {"Parameter": "NC_Si_P_Combined", "Value": qn_res["nc_combined"]},
            {"Parameter": "Paper_NC_Combined", "Value": PAPER_NC.get(x_val, np.nan)},
        ]

        if block_summary is not None:
            metadata_rows.append({
                "Parameter": "Block_avg_energy_eV_per_atom",
                "Value": block_summary["mean_eV_per_atom"],
            })
            metadata_rows.append({
                "Parameter": "Block_avg_energy_std_eV_per_atom",
                "Value": block_summary["std_eV_per_atom"],
            })

        pd.DataFrame(metadata_rows).to_excel(
            writer,
            sheet_name="Metadata",
            index=False
        )

        # Bond lengths
        pd.DataFrame(bond_rows).to_excel(
            writer,
            sheet_name="Bond_Lengths",
            index=False
        )

        # CN summary
        pd.DataFrame(cn_rows).to_excel(
            writer,
            sheet_name="CN",
            index=False
        )

        # RDF and CN curve sheets
        for pair in present_pairs:
            e1, e2 = pair
            pair_name = f"{e1}-{e2}"

            r, gr = rdf_data[pair]
            pd.DataFrame({
                "r_A": r,
                "g_r": gr,
            }).to_excel(
                writer,
                sheet_name=f"RDF_{pair_name}",
                index=False
            )

            r_cn, cn_curve = cn_curve_data[pair]
            pd.DataFrame({
                "r_A": r_cn,
                "CN": cn_curve,
            }).to_excel(
                writer,
                sheet_name=f"CN_{pair_name}",
                index=False
            )

        # Qn Si
        n_si = max(1, qn_res["n_si"])
        qn_si_rows = []
        for n in range(5):
            qn_si_rows.append({
                "Qn": f"Q{n}",
                "Count": qn_res["qn_si"][n],
                "Percentage": qn_res["qn_si"][n] / n_si * 100.0,
            })

        pd.DataFrame(qn_si_rows).to_excel(
            writer,
            sheet_name="Qn_Si",
            index=False
        )

        # Qn P
        n_p = max(1, qn_res["n_p"])
        qn_p_rows = []
        for n in range(5):
            qn_p_rows.append({
                "Qn": f"Q{n}",
                "Count": qn_res["qn_p"][n],
                "Percentage": qn_res["qn_p"][n] / n_p * 100.0,
            })

        pd.DataFrame(qn_p_rows).to_excel(
            writer,
            sheet_name="Qn_P",
            index=False
        )

        # Qn Combined
        n_net = max(1, qn_res["n_net"])
        paper_qn_combined = PAPER_QN_COMBINED.get(x_val, {})

        qn_combined_rows = []
        for n in range(5):
            qn_combined_rows.append({
                "Qn": f"Q{n}",
                "Count": qn_res["qn_combined"][n],
                "Percentage": qn_res["qn_combined"][n] / n_net * 100.0,
                "Paper_Percentage": paper_qn_combined.get(n, np.nan),
            })

        pd.DataFrame(qn_combined_rows).to_excel(
            writer,
            sheet_name="Qn_Combined",
            index=False
        )

        # NC sheet
        nc_rows = [
            {
                "Type": "Si",
                "NC": qn_res["nc_si"],
                "Paper_NC": np.nan,
            },
            {
                "Type": "P",
                "NC": qn_res["nc_p"],
                "Paper_NC": np.nan,
            },
            {
                "Type": "Si-P Combined",
                "NC": qn_res["nc_combined"],
                "Paper_NC": PAPER_NC.get(x_val, np.nan),
            },
        ]

        pd.DataFrame(nc_rows).to_excel(
            writer,
            sheet_name="NC",
            index=False
        )

        # Oxygen speciation
        n_o = max(1, counts.get("O", 0))
        o_spec_rows = []
        for k, v in qn_res["o_speciation"].items():
            o_spec_rows.append({
                "Type": k,
                "Count": v,
                "Percentage": v / n_o * 100.0,
            })

        pd.DataFrame(o_spec_rows).to_excel(
            writer,
            sheet_name="O_Speciation",
            index=False
        )

        # BO types
        bo_type_rows = []
        for k, v in qn_res["bo_types"].items():
            bo_type_rows.append({
                "Type": k,
                "Count": v,
                "Percentage_of_total_O": v / n_o * 100.0,
            })

        pd.DataFrame(bo_type_rows).to_excel(
            writer,
            sheet_name="BO_Types",
            index=False
        )

        # Energy sheet
        if energy_df is not None:
            export_energy_sheet(writer, energy_df, args.max_energy_rows)

        # Block averaging sheets
        if block_summary is not None:
            pd.DataFrame([block_summary]).to_excel(
                writer,
                sheet_name="Block_Average",
                index=False
            )

        if block_df is not None:
            block_df.to_excel(
                writer,
                sheet_name="Block_Means",
                index=False
            )

    logger.info("[OK] Analysis completed.")
    logger.info(f"[OK] Excel file saved to: {output_path}")


if __name__ == "__main__":
    main()
    