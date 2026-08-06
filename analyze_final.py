#!/usr/bin/env python3
"""
========================================================================
Standalone Structural Analysis for 45S5/Sr Bioglass - FULL VERSION
Based on: Xiang & Du, Chem. Mater. 2011, 23, 2703-2717

Includes ALL analysis from Sr.py v6.6 + new features:
  - R_X_Si/P (cation preference: Si vs P)
  - Modifier Preference ratios (Sr vs Ca, Ca vs Na, Sr vs Na)
  - Sensitivity Analysis (energy vs cutoff/alpha)
  - Energy / Block Energy (reads energy_log.csv if available)
  - CN curves for all pairs
  - Neutron Structure Factor
  - Qn Combined, NC, O speciation, Clustering R_XX
  - Fnet, Si-O-P linkages

Reads:  final_structure.xyz
Writes: analysis_results.xlsx + plots/

Usage:
    python analyze_final.py Bioglass_x0_N11340_seed42/final_structure.xyz
    python analyze_final.py final_structure.xyz --output-dir my_analysis
    python analyze_final.py final_structure.xyz --energy-log Bioglass_x0_N11340_seed42/energy_log.csv
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
from math import pi, sqrt, erfc, exp
from collections import defaultdict
from numba import njit

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================= CONSTANTS =========================
NA = 6.02214076e23
KE = 14.3996454784255
KB_EV = 8.617333262145e-5
SWITCH_DR = 0.3

TYPE_MAP = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Sr': 5}
ELEM_MAP = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Sr'}
MASSES = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
          'P': 30.97376, 'O': 15.999, 'Sr': 87.62}
NEUTRON_B = {'O': 5.803, 'Si': 4.1491, 'Na': 3.63,
             'Ca': 4.70, 'Sr': 7.02, 'P': 5.13}
CHARGES_BASE = {'Si': 2.4, 'Ca': 1.2, 'Na': 0.6, 'P': 3.0, 'O': -1.2, 'Sr': 1.2}

# Xiang & Du 2011, Table 4 - Cutoff distances
CUTOFFS = {
    ('Si','O'):2.25, ('P','O'):2.25, ('Na','O'):3.34, ('Ca','O'):3.14,
    ('Sr','O'):3.35, ('O','O'):2.91,
    ('P','Ca'):4.44, ('P','Na'):4.44, ('P','Sr'):4.58,
    ('Si','Ca'):4.37, ('Si','Na'):4.42, ('Si','Sr'):4.51,
    ('Ca','Ca'):4.85, ('Na','Na'):4.15, ('Sr','Sr'):5.14,
    ('Ca','Na'):4.94, ('Ca','Sr'):5.00, ('Na','Sr'):4.40
}

PEAK_RANGES = {
    ('Si','O'):(1.40,2.00), ('P','O'):(1.30,1.90), ('Na','O'):(2.00,2.80),
    ('Ca','O'):(2.00,2.80), ('Sr','O'):(2.20,3.00), ('O','O'):(2.30,3.10),
    ('Si','Si'):(2.80,3.60), ('Si','P'):(2.70,3.40), ('P','P'):(2.70,3.40),
    ('Si','Na'):(2.90,3.70), ('Si','Ca'):(3.20,4.10), ('Si','Sr'):(3.20,4.10),
    ('Na','Na'):(2.50,3.50), ('Ca','Ca'):(3.20,4.20), ('Sr','Sr'):(3.50,4.50),
    ('Ca','Na'):(3.00,4.00), ('Ca','Sr'):(3.20,4.20), ('Na','Sr'):(3.00,4.00),
    ('P','Ca'):(3.20,4.20), ('P','Na'):(3.00,4.00), ('P','Sr'):(3.20,4.20),
}

MINIMUM_RMIN = {
    ('Si','O'):1.4, ('P','O'):1.3, ('Na','O'):2.0, ('Ca','O'):2.0,
    ('Sr','O'):2.2, ('O','O'):2.2,
    ('P','Ca'):3.5, ('P','Na'):3.5, ('P','Sr'):3.5,
    ('Si','Ca'):3.5, ('Si','Na'):3.0, ('Si','Sr'):3.5,
    ('Ca','Ca'):3.5, ('Na','Na'):3.0, ('Sr','Sr'):4.0,
    ('Ca','Na'):3.5, ('Ca','Sr'):4.0, ('Na','Sr'):3.5,
}

# Paper reference values
PAPER_BOND_LENGTHS = {
    ('O','O'):2.63, ('Si','O'):1.61, ('Ca','O'):2.38, ('Na','O'):2.40,
    ('Sr','O'):2.59, ('Si','Si'):3.16, ('Si','Na'):3.29, ('Si','Ca'):3.64
}
PAPER_CN = {'Ca': 6.2, 'Na': 5.7, 'Sr': 7.0}
PAPER_QN_SI = {'Q0':3.7, 'Q1':23.2, 'Q2':39.8, 'Q3':27.8, 'Q4':5.6}
PAPER_QN_P = {'Q0':50.0, 'Q1':42.9, 'Q2':6.4, 'Q3':0.6, 'Q4':0.0}
PAPER_NC = {'Si': 2.07, 'P': 0.62, 'overall': 1.92}

# Buckingham parameters (Table 2) for sensitivity analysis
BUCK_PARAMS = {
    ('Si','O'): {'A': 13702.905, 'F': 0.193817, 'C': 54.681},
    ('P','O'):  {'A': 26655.472, 'F': 0.181968, 'C': 86.856},
    ('O','O'):  {'A': 2029.2204, 'F': 0.343645, 'C': 192.58},
    ('Na','O'): {'A': 4383.7555, 'F': 0.243838, 'C': 30.70},
    ('Ca','O'): {'A': 7747.1834, 'F': 0.252623, 'C': 93.109},
    ('Sr','O'): {'A': 14566.637, 'F': 0.245015, 'C': 81.773},
}
ZBL_Z = {'Si': 14.0, 'Ca': 20.0, 'Na': 11.0, 'P': 15.0, 'O': 8.0, 'Sr': 38.0}
ZBL_A0 = 0.46850
ZBL_C = np.array([0.1818, 0.5099, 0.2802, 0.02817])
ZBL_D = np.array([3.2, 0.9423, 0.4029, 0.2016])


# ========================= ENERGY ROUTINES (for Sensitivity) =========================
@njit(fastmath=True, cache=True)
def _zbl_repulsion(r, Zi, Zj):
    if r < 1e-6: return 1e8
    a = ZBL_A0 / (Zi**0.23 + Zj**0.23)
    x = r / a
    phi = (ZBL_C[0]*exp(-ZBL_D[0]*x) + ZBL_C[1]*exp(-ZBL_D[1]*x) +
           ZBL_C[2]*exp(-ZBL_D[2]*x) + ZBL_C[3]*exp(-ZBL_D[3]*x))
    return KE * Zi * Zj * phi / r


@njit(fastmath=True, cache=True)
def _wolf_coulomb(qi, qj, r, alpha, cutoff):
    if r >= cutoff or r < 1e-12: return 0.0
    ar = alpha * r
    ac = alpha * cutoff
    er = erfc(ar); ec = erfc(ac)
    t1 = er / r
    t2 = ec / cutoff
    t3 = ((ec/(cutoff*cutoff)) + (2.0*alpha/sqrt(pi))*exp(-ac*ac)/cutoff) * (r - cutoff)
    return KE * qi * qj * (t1 - t2 + t3)


def build_energy_matrices():
    nt = 6
    A_mat = np.zeros((nt, nt)); F_mat = np.zeros((nt, nt)); C_mat = np.zeros((nt, nt))
    R_HARD = np.full((nt, nt), 0.9); R_HARD[4, 4] = 1.4
    for (e1, e2), p in BUCK_PARAMS.items():
        t1 = TYPE_MAP[e1]; t2 = TYPE_MAP[e2]
        A_mat[t1, t2] = p['A']; F_mat[t1, t2] = p['F']; C_mat[t1, t2] = p['C']
        if t1 != t2:
            A_mat[t2, t1] = p['A']; F_mat[t2, t1] = p['F']; C_mat[t2, t1] = p['C']
    type_Z = np.array([ZBL_Z[ELEM_MAP[i]] for i in range(6)])
    return A_mat, F_mat, C_mat, R_HARD, type_Z


def compute_total_energy(coords, types, charges, box, cutoff, alpha):
    """Compute total energy for sensitivity analysis."""
    A_mat, F_mat, C_mat, R_HARD, type_Z = build_energy_matrices()
    tree = cKDTree(coords, boxsize=box)
    pairs = tree.query_pairs(cutoff + 0.5, output_type='ndarray')
    e_total = 0.0
    for i, j in pairs:
        dx = coords[i,0]-coords[j,0]; dy = coords[i,1]-coords[j,1]; dz = coords[i,2]-coords[j,2]
        dx -= box*np.round(dx/box); dy -= box*np.round(dy/box); dz -= box*np.round(dz/box)
        r = sqrt(dx*dx+dy*dy+dz*dz)
        if r >= cutoff or r < 1e-12: continue
        ti = types[i]; tj = types[j]
        qi = charges[i]; qj = charges[j]
        rh = R_HARD[ti, tj]
        if r < rh:
            e_total += _zbl_repulsion(r, type_Z[ti], type_Z[tj])
        else:
            A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
            e_buck = 0.0
            if A > 0 and F > 1e-12: e_buck += A*exp(-r/F)
            if C > 0: e_buck -= C/(r**6)
            e_coul = _wolf_coulomb(qi, qj, r, alpha, cutoff)
            if r < rh + SWITCH_DR:
                e_zbl = _zbl_repulsion(r, type_Z[ti], type_Z[tj])
                x_s = (r - rh) / SWITCH_DR
                s = x_s**3*(10.0-15.0*x_s+6.0*x_s**2)
                e_total += s*(e_buck+e_coul) + (1.0-s)*e_zbl
            else:
                e_total += e_buck + e_coul
    wolf_self = -KE*(alpha/sqrt(pi))*np.sum(charges**2)
    return e_total + wolf_self


def sensitivity_analysis(coords, types, charges, box, n_atoms):
    """Energy sensitivity vs cutoff and Wolf alpha."""
    logger.info("=" * 60)
    logger.info("SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    test_cutoffs = [8.0, 10.0, 12.0]
    test_alphas = [0.20, 0.25, 0.30]
    results = []
    for cut in test_cutoffs:
        for alpha in test_alphas:
            U = compute_total_energy(coords, types, charges, box, cut, alpha)
            U_tot = U / n_atoms
            results.append({'Cutoff': cut, 'Alpha': alpha, 'Energy': U_tot})
            logger.info(f"  Cutoff={cut:.1f} A, alpha={alpha:.2f}: E={U_tot:.6f} eV/atom")
    return results


# ========================= STRUCTURE READER =========================
def read_xyz(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    meta = {'x': 0, 'rho': None, 'box': None, 'seed': None}
    for key, pat, conv in [
        ('x', r'x\s*=\s*(\d+)', int),
        ('rho', r'rho\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('box', r'box\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('seed', r'seed\s*=\s*(\d+)', int),
    ]:
        m = re.search(pat, comment)
        if m: meta[key] = conv(m.group(1))
    symbols, coords = [], []
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords, dtype=np.float64)
    types = np.array([TYPE_MAP[s] for s in symbols], dtype=np.int32)
    counts = {e: int(np.sum(types == TYPE_MAP[e])) for e in set(symbols)}
    return {'symbols': symbols, 'coords': coords, 'types': types,
            'n_atoms': n_atoms, 'counts': counts, 'meta': meta}


def minimum_image(dr, box):
    return dr - box * np.round(dr / box)


# ========================= RDF & CN =========================
def compute_rdf(coords, types, ti, tj, box, rmax=8.0, nbins=800):
    mi = types == ti; mj = types == tj
    ni = int(mi.sum()); nj = int(mj.sum())
    if ni == 0 or nj == 0 or (ti == tj and ni < 2):
        return np.linspace(0, rmax, nbins), np.zeros(nbins)
    idx_i = np.where(mi)[0]
    hist = np.zeros(nbins)
    dr_bin = rmax / nbins
    tree = cKDTree(coords, boxsize=box)
    for idx in idx_i:
        neigh = tree.query_ball_point(coords[idx], rmax)
        for j in neigh:
            if j == idx or types[j] != tj: continue
            d = minimum_image(coords[idx] - coords[j], box)
            r = np.linalg.norm(d)
            if 0 < r < rmax:
                b = int(r / dr_bin)
                if b < nbins: hist[b] += 1
    r_edges = np.linspace(0, rmax, nbins + 1)
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    vol_shell = (4.0/3.0) * pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    V = box**3
    rho = (ni - 1) / V if ti == tj else nj / V
    gr = hist / (ni * rho * vol_shell)
    return rc, gr


def find_first_peak(r, gr, pair):
    key = pair if pair in PEAK_RANGES else (pair[1], pair[0])
    rmin, rmax = PEAK_RANGES.get(key, (1.0, 4.0))
    mask = (r >= rmin) & (r <= rmax)
    if not mask.any(): return np.nan
    rm, gm = r[mask], gr[mask]
    gm_smooth = gaussian_filter1d(gm, sigma=1.0)
    return rm[np.argmax(gm_smooth)]


def find_first_minimum(r, gr, pair):
    peak_r = find_first_peak(r, gr, pair)
    if np.isnan(peak_r): return np.nan
    key = pair if pair in MINIMUM_RMIN else (pair[1], pair[0])
    rmin_search = MINIMUM_RMIN.get(key, peak_r + 0.2)
    rmax_search = rmin_search + 1.5
    mask = (r > rmin_search) & (r <= rmax_search)
    if not mask.any(): return np.nan
    rm, gm = r[mask], gr[mask]
    gm_smooth = gaussian_filter1d(gm, sigma=1.5)
    return rm[np.argmin(gm_smooth)]


def coordination_number(r, gr, cutoff, n_target, box):
    mask = (r > 0.3) & (r <= cutoff)
    if not mask.any(): return 0.0
    rm, gm = r[mask], gr[mask]
    rho = n_target / box**3
    return trapezoid(4 * pi * rm**2 * rho * gm, rm)


# ========================= ANGLES =========================
def compute_angles(coords, types, box, central, ligand, cutoff, mode='X-Y-X'):
    tree = cKDTree(coords, boxsize=box)
    angles = []
    if mode == 'X-Y-X':
        central_idx = np.where(types == central)[0]
        for c in central_idx:
            neigh = tree.query_ball_point(coords[c], cutoff)
            valid = [j for j in neigh if j != c and types[j] == ligand
                     and np.linalg.norm(minimum_image(coords[c]-coords[j], box)) < cutoff]
            for a in range(len(valid)):
                for b in range(a+1, len(valid)):
                    v1 = minimum_image(coords[valid[a]]-coords[c], box)
                    v2 = minimum_image(coords[valid[b]]-coords[c], box)
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-6 and n2 > 1e-6:
                        cosang = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
                        angles.append(np.degrees(np.arccos(cosang)))
    elif mode == 'Y-X-Y':
        bridge_idx = np.where(types == central)[0]
        for c in bridge_idx:
            neigh = tree.query_ball_point(coords[c], cutoff)
            lig_atoms = [j for j in neigh if j != c and types[j] == ligand
                         and np.linalg.norm(minimum_image(coords[c]-coords[j], box)) < cutoff]
            if len(lig_atoms) >= 2:
                for a in range(len(lig_atoms)):
                    for b in range(a+1, len(lig_atoms)):
                        v1 = minimum_image(coords[lig_atoms[a]]-coords[c], box)
                        v2 = minimum_image(coords[lig_atoms[b]]-coords[c], box)
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 1e-6 and n2 > 1e-6:
                            cosang = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
                            angles.append(np.degrees(np.arccos(cosang)))
    return np.array(angles)


# ========================= Qn & O SPECIATION =========================
def compute_qn_and_speciation(coords, types, box):
    si_o_cut = CUTOFFS[('Si','O')]
    p_o_cut = CUTOFFS[('P','O')]
    tree = cKDTree(coords, boxsize=box)
    si_idx = np.where(types == TYPE_MAP['Si'])[0]
    p_idx = np.where(types == TYPE_MAP['P'])[0]
    o_idx = np.where(types == TYPE_MAP['O'])[0]

    o_nf_count = np.zeros(len(coords), dtype=np.int32)
    for o in o_idx:
        neigh = tree.query_ball_point(coords[o], max(si_o_cut, p_o_cut))
        cnt = 0
        for j in neigh:
            if j == o: continue
            d = np.linalg.norm(minimum_image(coords[o]-coords[j], box))
            if types[j] == TYPE_MAP['Si'] and d < si_o_cut: cnt += 1
            elif types[j] == TYPE_MAP['P'] and d < p_o_cut: cnt += 1
        o_nf_count[o] = cnt

    fo = int(np.sum(o_nf_count[o_idx] == 0))
    nbo = int(np.sum(o_nf_count[o_idx] == 1))
    bo = int(np.sum(o_nf_count[o_idx] == 2))
    to = int(np.sum(o_nf_count[o_idx] >= 3))
    bridging = set(o_idx[o_nf_count[o_idx] >= 2])

    def qn_for(center_idx, center_cut):
        qn = []
        for c in center_idx:
            neigh = tree.query_ball_point(coords[c], center_cut)
            bcnt = sum(1 for j in neigh if j != c and types[j] == TYPE_MAP['O']
                       and np.linalg.norm(minimum_image(coords[c]-coords[j], box)) < center_cut
                       and j in bridging)
            qn.append(bcnt)
        qn = np.array(qn)
        return {n: int(np.sum(qn == n)) for n in range(5)}

    return qn_for(si_idx, si_o_cut), qn_for(p_idx, p_o_cut), {'FO': fo, 'NBO': nbo, 'BO': bo, 'TO': to}


def network_connectivity(qn_dist, total):
    if total == 0: return 0.0
    return sum(qn_dist.get(n, 0) * n for n in range(5)) / total


# ========================= CLUSTERING =========================
def compute_rxx(coords, types, box, elem):
    ti = TYPE_MAP[elem]
    pair = (elem, elem)
    cutoff = CUTOFFS.get(pair)
    if cutoff is None: return np.nan, np.nan, np.nan
    r, gr = compute_rdf(coords, types, ti, ti, box, rmax=cutoff+2.0, nbins=600)
    n_elem = int(np.sum(types == ti))
    cn_obs = coordination_number(r, gr, cutoff, n_elem, box)
    rho = n_elem / box**3
    cn_hom = (4.0/3.0) * pi * cutoff**3 * rho
    rxx = cn_obs / cn_hom if cn_hom > 0 else np.nan
    return rxx, cn_obs, cn_hom


def compute_preference(coords, types, box, A, B, C):
    ta, tb, tc = TYPE_MAP[A], TYPE_MAP[B], TYPE_MAP[C]
    nb = int(np.sum(types == tb)); nc_count = int(np.sum(types == tc))
    if nb == 0 or nc_count == 0: return np.nan
    def cn_around(around_type, neighbor_type, cutoff):
        r, gr = compute_rdf(coords, types, around_type, neighbor_type,
                            box, rmax=cutoff+2.0, nbins=600)
        n_n = int(np.sum(types == neighbor_type))
        return coordination_number(r, gr, cutoff, n_n, box)
    pair_ab = (A,B) if (A,B) in CUTOFFS else (B,A)
    pair_ac = (A,C) if (A,C) in CUTOFFS else (C,A)
    cn_ab = cn_around(ta, tb, CUTOFFS.get(pair_ab, 4.5))
    cn_ac = cn_around(ta, tc, CUTOFFS.get(pair_ac, 4.5))
    if cn_ac == 0: return np.nan
    return (cn_ab / cn_ac) * (nc_count / nb)


def compute_r_x_si_p(coords, types, box, counts):
    """R_X_Si/P: ratio of modifier CN around Si vs P (from Sr.py v6.6)."""
    n_si = counts.get('Si', 0)
    n_p = counts.get('P', 0)
    if n_si == 0 or n_p == 0:
        return {e: 0.0 for e in ['Ca', 'Na', 'Sr']}
    result = {}
    for elem in ['Ca', 'Na', 'Sr']:
        if counts.get(elem, 0) == 0:
            result[elem] = 0.0
            continue
        pair_se = ('Si', elem) if ('Si', elem) in CUTOFFS else (elem, 'Si')
        pair_pe = ('P', elem) if ('P', elem) in CUTOFFS else (elem, 'P')
        cut_se = CUTOFFS.get(pair_se, 4.5)
        cut_pe = CUTOFFS.get(pair_pe, 4.5)
        n_elem = counts.get(elem, 0)
        r_se, gr_se = compute_rdf(coords, types, TYPE_MAP['Si'], TYPE_MAP[elem],
                                   box, rmax=cut_se+2.0, nbins=600)
        r_pe, gr_pe = compute_rdf(coords, types, TYPE_MAP['P'], TYPE_MAP[elem],
                                   box, rmax=cut_pe+2.0, nbins=600)
        cn_si = coordination_number(r_se, gr_se, cut_se, n_elem, box)
        cn_p = coordination_number(r_pe, gr_pe, cut_pe, n_elem, box)
        result[elem] = (cn_si / cn_p) * (n_p / n_si) if cn_p != 0 else 0.0
    return result


def cn_modifiers_around_nf(coords, types, box, nf_elem):
    ta = TYPE_MAP[nf_elem]
    results = {}
    for mod in ['Na', 'Ca', 'Sr']:
        if mod not in [ELEM_MAP[t] for t in np.unique(types)]: continue
        tm = TYPE_MAP[mod]
        pair = (nf_elem, mod) if (nf_elem, mod) in CUTOFFS else (mod, nf_elem)
        cutoff = CUTOFFS.get(pair, 4.5)
        r, gr = compute_rdf(coords, types, ta, tm, box, rmax=cutoff+2.0, nbins=600)
        n_mod = int(np.sum(types == tm))
        results[mod] = coordination_number(r, gr, cutoff, n_mod, box)
    return results


def modifier_cn_distribution(coords, types, box, elem, cutoff):
    ti = TYPE_MAP[elem]
    o_type = TYPE_MAP['O']
    idx = np.where(types == ti)[0]
    tree = cKDTree(coords, boxsize=box)
    cn_list = []
    for c in idx:
        neigh = tree.query_ball_point(coords[c], cutoff)
        cnt = sum(1 for j in neigh if j != c and types[j] == o_type
                  and np.linalg.norm(minimum_image(coords[c]-coords[j], box)) < cutoff)
        cn_list.append(cnt)
    return np.array(cn_list)


# ========================= NEUTRON STRUCTURE FACTOR =========================
def neutron_structure_factor(coords, types, symbols, box, qmax=25.0, nq=500):
    elements = list(set(symbols))
    n_atoms = len(symbols)
    counts = {e: int(np.sum(np.array(symbols) == e)) for e in elements}
    c_frac = {e: counts[e]/n_atoms for e in elements}
    q = np.linspace(0.5, qmax, nq)
    rmax = box / 2.0
    nr = 800
    total_S = np.zeros_like(q)
    sum_cb_sq = (sum(c_frac[e]*NEUTRON_B[e] for e in elements))**2
    for i, ei in enumerate(elements):
        for j, ej in enumerate(elements):
            if j < i: continue
            ti, tj = TYPE_MAP[ei], TYPE_MAP[ej]
            rc, gr = compute_rdf(coords, types, ti, tj, box, rmax=rmax, nbins=nr)
            window = np.sinc(rc / rmax)
            coeff = 4 * pi * (n_atoms / box**3)
            Sij = np.ones_like(q)
            integrand_base = rc**2 * (gr - 1) * window
            for kq, qv in enumerate(q):
                sinc = np.sin(qv * rc) / (qv * rc)
                Sij[kq] += coeff * trapezoid(integrand_base * sinc, rc)
            weight = c_frac[ei] * c_frac[ej] * NEUTRON_B[ei] * NEUTRON_B[ej]
            if ei != ej: weight *= 2
            total_S += weight * Sij
    total_S /= sum_cb_sq
    return q, total_S


# ========================= MAIN ANALYSIS =========================
def run_analysis(xyz_path, output_dir=None, energy_log_path=None):
    xyz_path = Path(xyz_path)
    if not xyz_path.exists():
        logger.error(f"File not found: {xyz_path}")
        sys.exit(1)

    logger.info(f"Reading structure: {xyz_path}")
    struct = read_xyz(xyz_path)
    coords, types = struct['coords'], struct['types']
    counts = struct['counts']
    x_val = struct['meta']['x']

    if struct['meta']['box'] is not None:
        box = struct['meta']['box']
    else:
        total_mass = sum(MASSES[s] for s in struct['symbols'])
        rho = struct['meta']['rho'] or 2.649
        box = ((total_mass / NA) / rho * 1e24) ** (1/3)

    if output_dir is None:
        output_dir = Path(f"analysis_x{x_val}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info(f"Box = {box:.4f} A, N = {struct['n_atoms']}, x = {x_val}")
    logger.info(f"Composition: {counts}")

    # Compute charges
    charges = np.array([CHARGES_BASE[s] for s in struct['symbols']], dtype=np.float64)
    total_pos = sum(counts.get(e, 0)*CHARGES_BASE.get(e, 0) for e in counts if e != 'O')
    n_o = counts.get('O', 0)
    if n_o > 0:
        charges[types == TYPE_MAP['O']] = -total_pos / n_o

    present = [e for e in ['Si', 'P', 'Na', 'Ca', 'Sr', 'O'] if counts.get(e, 0) > 0]
    results = {}

    # ---- Density & Molar Volume ----
    total_mass = sum(MASSES[s] for s in struct['symbols'])
    eff_density = total_mass / (NA * box**3) * 1e24
    n_oxide_units = struct['n_atoms'] / 2.835
    molar_volume = (total_mass / n_oxide_units) / eff_density
    results['metadata'] = {
        'x_mol_percent_SrO': x_val, 'N_atoms': struct['n_atoms'], 'Box_A': box,
        'Density_g_cm3': eff_density, 'Molar_volume_cm3_mol': molar_volume,
        **{f'N_{e}': counts.get(e, 0) for e in ['Si', 'P', 'Na', 'Ca', 'Sr', 'O']}
    }

    # ---- RDFs, bond lengths, CN ----
    logger.info("Computing RDFs, bond lengths, coordination numbers...")
    rdf_data = {}; bond_lengths = {}; cn_fixed = {}; cn_auto = {}; cn_extra = {}
    si_o_cut_extra = [2.15, 2.20, 2.25, 2.30, 2.35]

    for e1 in present:
        for e2 in present:
            if TYPE_MAP[e1] > TYPE_MAP[e2]: continue
            ti, tj = TYPE_MAP[e1], TYPE_MAP[e2]
            r, gr = compute_rdf(coords, types, ti, tj, box, rmax=8.0, nbins=800)
            rdf_data[(e1, e2)] = (r, gr)
            bond_lengths[(e1, e2)] = find_first_peak(r, gr, (e1, e2))
            pair_key = (e1, e2) if (e1, e2) in CUTOFFS else (e2, e1)
            if pair_key in CUTOFFS:
                cutoff = CUTOFFS[pair_key]
                n_target = counts.get(e2, 0)
                cn_fixed[(e1, e2)] = coordination_number(r, gr, cutoff, n_target, box)
                first_min = find_first_minimum(r, gr, (e1, e2))
                if not np.isnan(first_min):
                    cn_auto[(e1, e2)] = coordination_number(r, gr, first_min, n_target, box)
            if (e1, e2) == ('Si', 'O'):
                cn_extra[('Si', 'O')] = {}
                for rcut in si_o_cut_extra:
                    mask_cn = (r > 0.3) & (r <= rcut)
                    if mask_cn.any():
                        n_target = counts.get('O', 0)
                        cn_extra[('Si', 'O')][rcut] = coordination_number(r, gr, rcut, n_target, box)
                    else:
                        cn_extra[('Si', 'O')][rcut] = 0.0

    results['bond_lengths'] = bond_lengths
    results['cn_fixed'] = cn_fixed
    results['cn_auto'] = cn_auto
    results['cn_extra'] = cn_extra

    # ---- Bond Angles ----
    logger.info("Computing bond angle distributions...")
    angles = {}
    angles['O-Si-O'] = compute_angles(coords, types, box, TYPE_MAP['Si'], TYPE_MAP['O'],
                                      CUTOFFS[('Si','O')], mode='X-Y-X')
    angles['O-P-O'] = compute_angles(coords, types, box, TYPE_MAP['P'], TYPE_MAP['O'],
                                     CUTOFFS[('P','O')], mode='X-Y-X')
    angles['Si-O-Si'] = compute_angles(coords, types, box, TYPE_MAP['O'], TYPE_MAP['Si'],
                                       CUTOFFS[('Si','O')], mode='Y-X-Y')
    angles['Si-O-P'] = compute_angles(coords, types, box, TYPE_MAP['O'], TYPE_MAP['P'],
                                      CUTOFFS[('P','O')], mode='Y-X-Y')
    for mod in ['Na', 'Ca', 'Sr']:
        if counts.get(mod, 0) > 0:
            angles[f'O-{mod}-O'] = compute_angles(coords, types, box, TYPE_MAP[mod],
                                                  TYPE_MAP['O'], CUTOFFS[(mod, 'O')], mode='X-Y-X')
    results['angles'] = {k: (float(np.mean(v)) if len(v) > 0 else np.nan) for k, v in angles.items()}

    # ---- Qn + Oxygen Speciation ----
    logger.info("Computing Qn distributions and oxygen speciation...")
    qn_si, qn_p, o_spec = compute_qn_and_speciation(coords, types, box)
    n_si, n_p = counts.get('Si', 0), counts.get('P', 0)
    qn_si_pct = {f'Q{n}': qn_si.get(n, 0)/n_si*100 if n_si > 0 else 0 for n in range(5)}
    qn_p_pct = {f'Q{n}': qn_p.get(n, 0)/n_p*100 if n_p > 0 else 0 for n in range(5)}
    nc_si = network_connectivity(qn_si, n_si)
    nc_p = network_connectivity(qn_p, n_p)
    n_net = n_si + n_p
    qn_combined = {n: qn_si.get(n, 0) + qn_p.get(n, 0) for n in range(5)}
    nc_overall = network_connectivity(qn_combined, n_net)
    qn_c_pct = {f'Q{n}': qn_combined[n]/n_net*100 if n_net > 0 else 0 for n in range(5)}

    results['qn_si'] = qn_si_pct
    results['qn_p'] = qn_p_pct
    results['qn_combined'] = qn_c_pct
    results['nc'] = {'Si': nc_si, 'P': nc_p, 'overall': nc_overall}
    results['oxygen_speciation'] = o_spec

    # ---- Modifier CN ----
    logger.info("Computing modifier coordination numbers...")
    modifier_cn = {}; modifier_cn_dist = {}
    for mod in ['Na', 'Ca', 'Sr']:
        if counts.get(mod, 0) > 0:
            pair = (mod, 'O')
            cutoff = CUTOFFS[pair]
            ti, tj = TYPE_MAP[mod], TYPE_MAP['O']
            r, gr = compute_rdf(coords, types, ti, tj, box, rmax=cutoff+2.0)
            modifier_cn[mod] = coordination_number(r, gr, cutoff, counts.get('O', 0), box)
            modifier_cn_dist[mod] = modifier_cn_distribution(coords, types, box, mod, cutoff)
    results['modifier_cn'] = modifier_cn

    # ---- Clustering R_XX ----
    logger.info("Computing clustering ratios R_XX...")
    rxx = {}
    for mod in ['Na', 'Ca', 'Sr']:
        if counts.get(mod, 0) > 0:
            rxx[mod] = compute_rxx(coords, types, box, mod)
    results['rxx'] = rxx

    # ---- Modifier Preference (Sr vs Ca, Ca vs Na, Sr vs Na) ----
    logger.info("Computing modifier preference ratios...")
    preference = {}
    for A in ['Si', 'P']:
        if counts.get(A, 0) == 0: continue
        for B, C in [('Sr', 'Ca'), ('Ca', 'Na'), ('Sr', 'Na')]:
            if counts.get(B, 0) > 0 and counts.get(C, 0) > 0:
                preference[f'{A}_{B}_vs_{C}'] = compute_preference(coords, types, box, A, B, C)
    results['preference'] = preference

    # ---- R_X_Si/P (from Sr.py v6.6) ----
    logger.info("Computing R_X_Si/P ratios...")
    r_x_si_p = compute_r_x_si_p(coords, types, box, counts)
    results['r_x_si_p'] = r_x_si_p

    # ---- CN of modifiers around NF ----
    logger.info("Computing CN of modifiers around network formers...")
    cn_around = {}
    for nf in ['Si', 'P']:
        if counts.get(nf, 0) > 0:
            cn_around[nf] = cn_modifiers_around_nf(coords, types, box, nf)
    results['cn_around_nf'] = cn_around

    # ---- Fnet ----
    logger.info("Computing Fnet...")
    sbs_x_o = {'Ca': 32.0, 'Na': 20.0, 'Sr': 32.0}
    fnet = 0.0
    if n_net > 0:
        s = 0.0
        for elem in ['Ca', 'Na', 'Sr']:
            Cx = counts.get(elem, 0)
            if Cx == 0: continue
            nv = {'Ca': 2, 'Na': 1, 'Sr': 2}[elem]
            sbs = sbs_x_o[elem]
            pair = (elem, 'O')
            cutoff = CUTOFFS[pair]
            r, gr = rdf_data.get(pair, rdf_data.get(('O', elem), (None, None)))
            if r is None:
                ti, tj = TYPE_MAP[elem], TYPE_MAP['O']
                r, gr = compute_rdf(coords, types, ti, tj, box, rmax=cutoff+2.0)
            cn_o = coordination_number(r, gr, cutoff, counts.get('O', 0), box)
            s += Cx * nv * sbs * cn_o * nc_overall
        fnet = s / n_net
    results['fnet'] = fnet

    # ---- Si-O-P linkages ----
    logger.info("Computing Si-O-P linkages...")
    si_o_cut = CUTOFFS[('Si','O')]
    p_o_cut = CUTOFFS[('P','O')]
    tree = cKDTree(coords, boxsize=box)
    p_idx = np.where(types == TYPE_MAP['P'])[0]
    p_si_links = 0
    total_p_o = 0
    if len(p_idx) > 0:
        for p in p_idx:
            neigh = tree.query_ball_point(coords[p], p_o_cut)
            o_n = [j for j in neigh if j != p and types[j] == TYPE_MAP['O']
                   and np.linalg.norm(minimum_image(coords[p]-coords[j], box)) < p_o_cut]
            total_p_o += len(o_n)
            for o in o_n:
                o_neigh = tree.query_ball_point(coords[o], si_o_cut)
                for j2 in o_neigh:
                    if j2 != o and types[j2] == TYPE_MAP['Si']:
                        if np.linalg.norm(minimum_image(coords[o]-coords[j2], box)) < si_o_cut:
                            p_si_links += 1
                            break
    results['p_si_links'] = p_si_links
    results['total_p_o_bonds'] = total_p_o
    results['frac_p_si'] = p_si_links / total_p_o if total_p_o > 0 else 0.0

    # ---- Sensitivity Analysis ----
    logger.info("Computing sensitivity analysis...")
    results['sensitivity'] = sensitivity_analysis(coords, types, charges, box, struct['n_atoms'])

    # ---- Energy Log (if available) ----
    energy_data = None
    if energy_log_path is not None and Path(energy_log_path).exists():
        logger.info(f"Reading energy log: {energy_log_path}")
        try:
            df_e = pd.read_csv(energy_log_path)
            energy_data = {
                'total': df_e['Total_eV_per_atom'].tolist(),
                'short': df_e['ShortRange_eV_per_atom'].tolist(),
                'coul': df_e['Coulomb_eV_per_atom'].tolist(),
            }
            # Block averaging
            block_size = 5000
            prod_energy = np.array(energy_data['total'])
            n_blocks = len(prod_energy) // block_size
            if n_blocks >= 3:
                block_avgs = [np.mean(prod_energy[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
                results['block_energy'] = {
                    'mean': float(np.mean(block_avgs)),
                    'std': float(np.std(block_avgs, ddof=1) / np.sqrt(n_blocks))
                }
            else:
                results['block_energy'] = {'mean': float(np.mean(prod_energy)), 'std': 0.0}
        except Exception as e:
            logger.warning(f"Could not read energy log: {e}")
    results['energy_data'] = energy_data

    # ---- Neutron SF ----
    logger.info("Computing neutron structure factor...")
    q, sn_q = neutron_structure_factor(coords, types, struct['symbols'], box, qmax=25.0, nq=500)
    results['neutron_sf'] = (q, sn_q)

    # ---- Export ----
    logger.info("Exporting results...")
    export_excel(results, rdf_data, angles, modifier_cn_dist, output_dir, x_val, si_o_cut_extra)
    make_plots(results, rdf_data, angles, modifier_cn_dist, q, sn_q, plots_dir)
    make_cn_curves(rdf_data, cn_fixed, box, counts, CUTOFFS, plots_dir)
    print_summary(results, x_val)
    logger.info(f"\n[OK] All results saved to: {output_dir}")


# ========================= EXCEL EXPORT =========================
def export_excel(results, rdf_data, angles, modifier_cn_dist, output_dir, x_val, si_o_cut_extra):
    path = output_dir / "analysis_results.xlsx"
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Metadata
        md = results['metadata']
        pd.DataFrame([md.items()], columns=[k for k in md.keys()]).T.to_excel(
            writer, sheet_name='Metadata', header=['Value'])

        # Bond Lengths
        bl_rows = []
        for (e1, e2), bl in results['bond_lengths'].items():
            paper = PAPER_BOND_LENGTHS.get((e1, e2), PAPER_BOND_LENGTHS.get((e2, e1), np.nan))
            bl_rows.append({'Pair': f'{e1}-{e2}', 'Bond_length_A': bl, 'Paper_value_A': paper})
        pd.DataFrame(bl_rows).to_excel(writer, sheet_name='Bond_Lengths', index=False)

        # CN
        cn_rows = []
        for (e1, e2), cn in results['cn_fixed'].items():
            pair_key = (e1, e2) if (e1, e2) in CUTOFFS else (e2, e1)
            cutoff = CUTOFFS.get(pair_key, np.nan)
            cn_auto_val = results['cn_auto'].get((e1, e2), np.nan)
            row = {'Pair': f'{e1}-{e2}', 'Cutoff_A': cutoff,
                   'CN_fixed': cn, 'CN_auto': cn_auto_val}
            if (e1, e2) == ('Si', 'O') and 'cn_extra' in results:
                for rcut in si_o_cut_extra:
                    row[f'CN_{rcut:.2f}'] = results['cn_extra'].get(('Si','O'), {}).get(rcut, np.nan)
            cn_rows.append(row)
        pd.DataFrame(cn_rows).to_excel(writer, sheet_name='CN', index=False)

        # Qn Si
        qn_rows = [{'Qn': k, 'Percent': v, 'Paper_Percent': PAPER_QN_SI.get(k, np.nan)}
                   for k, v in results['qn_si'].items()]
        pd.DataFrame(qn_rows).to_excel(writer, sheet_name='Qn_Si', index=False)

        # Qn P
        qn_rows = [{'Qn': k, 'Percent': v, 'Paper_Percent': PAPER_QN_P.get(k, np.nan)}
                   for k, v in results['qn_p'].items()]
        pd.DataFrame(qn_rows).to_excel(writer, sheet_name='Qn_P', index=False)

        # Qn Combined
        qn_rows = [{'Qn': k, 'Percent': v} for k, v in results['qn_combined'].items()]
        pd.DataFrame(qn_rows).to_excel(writer, sheet_name='Qn_Combined', index=False)

        # NC
        nc_rows = [{'Type': k, 'NC': v, 'Paper_NC': PAPER_NC.get(k, np.nan)}
                   for k, v in results['nc'].items()]
        pd.DataFrame(nc_rows).to_excel(writer, sheet_name='NC', index=False)

        # O speciation
        n_o = sum(results['oxygen_speciation'].values())
        os_rows = [{'Type': k, 'Count': v, 'Percent': v/n_o*100 if n_o > 0 else 0}
                   for k, v in results['oxygen_speciation'].items()]
        pd.DataFrame(os_rows).to_excel(writer, sheet_name='O_speciation', index=False)

        # Clustering R_XX
        rxx_rows = [{'Modifier': m, 'R_obs_hom': v[0], 'CN_obs': v[1], 'CN_hom': v[2]}
                    for m, v in results['rxx'].items()]
        pd.DataFrame(rxx_rows).to_excel(writer, sheet_name='Clustering_Rxx', index=False)

        # Modifier Preference
        pref_rows = [{'Ratio': k, 'Value': v} for k, v in results['preference'].items()]
        if pref_rows:
            pd.DataFrame(pref_rows).to_excel(writer, sheet_name='Modifier_Preference', index=False)

        # R_X_Si/P
        rx_rows = [{'Element': e, 'R_X_Si/P': v} for e, v in results['r_x_si_p'].items()]
        pd.DataFrame(rx_rows).to_excel(writer, sheet_name='R_X_SiP', index=False)

        # Fnet
        pd.DataFrame({'Fnet_new': [results['fnet']], 'SBS_Ca': [32.0],
                      'SBS_Na': [20.0], 'SBS_Sr': [32.0]}).to_excel(
            writer, sheet_name='Fnet_New', index=False)

        # Si-O-P
        if results.get('total_p_o_bonds', 0) > 0:
            pd.DataFrame({'Total_P-O_bonds': [results['total_p_o_bonds']],
                          'P-O-Si_bonds': [results['p_si_links']],
                          'Fraction_P-O-Si': [results['frac_p_si']]}).to_excel(
                writer, sheet_name='Si-O-P', index=False)

        # Sensitivity
        if results.get('sensitivity'):
            pd.DataFrame(results['sensitivity']).to_excel(writer, sheet_name='Sensitivity', index=False)

        # Block Energy
        if 'block_energy' in results:
            pd.DataFrame([results['block_energy']]).to_excel(writer, sheet_name='Block_Energy', index=False)

        # Energy
        if results.get('energy_data'):
            ed = results['energy_data']
            total_steps = len(ed['total'])
            step_interval = max(1, total_steps // 10000)
            indices = np.arange(0, total_steps, step_interval)
            df_energy = pd.DataFrame({
                'Step': indices + 1,
                'Total_Energy_eV_per_atom': np.array(ed['total'])[indices],
                'Short_Range_eV_per_atom': np.array(ed['short'])[indices],
                'Coulomb_eV_per_atom': np.array(ed['coul'])[indices]
            })
            df_energy.to_excel(writer, sheet_name='Energy', index=False)

        # Angles Summary
        ang_rows = [{'Angle': k, 'Mean_deg': v} for k, v in results['angles'].items()]
        pd.DataFrame(ang_rows).to_excel(writer, sheet_name='Angles_Summary', index=False)

        # RDFs
        for (e1, e2), (r, gr) in rdf_data.items():
            pd.DataFrame({'r_A': r, 'g_r': gr}).to_excel(
                writer, sheet_name=f'RDF_{e1}{e2}', index=False)

    logger.info(f"Excel saved: {path}")


# ========================= CN CURVES =========================
def make_cn_curves(rdf_data, cn_fixed, box, counts, cutoffs, plots_dir):
    cn_results = {}
    for (e1, e2), (r, gr) in rdf_data.items():
        pair_key = (e1, e2) if (e1, e2) in cutoffs else (e2, e1)
        if pair_key not in cutoffs: continue
        n_target = counts.get(e2, 0)
        if n_target == 0: continue
        cutoff = cutoffs[pair_key]
        rho = n_target / box**3
        mask = r <= 6.0
        r_m = r[mask]; gr_m = gr[mask]
        if len(r_m) < 2: continue
        integrand = 4 * np.pi * r_m**2 * rho * gr_m
        cn = np.zeros_like(r_m)
        for i in range(1, len(r_m)):
            cn[i] = cn[i-1] + trapezoid(integrand[i-1:i+1], r_m[i-1:i+1])
        cn_results[f'{e1}-{e2}'] = (r_m, cn, cutoff)

    if not cn_results: return
    n_pairs = len(cn_results)
    ncols = 4; nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
    axes = axes.flatten() if nrows*ncols > 1 else [axes]
    for idx, (pair_name, (r_cn, cn, cutoff)) in enumerate(cn_results.items()):
        ax = axes[idx]
        ax.plot(r_cn, cn, color='#2980b9', linewidth=1.5)
        ax.axvline(x=cutoff, color='#e74c3c', linestyle='--',
                   label=f'Cutoff={cutoff:.2f} A, CN={cn[np.argmin(np.abs(r_cn-cutoff))]:.2f}')
        ax.set_xlabel('r (A)'); ax.set_ylabel('CN(r)'); ax.set_title(pair_name)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    for j in range(n_pairs, len(axes)): axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(plots_dir / 'CN_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"CN curves saved to {plots_dir / 'CN_curves.png'}")


# ========================= PLOTS =========================
def make_plots(results, rdf_data, angles, modifier_cn_dist, q, sn_q, plots_dir):
    # RDFs
    n_pairs = len(rdf_data)
    ncols = 4; nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.atleast_1d(axes).flatten()
    for idx, ((e1, e2), (r, gr)) in enumerate(rdf_data.items()):
        if idx >= len(axes): break
        ax = axes[idx]
        ax.plot(r, gr, lw=1)
        pair_key = (e1, e2) if (e1, e2) in CUTOFFS else (e2, e1)
        if pair_key in CUTOFFS:
            ax.axvline(CUTOFFS[pair_key], color='r', ls='--', lw=0.8,
                       label=f'cut={CUTOFFS[pair_key]}')
            ax.legend(fontsize=7)
        ax.set_xlabel('r (A)'); ax.set_ylabel('g(r)')
        ax.set_title(f'{e1}-{e2}'); ax.set_xlim(0, 8)
    for idx in range(n_pairs, len(axes)): axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig(plots_dir / 'rdf_all.png', dpi=300, bbox_inches='tight'); plt.close()

    # Bond angles
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for idx, (name, ang) in enumerate(angles.items()):
        if idx >= len(axes): break
        if len(ang) == 0:
            axes[idx].set_title(f'{name} (no data)'); continue
        axes[idx].hist(ang, bins=60, range=(30, 180), density=True, alpha=0.7)
        axes[idx].axvline(np.mean(ang), color='r', ls='--', label=f'mean={np.mean(ang):.1f}')
        axes[idx].set_xlabel('Angle (deg)'); axes[idx].set_ylabel('P(theta)')
        axes[idx].set_title(name); axes[idx].legend(fontsize=8)
    for idx in range(len(angles), len(axes)): axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig(plots_dir / 'bond_angles.png', dpi=300, bbox_inches='tight'); plt.close()

    # Qn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Q0', 'Q1', 'Q2', 'Q3', 'Q4']
    x_pos = np.arange(len(labels))
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    si_vals = [results['qn_si'].get(l, 0) for l in labels]
    si_paper = [PAPER_QN_SI.get(l, 0) for l in labels]
    ax1.bar(x_pos - 0.2, si_vals, 0.4, color=colors, label='Simulation')
    ax1.bar(x_pos + 0.2, si_paper, 0.4, color='gray', alpha=0.5, label='Paper')
    ax1.set_xticks(x_pos); ax1.set_xticklabels(labels)
    ax1.set_ylabel('%'); ax1.set_title('Si Qn'); ax1.legend()
    p_vals = [results['qn_p'].get(l, 0) for l in labels]
    p_paper = [PAPER_QN_P.get(l, 0) for l in labels]
    ax2.bar(x_pos - 0.2, p_vals, 0.4, color=colors, label='Simulation')
    ax2.bar(x_pos + 0.2, p_paper, 0.4, color='gray', alpha=0.5, label='Paper')
    ax2.set_xticks(x_pos); ax2.set_xticklabels(labels)
    ax2.set_ylabel('%'); ax2.set_title('P Qn'); ax2.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'qn_distribution.png', dpi=300, bbox_inches='tight'); plt.close()

    # Modifier CN distribution
    n_mod = len(modifier_cn_dist)
    if n_mod > 0:
        fig, axes = plt.subplots(1, n_mod, figsize=(5*n_mod, 4))
        axes = np.atleast_1d(axes)
        for idx, (mod, cn_arr) in enumerate(modifier_cn_dist.items()):
            if len(cn_arr) == 0: continue
            bins = np.arange(cn_arr.min(), cn_arr.max()+2) - 0.5
            axes[idx].hist(cn_arr, bins=bins, alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.mean(cn_arr), color='r', ls='--',
                              label=f'mean={np.mean(cn_arr):.2f}')
            axes[idx].set_xlabel('CN'); axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'{mod}-O CN distribution'); axes[idx].legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'modifier_cn_dist.png', dpi=300, bbox_inches='tight'); plt.close()

    # Neutron SF
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(q, sn_q, lw=1)
    ax.set_xlabel('Q (1/A)'); ax.set_ylabel('S_N(Q)')
    ax.set_title('Neutron Structure Factor'); ax.set_xlim(0, 25)
    plt.tight_layout()
    plt.savefig(plots_dir / 'neutron_sf.png', dpi=300, bbox_inches='tight'); plt.close()

    logger.info(f"Plots saved to: {plots_dir}")


# ========================= SUMMARY =========================
def print_summary(results, x_val):
    print("\n" + "=" * 70)
    print(f"  ANALYSIS SUMMARY - 45S5 + {x_val} mol% SrO")
    print("=" * 70)

    md = results['metadata']
    print("\n--- Density & Structure ---")
    print(f"  Density:        {md.get('Density_g_cm3', 0):.4f} g/cm3")
    print(f"  Molar volume:   {md.get('Molar_volume_cm3_mol', 0):.3f} cm3/mol")

    print("\n--- Bond Lengths (A) [Sim vs Paper] ---")
    for (e1, e2), bl in sorted(results['bond_lengths'].items()):
        if np.isnan(bl): continue
        paper = PAPER_BOND_LENGTHS.get((e1, e2), PAPER_BOND_LENGTHS.get((e2, e1), None))
        ps = f"{paper:.2f}" if paper else "  -  "
        flag = "  OK" if paper and abs(bl - paper) < 0.1 else ("  DIFF" if paper else "")
        print(f"  {e1}-{e2:2s}: {bl:.3f}  (paper: {ps}){flag}")

    print("\n--- Modifier CN [Sim vs Paper] ---")
    for mod, cn in results['modifier_cn'].items():
        paper = PAPER_CN.get(mod, None)
        ps = f"{paper:.1f}" if paper else "  -  "
        flag = "  OK" if paper and abs(cn - paper) < 0.5 else "  DIFF"
        print(f"  {mod}-O: {cn:.2f}  (paper: {ps}){flag}")

    print("\n--- Network Connectivity [Sim vs Paper] ---")
    for k, v in results['nc'].items():
        paper = PAPER_NC.get(k, None)
        ps = f"{paper:.2f}" if paper else "  -  "
        flag = "  OK" if paper and abs(v - paper) < 0.2 else "  DIFF"
        print(f"  NC ({k:8s}): {v:.3f}  (paper: {ps}){flag}")

    print("\n--- Oxygen Speciation ---")
    tot = sum(results['oxygen_speciation'].values())
    for k, v in results['oxygen_speciation'].items():
        print(f"  {k:4s}: {v:5d}  ({v/tot*100:5.1f}%)")

    print("\n--- Clustering R_obs/R_hom ---")
    for mod, (rxx, cn_obs, cn_hom) in results['rxx'].items():
        if rxx is not None and not np.isnan(rxx):
            print(f"  {mod}: R={rxx:.3f}  (CN_obs={cn_obs:.2f}, CN_hom={cn_hom:.2f})")

    print("\n--- R_X_Si/P (Si vs P preference) ---")
    for elem, v in results['r_x_si_p'].items():
        print(f"  {elem}: {v:.3f}")

    print("\n--- Modifier Preference (B vs C around A) ---")
    for k, v in results['preference'].items():
        print(f"  {k}: {v:.3f}")

    print("\n--- Qn Distribution (%) ---")
    print("  Si:  " + "  ".join(f"{k}:{v:.1f}" for k, v in results['qn_si'].items()))
    print("  P:   " + "  ".join(f"{k}:{v:.1f}" for k, v in results['qn_p'].items()))
    print("  Comb:" + "  ".join(f"{k}:{v:.1f}" for k, v in results['qn_combined'].items()))

    print("\n--- Sensitivity Analysis ---")
    if results.get('sensitivity'):
        for s in results['sensitivity']:
            print(f"  Cutoff={s['Cutoff']:.0f}A, alpha={s['Alpha']:.2f}: E={s['Energy']:.4f} eV/atom")

    if 'block_energy' in results:
        be = results['block_energy']
        print(f"\n--- Block Energy ---")
        print(f"  Mean: {be['mean']:.4f} eV/atom, Std: {be['std']:.4f}")

    print("=" * 70 + "\n")


# ========================= CLI =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone structural analysis for 45S5/Sr bioglass (FULL VERSION)")
    parser.add_argument("input", help="Path to final_structure.xyz")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--energy-log", type=str, default=None,
                        help="Path to energy_log.csv from Sr.py (optional)")
    args = parser.parse_args()
    run_analysis(args.input, output_dir=args.output_dir, energy_log_path=args.energy_log)
    