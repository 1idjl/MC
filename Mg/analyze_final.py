#!/usr/bin/env python3
"""
========================================================================
Final Structure Analysis for 45S5/Mg Bioglass v6.6 (NO Sr)
Based on: Xiang & Du, Chem. Mater. 2011 + Manuscript.docx (Mg Substitution)
ANALYSIS FEATURES:
- RDF, CN, Qn, Angle distributions
- Oxygen speciation (NBO, BO, Free)
- Clustering (Rxx), Modifier preference (around Si/P)
- R_X_Si/P ratios (Ca/Na/Mg around Si vs P)
- Neutron Structure Factor (S(Q))
- Sensitivity Analysis
- Excel export + plots
Usage:
python analyze_final.py --input final_structure.xyz --output-dir analysis_output
python analyze_final.py --input final_structure.xyz --energy-log energy_log.csv --output-dir analysis_output
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from numba import njit
from math import erf, exp, sqrt, pi
import os, sys, logging, pickle, re, warnings, argparse, codecs
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import typer

# ========================= UTF-8 LOGGING =========================
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None: stream = sys.stdout
        if hasattr(stream, 'buffer'):
            stream = codecs.getwriter('utf-8')(stream.buffer)
        super().__init__(stream)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('analysis.log', 'w', 'utf-8'), UTF8StreamHandler()])
logger = logging.getLogger(__name__)

# ========================= CONSTANTS =========================
KB_EV = 8.617333262145e-5
NA = 6.02214076e23
KE = 14.3996454784255
SKIN = 1.5
SWITCH_DR = 0.3
BASE_N_ATOMS = 2835

# Type maps (NO Sr)
TYPE_MAP = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Mg': 5} # Sr removed, Mg added
ELEM_MAP = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Mg'} # Sr removed, Mg added
# Masses (NO Sr)
MASS = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
        'P': 30.97376, 'O': 15.999, 'Mg': 24.305} # Sr removed, Mg added
# Neutron scattering lengths (NO Sr)
NEUTRON_B = {'O': 5.803, 'Si': 4.1491, 'Na': 3.63, 'Ca': 4.70, 'P': 5.13, 'Mg': 5.378} # Sr removed, Mg added

# Paper reference values (NO Sr)
PAPER_BOND_LENGTHS = {('O','O'):2.63, ('Si','O'):1.61, ('Ca','O'):2.38, ('Na','O'):2.40, ('Mg','O'): 2.1} # Added Mg-O
PAPER_CN = {'Ca': 6.2, 'Na': 5.7, 'Mg': 6.0} # Removed Sr, added Mg
PAPER_QN_SI = {'Q0':3.7, 'Q1':23.2, 'Q2':39.8, 'Q3':27.8, 'Q4':5.6}
PAPER_QN_P = {'Q0':50.0, 'Q1':42.9, 'Q2':6.4, 'Q3':0.6, 'Q4':0.0}
PAPER_NC = {'Si': 2.07, 'P': 0.62, 'overall': 1.953} # Example for 45-M5 from manuscript

# Cutoffs for CN calculation (NO Sr)
CUTOFFS = {('Si','O'):2.25, ('P','O'):2.25, ('Na','O'):3.34, ('Ca','O'):3.14, ('O','O'):2.91,
           ('P','Ca'):4.44, ('P','Na'):4.44,
           ('Si','Ca'):4.37, ('Si','Na'):4.42,
           ('Ca','Ca'):4.85, ('Na','Na'):4.15,
           ('Ca','Na'):4.94,
           # Mg cutoffs
           ('Mg','O'): 2.5, ('Mg','Ca'): 3.2, ('Mg','Na'): 3.0, ('Mg','Mg'): 3.5, ('Mg','Si'): 3.5, ('Mg','P'): 3.5 # Added Mg cutoffs
          }

PEAK_RANGES = {('Si','O'):(1.40,2.00), ('P','O'):(1.30,1.90), ('Na','O'):(2.00,2.80),
               ('Ca','O'):(2.00,2.80), ('O','O'):(2.30,3.10),
               ('Si','Si'):(2.80,3.60), ('Si','P'):(2.70,3.40), ('P','P'):(2.70,3.40),
               ('Si','Na'):(2.90,3.70), ('Si','Ca'):(3.20,4.10),
               ('Na','Na'):(2.50,3.50), ('Ca','Ca'):(3.20,4.20),
               ('Ca','Na'):(3.00,4.00),
               ('P','Ca'):(3.20,4.20), ('P','Na'):(3.00,4.00),
               # Mg peak ranges
               ('Mg','O'):(1.8, 2.6), ('Mg','Mg'):(2.5, 3.5), ('Mg','Ca'):(2.8, 4.0), ('Mg','Na'):(2.5, 3.5), ('Mg','Si'):(3.0, 4.0), ('Mg','P'):(3.0, 4.0) # Added Mg ranges
              }

RMIN_VALUES = {('Si','O'):1.4, ('P','O'):1.3, ('Na','O'):2.0, ('Ca','O'):2.0, ('O','O'):2.2,
               ('P','Ca'):3.5, ('P','Na'):3.5,
               ('Si','Ca'):3.5, ('Si','Na'):3.0,
               ('Ca','Ca'):3.5, ('Na','Na'):3.0,
               ('Ca','Na'):3.5,
               # Mg Rmin
               ('Mg','O'): 1.8, ('Mg','Mg'): 2.5, ('Mg','Ca'): 2.8, ('Mg','Na'): 2.5, ('Mg','Si'): 3.0, ('Mg','P'): 3.0 # Added Mg Rmin
              }

# Specific Born-Mayer parameters (NO Sr)
SBS_X_O = {'Ca': 32.0, 'Na': 20.0, 'Mg': 25.0} # Sr removed, Mg added (example value)

# ========================= NUMBA FUNCTIONS =========================
@njit(fastmath=True, cache=True)
def minimum_image(dr, box):
    dr -= box * np.floor(dr / box + 0.5)
    return dr

@njit(fastmath=True, cache=True)
def compute_rdf(coords, types, ti, tj, box, cutoff, nbins=500, rmax=None):
    if rmax is None: rmax = cutoff + 2.0
    n_bins = nbins
    dr = rmax / n_bins
    r = np.arange(0.0, rmax, dr) + dr/2
    rdf = np.zeros(n_bins)
    mi = types == ti
    mj = types == tj
    ni = int(np.sum(mi))
    nj = int(np.sum(mj))
    if ni == 0 or nj == 0 or (ti == tj and ni < 2):
        return r, np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)

    tree = cKDTree(coords[mi], boxsize=box)
    all_neigh = tree.query_ball_point(coords[mj], cutoff)

    for idx_j, neigh_list in enumerate(all_neigh):
        j_global = np.where(mj)[0][idx_j]
        for idx_i_local in neigh_list:
            i_global = np.where(mi)[0][idx_i_local]
            if i_global == j_global: continue # Avoid self-pairing
            dr_vec = coords[i_global] - coords[j_global]
            dr_vec = minimum_image(dr_vec, box)
            dist = np.sqrt(np.sum(dr_vec**2))
            bin_idx = int(dist / dr)
            if 0 <= bin_idx < n_bins:
                rdf[bin_idx] += 1.0

    # Normalize RDF
    rho_j = nj / box**3 # Number density of type j
    shell_volumes = 4 * np.pi * (r**2) * dr
    norm_factors = ni * rho_j * shell_volumes
    rdf = np.divide(rdf, norm_factors, out=np.zeros_like(rdf), where=norm_factors!=0)
    return r, rdf, np.zeros_like(r), np.zeros_like(r) # g(r), gr(r), F(r), dF(r)

@njit(fastmath=True, cache=True)
def coordination_number(r, gr, cutoff, n_o, box):
    """Calculate coordination number by integrating g(r) up to cutoff."""
    # Use trapezoidal rule for integration
    mask = (r > 0.0) & (r <= cutoff)
    if not np.any(mask):
        return 0.0
    r_int = r[mask]
    gr_int = gr[mask]
    integrand = 4 * np.pi * r_int**2 * (gr_int - 1) / box**3
    cn = np.trapz(integrand, r_int)
    return cn

# ========================= ANALYSIS FUNCTIONS =========================
def compute_qn(coords, types, network_elem, o_elem, box, cutoffs=CUTOFFS):
    """Compute Qn distribution for network former (Si/P)."""
    network_type = TYPE_MAP[network_elem]
    o_type = TYPE_MAP[o_elem]
    o_indices = np.where(types == o_type)[0]
    network_indices = np.where(types == network_type)[0]

    qn_counts = {f'Q{n}': 0 for n in range(5)} # Q0 to Q4

    for net_idx in network_indices:
        neigh_indices = _get_neighbors(coords, types, net_idx, o_type, box, cutoffs.get((network_elem, o_elem), 2.25))
        bridging_o_count = 0
        for o_idx in neigh_indices:
            # Check if this O is bonded to another network former
            o_neigh_indices = _get_neighbors(coords, types, o_idx, network_type, box, cutoffs.get((o_elem, network_elem), 2.25))
            if len(o_neigh_indices) >= 2: # If O connects 2 or more network formers, it's bridging
                bridging_o_count += 1
        qn_state = len(neigh_indices) - bridging_o_count # Non-bridging O count
        qn_label = f'Q{min(qn_state, 4)}' # Cap at Q4
        qn_counts[qn_label] += 1

    # Convert counts to percentages
    total_network = len(network_indices)
    if total_network > 0:
        qn_percentages = {k: (v / total_network) * 100 for k, v in qn_counts.items()}
    else:
        qn_percentages = {k: 0.0 for k in qn_counts}
    return qn_percentages

def _get_neighbors(coords, types, idx, target_type, box, cutoff):
    """Helper to find neighbors of a specific type within a cutoff."""
    pos = coords[idx]
    tree = cKDTree(np.delete(coords, idx, axis=0), boxsize=box)
    # Adjust indices because we removed 'idx'
    rel_coords = np.delete(coords, idx, axis=0)
    rel_types = np.delete(types, idx)
    neigh_indices_rel = tree.query_ball_point(pos, cutoff)
    # Map relative indices back to original indices
    original_neigh_indices = []
    for rel_idx in neigh_indices_rel:
        orig_idx = rel_idx if rel_idx < idx else rel_idx + 1
        if types[orig_idx] == target_type:
            original_neigh_indices.append(orig_idx)
    return original_neigh_indices

def compute_network_connectivity(coords, types, box, cutoffs=CUTOFFS):
    """Compute average connectivity for Si, P, and overall."""
    si_type = TYPE_MAP['Si']
    p_type = TYPE_MAP['P']
    o_type = TYPE_MAP['O']

    si_cn_sum = 0
    p_cn_sum = 0
    si_count = 0
    p_count = 0

    for i in range(len(coords)):
        if types[i] == si_type:
            neigh_o = _get_neighbors(coords, types, i, o_type, box, cutoffs.get(('Si', 'O'), 2.25))
            si_cn_sum += len(neigh_o)
            si_count += 1
        elif types[i] == p_type:
            neigh_o = _get_neighbors(coords, types, i, o_type, box, cutoffs.get(('P', 'O'), 2.25))
            p_cn_sum += len(neigh_o)
            p_count += 1

    avg_si_cn = si_cn_sum / si_count if si_count > 0 else 0
    avg_p_cn = p_cn_sum / p_count if p_count > 0 else 0
    total_o = np.sum(types == o_type)
    total_network = si_count + p_count
    avg_overall_cn = (si_cn_sum + p_cn_sum) / total_network if total_network > 0 else 0

    return {'Si': avg_si_cn, 'P': avg_p_cn, 'overall': avg_overall_cn}

def compute_oxygen_speciation(coords, types, box, cutoffs=CUTOFFS):
    """Compute distribution of O types (Q, NBO, BO, free)."""
    o_type = TYPE_MAP['O']
    si_type = TYPE_MAP['Si']
    p_type = TYPE_MAP['P']
    o_indices = np.where(types == o_type)[0]

    nbo_count = 0 # Non-Bridging O
    bo_count = 0  # Bridging O (Si-O-Si, Si-O-P, P-O-P, etc.)
    free_o_count = 0 # O not bonded to Si or P

    for o_idx in o_indices:
        o_neigh_types = set()
        # Find neighbors bonded to Si or P
        si_neigh = _get_neighbors(coords, types, o_idx, si_type, box, cutoffs.get(('O', 'Si'), 2.25))
        p_neigh = _get_neighbors(coords, types, o_idx, p_type, box, cutoffs.get(('O', 'P'), 2.25))
        o_neigh_types.update([types[n] for n in si_neigh])
        o_neigh_types.update([types[n] for n in p_neigh])

        if len(o_neigh_types) == 0:
            free_o_count += 1
        elif len(o_neigh_types) == 1:
            nbo_count += 1 # Connected to only one network former type
        else:
            bo_count += 1 # Connected to multiple or different network formers (bridging)

    return {'NBO': nbo_count, 'BO': bo_count, 'Free_O': free_o_count}

def compute_clustering_rxx(coords, types, mod_elem, box, cutoffs=CUTOFFS):
    """Compute clustering parameter Rxx for a modifier."""
    mod_type = TYPE_MAP[mod_elem]
    mod_indices = np.where(types == mod_type)[0]
    if len(mod_indices) < 2: return np.nan, np.nan, np.nan

    mod_coords = coords[mod_indices]
    tree = cKDTree(mod_coords, boxsize=box)

    # Observed average CN
    obs_cn = 0.0
    cut = cutoffs.get((mod_elem, mod_elem), 4.0) # Use homonuclear cutoff
    for i, coord in enumerate(mod_coords):
        neigh = tree.query_ball_point(coord, cut)
        obs_cn += len([n for n in neigh if n != i]) # Exclude self
    obs_cn /= len(mod_coords)

    # Expected CN for homogeneous distribution
    n_mod = len(mod_indices)
    vol_sphere = 4/3 * np.pi * cut**3
    rho_mod = (n_mod - 1) / (box**3) # Excluding central atom
    hom_cn = rho_mod * vol_sphere

    rxx = obs_cn / hom_cn if hom_cn > 0 else np.nan
    return rxx, obs_cn, hom_cn

def compute_preference(coords, types, box, A, B, C, cutoffs=CUTOFFS):
    """Compute preference ratio B vs C around A."""
    ta, tb, tc = TYPE_MAP[A], TYPE_MAP[B], TYPE_MAP[C]
    a_indices = np.where(types == ta)[0]
    b_indices = set(np.where(types == tb)[0])
    c_indices = set(np.where(types == tc)[0])
    if not a_indices.size or not b_indices or not c_indices: return np.nan

    cut_ab = cutoffs.get((A, B), cutoffs.get((B, A), 4.0))
    cut_ac = cutoffs.get((A, C), cutoffs.get((C, A), 4.0))

    pref_ratio = 0.0
    for a_idx in a_indices:
        a_coord = coords[a_idx]
        tree = cKDTree(coords, boxsize=box)
        neigh = tree.query_ball_point(a_coord, max(cut_ab, cut_ac))
        b_count = sum(1 for n in neigh if n in b_indices and np.linalg.norm(minimum_image(coords[n] - a_coord, box)) <= cut_ab)
        c_count = sum(1 for n in neigh if n in c_indices and np.linalg.norm(minimum_image(coords[n] - a_coord, box)) <= cut_ac))
        if b_count + c_count > 0:
            pref_ratio += b_count / (b_count + c_count)
    return pref_ratio / len(a_indices) if len(a_indices) > 0 else np.nan

def compute_r_x_si_p(coords, types, box, counts, cutoffs=CUTOFFS):
    """Compute R_X_Si/P ratios (Ca/Na/Mg around Si vs P)."""
    r_x_si_p = {}
    for X in ['Ca', 'Na', 'Mg']: # Removed Sr
        if counts.get(X, 0) == 0: continue
        tx = TYPE_MAP[X]
        si_indices = np.where(types == TYPE_MAP['Si'])[0]
        p_indices = np.where(types == TYPE_MAP['P'])[0]
        x_indices = set(np.where(types == tx)[0])

        cut_xsi = cutoffs.get((X, 'Si'), cutoffs.get(('Si', X), 3.5))
        cut_xp = cutoffs.get((X, 'P'), cutoffs.get(('P', X), 3.5))

        x_around_si = 0
        x_around_p = 0

        for si_idx in si_indices:
             si_coord = coords[si_idx]
             tree = cKDTree(coords, boxsize=box)
             neigh_si = tree.query_ball_point(si_coord, cut_xsi)
             x_around_si += sum(1 for n in neigh_si if n in x_indices and n != si_idx)

        for p_idx in p_indices:
             p_coord = coords[p_idx]
             tree = cKDTree(coords, boxsize=box)
             neigh_p = tree.query_ball_point(p_coord, cut_xp)
             x_around_p += sum(1 for n in neigh_p if n in x_indices and n != p_idx)

        total_x_around_net = x_around_si + x_around_p
        r_x_si_p[X] = x_around_si / total_x_around_net if total_x_around_net > 0 else np.nan
    return r_x_si_p

def compute_neutron_sf(coords, types, box, qmin=0.5, qmax=12.0, nq=500):
    """Compute neutron structure factor S(Q)."""
    n = len(coords)
    q_values = np.linspace(qmin, qmax, nq)
    sq = np.ones(nq) # Initialize S(Q) = 1

    b_values = np.array([NEUTRON_B[ELEM_MAP[t]] for t in types])

    for i, q in enumerate(q_values):
        sum_real = 0.0
        sum_imag = 0.0
        b_sq_avg = np.mean(b_values**2)
        for j in range(n):
            for k in range(j+1, n):
                dr = coords[j] - coords[k]
                dr = minimum_image(dr, box)
                r = np.linalg.norm(dr)
                bq = b_values[j] * b_values[k] * np.sin(q * r) / (q * r) if r > 1e-6 else b_values[j] * b_values[k]
                sum_real += bq
        # S(Q) = 1 + (2/V) * sum_{j!=k} b_j b_k sin(qr)/qr / <b^2>
        # Simplified: S(Q) = sum_real / (n*(n-1)/2) / b_sq_avg
        # More accurately: S(Q) = (sum_real + sum(b_i^2)) / sum(b_i)^2
        # For large N: S(Q) ~ sum_real / (N*N_avg_b^2) + 1
        # Let's use the definition: S(Q) = |sum_n b_n exp(iQ.r_n)|^2 / sum_n b_n^2
        fq_real = np.sum(b_values * np.cos(q * coords[:, 0]))**2 + np.sum(b_values * np.cos(q * coords[:, 1]))**2 + np.sum(b_values * np.cos(q * coords[:, 2]))**2
        fq_imag = np.sum(b_values * np.sin(q * coords[:, 0]))**2 + np.sum(b_values * np.sin(q * coords[:, 1]))**2 + np.sum(b_values * np.sin(q * coords[:, 2]))**2
        sq[i] = (fq_real + fq_imag) / np.sum(b_values**2)

    return q_values, sq

# ========================= MAIN ANALYSIS =========================
def run_analysis(xyz_path, output_dir=None, energy_log_path=None):
    logger.info(f"Loading structure from: {xyz_path}")
    with open(xyz_path, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    comment_line = lines[1].strip()

    # Regex patterns to extract metadata
    patterns = [
        ('x', r'x\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('N', r'N\s*=\s*(\d+)', int),
        ('rho', r'rho\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('box', r'box\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('seed', r'seed\s*=\s*(\d+)', int),
    ]
    meta = {}
    for key, pat, conv in patterns:
        m = re.search(pat, comment_line)
        if m:
            meta[key] = conv(m.group(1))

    x_val = meta.get('x', 0.0) # Assume x represents Mg percentage now
    box = meta.get('box', None)
    if box is None:
        logger.error("Box size not found in XYZ comment line.")
        sys.exit(1)

    symbols, coords_list = [], []
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coords = np.array(coords_list, dtype=np.float64)
    types = np.array([TYPE_MAP[s] for s in symbols], dtype=np.int32)
    counts = {e: int(np.sum(types == TYPE_MAP[e])) for e in set(symbols)}
    logger.info(f"Composition: {counts}")

    # Calculate mass and effective density
    total_mass = sum(MASS[s] for s in symbols)
    eff_density = total_mass / (NA * box**3) * 1e24
    n_oxide_units = n_atoms / 2.835 # Keep original scaling denominator?
    molar_volume = (total_mass / n_oxide_units) / eff_density if n_oxide_units > 0 else np.nan

    results = {}
    results['metadata'] = {
        'x_mol_percent_MgO': x_val, # Store the x value used, removed Sr label
        'N_atoms': n_atoms,
        'Box_A': box,
        'Density_g_cm3': eff_density,
        'Molar_volume_cm3_mol': molar_volume,
        **{f'N_{e}': counts.get(e, 0) for e in ['Si', 'P', 'Na', 'Ca', 'O', 'Mg']} # Include all counts, removed Sr
    }

    logger.info("Computing RDFs and Coordination Numbers...")
    # --- 2. RDF & CN ---
    results['bond_lengths'] = {}
    results['modifier_cn'] = {}
    for mod in ['Na', 'Ca', 'Mg']: # Include Mg, removed Sr
        if TYPE_MAP[mod] not in types: continue
        ti = TYPE_MAP[mod]
        tj = TYPE_MAP['O']
        ei, ej = mod, 'O'
        cut = CUTOFFS.get((ei, ej), CUTOFFS.get((ej, ei), 3.5))
        r, gr, _, _ = compute_rdf(coords, types, ti, tj, box, cut)
        if len(r) > 0:
            results['modifier_cn'][mod] = coordination_number(r, gr, cut, counts.get('O', 1), box)

            # Extract bond length from peak
            peak_ranges = PEAK_RANGES.get((ei, ej), PEAK_RANGES.get((ej, ei), (1.0, 3.0)))
            peak_mask = (r >= peak_ranges[0]) & (r <= peak_ranges[1])
            if np.any(peak_mask) and np.max(gr[peak_mask]) > 0.1:
                peak_idx = np.argmax(gr[peak_mask])
                peak_r = r[peak_mask][peak_idx]
                results['bond_lengths'][(ei, ej)] = peak_r
            else:
                results['bond_lengths'][(ei, ej)] = np.nan

    # --- 3. Qn Distribution ---
    # Si Qn
    si_qn = compute_qn(coords, types, 'Si', 'O', box)
    results['qn_si'] = si_qn
    # P Qn
    p_qn = compute_qn(coords, types, 'P', 'O', box)
    results['qn_p'] = p_qn

    # --- 4. Network Connectivity ---
    nc = compute_network_connectivity(coords, types, box)
    results['nc'] = nc

    # --- 5. Oxygen Speciation ---
    o_spec = compute_oxygen_speciation(coords, types, box)
    results['oxygen_speciation'] = o_spec

    # --- 6. Clustering (Rxx) ---
    rxx_results = {}
    for mod in ['Ca', 'Na', 'Mg']: # Include Mg, removed Sr
         if TYPE_MAP[mod] not in types: continue
         rxx, obs_cn, hom_cn = compute_clustering_rxx(coords, types, mod, box)
         rxx_results[mod] = (rxx, obs_cn, hom_cn)
    results['rxx'] = rxx_results

    # --- 7. R_X_Si/P (from Sr.py v6.6 logic) ---
    r_x_si_p = compute_r_x_si_p(coords, types, box, counts)
    results['r_x_si_p'] = r_x_si_p

    # --- 8. Neutron Structure Factor (S(Q)) ---
    q, sq = compute_neutron_sf(coords, types, box)
    results['neutron_sf'] = {'q': q, 'sq': sq}

    # --- 9. Energy Log (if provided) ---
    if energy_log_path and Path(energy_log_path).exists():
        logger.info(f"Loading energy log from: {energy_log_path}")
        try:
            df_energy = pd.read_csv(energy_log_path)
            results['energy_log'] = df_energy.to_dict('list')
        except Exception as e:
            logger.error(f"Could not load energy log: {e}")

    # --- OUTPUT DIRECTORY ---
    if output_dir is None:
        output_dir = Path(f"analysis_x{x_val}_N{n_atoms}")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- PLOTS ---
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot Qn Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f'Q{n}' for n in range(5)]
    x_pos = np.arange(len(labels))
    ax1.bar(x_pos, [si_qn.get(l, 0) for l in labels], 0.6, color='skyblue')
    ax1.set_xticks(x_pos); ax1.set_xticklabels(labels); ax1.set_ylabel('%'); ax1.set_title('Si Qn Distribution'); ax1.grid(alpha=0.3, axis='y')
    ax2.bar(x_pos, [p_qn.get(l, 0) for l in labels], 0.6, color='lightcoral')
    ax2.set_xticks(x_pos); ax2.set_xticklabels(labels); ax2.set_ylabel('%'); ax2.set_title('P Qn Distribution'); ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(plots_dir / 'qn_distributions.png', dpi=300, bbox_inches='tight'); plt.close()

    # Plot Modifier CN
    fig, ax = plt.subplots(figsize=(8, 6))
    mods = [m for m in ['Na', 'Ca', 'Mg'] if m in results['modifier_cn']] # Removed Sr
    if mods:
        cn_vals = [results['modifier_cn'][m] for m in mods]
        bars = ax.bar(mods, cn_vals, color=['orange', 'lightgreen', 'pink']) # Color for Mg
        ax.set_ylabel('Coordination Number'); ax.set_title('Modifier-Oxygen CN');
        ax.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, cn_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(plots_dir / 'modifier_cn.png', dpi=300, bbox_inches='tight'); plt.close()

    # Plot Neutron SF
    q_vals = results['neutron_sf']['q']
    sq_vals = results['neutron_sf']['sq']
    plt.figure(figsize=(10, 6)); plt.plot(q_vals, sq_vals, 'b-', linewidth=1.5); plt.xlabel('Q (Å⁻¹)'); plt.ylabel('S(Q)'); plt.title('Neutron Structure Factor'); plt.grid(alpha=0.3); plt.savefig(plots_dir / 'neutron_sf.png', dpi=300, bbox_inches='tight'); plt.close()

    logger.info(f"Plots saved to: {plots_dir}")

    # --- EXCEL EXPORT ---
    excel_path = output_dir / "analysis_results.xlsx"
    logger.info(f"Exporting results to Excel: {excel_path}")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Metadata
        meta_df = pd.DataFrame(list(results['metadata'].items()), columns=['Property', 'Value'])
        meta_df.to_excel(writer, sheet_name='Metadata', index=False)

        # Summary Stats
        summary_data = []
        for key, value in results.items():
            if key in ['metadata', 'energy_log']: continue # Handle separately
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_key, tuple):
                         name = f"{sub_key[0]}-{sub_key[1]}"
                    else:
                         name = str(sub_key)
                    summary_data.append({'Measurement': f"{key}_{name}", 'Value': sub_value})
            else:
                summary_data.append({'Measurement': key, 'Value': value})
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary_Stats', index=False)

        # Energy Log
        if 'energy_log' in results:
             edf = pd.DataFrame(results['energy_log'])
             edf.to_excel(writer, sheet_name='Energy_Log', index=False)

        # Final Structure
        struct_df = pd.DataFrame({
            'Symbol': symbols,
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2]
        })
        struct_df.to_excel(writer, sheet_name='Final_Structure', index=False)

    logger.info(f"Excel export completed: {excel_path}")

    # --- PICKLE RESULTS ---
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results pickled to: {output_dir / 'results.pkl'}")

    # --- SUMMARY PRINT ---
    print("")
    print("=" * 70)
    print(f" ANALYSIS SUMMARY - 45S5 + {x_val} mol% MgO") # Updated label, removed Sr
    print("=" * 70)

    md = results['metadata']
    print("- Density & Structure -")
    print(f" Density: {md.get('Density_g_cm3', 0):.4f} g/cm3")
    print(f" Molar volume: {md.get('Molar_volume_cm3_mol', 0):.3f} cm3/mol")

    print("- Bond Lengths (A) [Sim vs Paper] -")
    for (e1, e2), bl in sorted(results['bond_lengths'].items()):
        if np.isnan(bl): continue
        paper = PAPER_BOND_LENGTHS.get((e1, e2), PAPER_BOND_LENGTHS.get((e2, e1), None))
        ps = f"{paper:.2f}" if paper else " - "
        flag = " OK" if paper and abs(bl - paper) < 0.1 else (" DIFF" if paper else "")
        print(f" {e1}-{e2:2s}: {bl:.3f} (paper: {ps}){flag}")

    print("- Modifier CN [Sim vs Paper] -")
    # Use appropriate paper dictionary based on simulated composition
    paper_dict = PAPER_CN # Or another dictionary based on x
    for mod, cn in results.get('modifier_cn', {}).items():
        paper = paper_dict.get(mod, None)
        ps = f"{paper:.1f}" if paper else " - "
        flag = " OK" if paper and abs(cn - paper) < 0.5 else " DIFF"
        print(f" {mod}-O: {cn:.2f} (paper: {ps}){flag}")

    print("- Network Connectivity [Sim vs Paper] -")
    for k, v in results.get('nc', {}).items():
        paper = PAPER_NC.get(k, None)
        ps = f"{paper:.2f}" if paper else " - "
        flag = " OK" if paper and abs(v - paper) < 0.2 else " DIFF"
        print(f" NC ({k:8s}): {v:.3f} (paper: {ps}){flag}")

    print("- Oxygen Speciation -")
    tot = sum(results['oxygen_speciation'].values())
    for k, v in results['oxygen_speciation'].items():
        print(f" {k:4s}: {v:5d} ({v/tot*100:5.1f}%)")

    print("- Clustering R_obs/R_hom -")
    for mod, (rxx, cn_obs, cn_hom) in results['rxx'].items():
        if rxx is not None and not np.isnan(rxx):
            print(f" {mod}: R={rxx:.3f} (CN_obs={cn_obs:.2f}, CN_hom={cn_hom:.2f})")

    print("- R_X_Si/P (Si vs P preference) -")
    for elem, v in results['r_x_si_p'].items():
        print(f" {elem}: {v:.3f}")

    print("- Qn Distribution (%) -")
    print(" Si: " + " ".join(f"{k}:{v:.1f}" for k, v in results['qn_si'].items()))
    print(" P:  " + " ".join(f"{k}:{v:.1f}" for k, v in results['qn_p'].items()))

    print("=" * 70 + "")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Final Structure Analysis for 45S5/Mg Bioglass (NO Sr)')
    parser.add_argument("--input", type=str, required=True, help="Input XYZ file (final structure)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory name")
    parser.add_argument("--energy-log", type=str, default=None, help="Path to energy_log.csv from simulation.py (optional)")

    args = parser.parse_args()

    run_analysis(args.input, output_dir=args.output_dir, energy_log_path=args.energy_log)
