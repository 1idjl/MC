
#!/usr/bin/env python3
"""
========================================================================
Structure Generator for Mg-doped 45S5 Bioactive Glass (FINAL PATCHED v2)
Based on: Moghanian et al. manuscript
========================================================================
"""
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial import cKDTree
from math import sqrt, sin, cos, pi

NA = 6.02214076e23

# ========================= PAPER DATA (Moghanian et al.) =========================
PAPER_COMPOSITION = {
    0:  {"label": "45-M0",  "N_Mg": 0,   "N_Ca": 269, "density": 2.673801},
    1:  {"label": "45-M1",  "N_Mg": 10,  "N_Ca": 259, "density": 2.654335},
    3:  {"label": "45-M3",  "N_Mg": 30,  "N_Ca": 239, "density": 2.638811},
    5:  {"label": "45-M5",  "N_Mg": 50,  "N_Ca": 219, "density": 2.629815},
    8:  {"label": "45-M8",  "N_Mg": 80,  "N_Ca": 189, "density": 2.618935},
    10: {"label": "45-M10", "N_Mg": 100, "N_Ca": 169, "density": 2.614384},
    15: {"label": "45-M15", "N_Mg": 150, "N_Ca": 119, "density": 2.584558},
    20: {"label": "45-M20", "N_Mg": 200, "N_Ca": 69,  "density": 2.550217},
}

PAPER_NC = {
    0: 1.926, 1: 1.922, 3: 1.938, 5: 1.953,
    8: 1.953, 10: 1.981, 15: 1.973, 20: 2.035,
}

# Qn Distribution from Table 4 (Probabilities for Q0 to Q4)
PAPER_QN = {
    0:  [0.0760, 0.2593, 0.3782, 0.2359, 0.0507],
    1:  [0.0916, 0.2300, 0.3743, 0.2729, 0.0312],
    3:  [0.0721, 0.2534, 0.3782, 0.2573, 0.0390],
    5:  [0.0975, 0.2281, 0.3645, 0.2437, 0.0663],
    8:  [0.0721, 0.2534, 0.3782, 0.2417, 0.0546],
    10: [0.0741, 0.2398, 0.3918, 0.2203, 0.0741],
    15: [0.0741, 0.2417, 0.3782, 0.2495, 0.0565],
    20: [0.0838, 0.2008, 0.3704, 0.2865, 0.0585],
}

# Bond lengths (Angstrom)
SI_O_BOND = 1.61
P_O_BOND = 1.50
NA_O_BOND = 2.40
CA_O_BOND = 2.36
MG_O_BOND = 2.00

# ========================= PATCHED CONSTANTS v2 =========================
# Network Former placement distances
MIN_SI_SI = 3.05      
MIN_SI_P  = 2.95      
MIN_P_P   = 3.40      

# Topology search range (Expanded to ensure enough candidates for BOs)
SI_SI_MIN, SI_SI_MAX = 3.00, 4.40   
SI_P_MIN,  SI_P_MAX  = 2.90, 4.10   
P_P_MIN,   P_P_MAX   = 2.85, 3.80   

# Safety distances
ANALYSIS_CUTOFF = 2.25
NBO_SAFE_DIST  = 2.50   # Increased margin
NBO_O_OVERLAP  = 1.95   


def minimum_image(dr, box):
    return dr - box * np.round(dr / box)


def random_rotation(rng):
    u1, u2, u3 = rng.random(3)
    q0 = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    q1 = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    q2 = sqrt(u1) * sin(2.0 * pi * u3)
    q3 = sqrt(u1) * cos(2.0 * pi * u3)
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0*(q2*q2 + q3*q3)
    R[0, 1] = 2.0*(q1*q2 - q0*q3)
    R[0, 2] = 2.0*(q1*q3 + q0*q2)
    R[1, 0] = 2.0*(q1*q2 + q0*q3)
    R[1, 1] = 1.0 - 2.0*(q1*q1 + q3*q3)
    R[1, 2] = 2.0*(q2*q3 - q0*q1)
    R[2, 0] = 2.0*(q1*q3 - q0*q2)
    R[2, 1] = 2.0*(q2*q3 + q0*q1)
    R[2, 2] = 1.0 - 2.0*(q1*q1 + q2*q2)
    return R


# ========================= STEP 1: NETWORK FORMERS =========================
def place_network_formers(n_si, n_p, box, rng):
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    types = np.zeros(n_total, dtype=np.int32)  # 0=Si, 1=P
    
    placed = 0
    for i in range(n_total):
        is_p = (i >= n_si)
        success = False
        
        for _ in range(200000):
            pos = rng.uniform(0.0, box, 3)
            if placed == 0:
                coords[placed] = pos; types[placed] = 1 if is_p else 0
                placed += 1; success = True; break

            d = coords[:placed] - pos
            d = minimum_image(d, box)
            r = np.sqrt(np.sum(d*d, axis=1))
            
            valid = True
            for k in range(placed):
                if types[k] == 1 and is_p:
                    if r[k] < MIN_P_P: valid = False; break
                elif types[k] == 1 and not is_p:
                    if r[k] < MIN_SI_P: valid = False; break
                elif types[k] == 0 and is_p:
                    if r[k] < MIN_SI_P: valid = False; break
                else:
                    if r[k] < MIN_SI_SI: valid = False; break
            
            if valid:
                coords[placed] = pos; types[placed] = 1 if is_p else 0
                placed += 1; success = True; break
        
        if not success:
            best_pos, best_min = None, -1
            for _ in range(5000):
                p = rng.uniform(0.0, box, 3)
                d = coords[:placed] - p
                d = minimum_image(d, box)
                r = np.sqrt(np.sum(d*d, axis=1)).min()
                if r > best_min:
                    best_min, best_pos = r, p
            coords[placed] = best_pos; types[placed] = 1 if is_p else 0
            placed += 1
            
    return coords, types


# ========================= STEP 2: BUILD TOPOLOGY WITH Qn TARGET =========================
def build_topology_with_qn(nf_coords, nf_types, box, n_bo_target, n_siop_max, qn_target, rng):
    n_nf = len(nf_coords)
    tree = cKDTree(nf_coords, boxsize=box)
    pairs = tree.query_pairs(4.5, output_type="ndarray")
    
    # Exact target degree assignment based on Qn
    qn_probs = qn_target[:5]
    target_degrees = []
    for d, prob in enumerate(qn_probs):
        count = int(round(prob * n_nf))
        target_degrees.extend([d] * count)
    while len(target_degrees) < n_nf:
        target_degrees.append(2)
    target_degrees = target_degrees[:n_nf]
    rng.shuffle(target_degrees)
    target_degrees = np.array(target_degrees, dtype=np.int32)
    
    # Adjust to exactly match n_bo_target
    current_bo = np.sum(target_degrees) // 2
    diff = n_bo_target - current_bo
    idx_list = list(range(n_nf))
    rng.shuffle(idx_list)
    
    if diff > 0:
        for idx in idx_list:
            if target_degrees[idx] < 4:
                target_degrees[idx] += 1
                diff -= 1
                if diff == 0: break
    elif diff < 0:
        for idx in idx_list:
            if target_degrees[idx] > 0:
                target_degrees[idx] -= 1
                diff += 1
                if diff == 0: break
                
    # Evaluate candidates
    si_si, si_p, p_p = [], [], []
    for i, j in pairs:
        ti, tj = nf_types[i], nf_types[j]
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        r = np.linalg.norm(d)
        if ti == 0 and tj == 0:
            if SI_SI_MIN <= r <= SI_SI_MAX:
                si_si.append((abs(r - 3.22) + rng.uniform(0, 0.05), i, j))
        elif ti != tj:
            if SI_P_MIN <= r <= SI_P_MAX:
                si_p.append((abs(r - 3.11) + rng.uniform(0, 0.05), i, j))
        else:
            if P_P_MIN <= r <= P_P_MAX:
                p_p.append((abs(r - 3.00) + rng.uniform(0, 0.05), i, j))
                
    si_si.sort()
    si_p.sort()
    p_p.sort()
    
    deg = np.zeros(n_nf, dtype=np.int32)
    edges = []
    selected = set()
    
    # 1. Si-O-P (limited)
    n_siop = 0
    for s, i, j in si_p:
        if n_siop >= n_siop_max: break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key); deg[i] += 1; deg[j] += 1
                edges.append((i, j)); n_siop += 1
                
    # 2. Si-O-Si with Qn bias (Pass 1: Strict)
    n_siosi = 0
    n_siosi_target = n_bo_target - n_siop
    for s, i, j in si_si:
        if n_siosi >= n_siosi_target: break
        if deg[i] < target_degrees[i] and deg[j] < target_degrees[j]:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key); deg[i] += 1; deg[j] += 1
                edges.append((i, j)); n_siosi += 1
                
    # 3. Si-O-Si (Pass 2: Fallback to reach target if strict fails)
    if n_siosi < n_siosi_target:
        for s, i, j in si_si:
            if n_siosi >= n_siosi_target: break
            if deg[i] < 4 and deg[j] < 4:
                key = (min(i, j), max(i, j))
                if key not in selected:
                    selected.add(key); deg[i] += 1; deg[j] += 1
                    edges.append((i, j)); n_siosi += 1
                    
    print(f"    Topology built: {len(edges)} BOs")
    return edges, deg


# ========================= STEP 3: PLACE BRIDGING OXYGENS =========================
def place_bridging_oxygens(nf_coords, nf_types, edges, box, rng):
    bo_coords = []
    for i, j in edges:
        ti, tj = nf_types[i], nf_types[j]
        r_i = SI_O_BOND if ti == 0 else P_O_BOND
        r_j = SI_O_BOND if tj == 0 else P_O_BOND
        
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        dist = np.linalg.norm(d)
        if dist < 1e-6: continue
        
        frac = r_i / (r_i + r_j)
        o_pos = nf_coords[i] + d * frac
        bo_coords.append(o_pos % box)
        
    return np.array(bo_coords, dtype=np.float64) if bo_coords else np.empty((0, 3))


# ========================= STEP 4: PLACE TERMINAL OXYGENS (NBO) =========================
def place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng):
    term_coords, term_owners = [], []
    n_nf = len(nf_coords)
    nf_tree = cKDTree(nf_coords, boxsize=box)
    
    all_o = list(bo_coords) if len(bo_coords) > 0 else []
    o_tree = cKDTree(np.array(all_o), boxsize=box) if all_o else None
    
    TETRA = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], dtype=np.float64)
    TETRA /= np.linalg.norm(TETRA, axis=1)[:, None]
    
    failed_nf = []
    for nf in range(n_nf):
        n_nbo_needed = max(0, 4 - int(deg[nf]))
        if n_nbo_needed == 0: continue
        
        pos = nf_coords[nf]
        bond = SI_O_BOND if nf_types[nf] == 0 else P_O_BOND
        placed_count = 0
        
        for _ in range(200):
            if placed_count >= n_nbo_needed: break
            R = random_rotation(rng)
            dirs = [(R @ TETRA[k]) for k in range(4)]
            rng.shuffle(dirs)
            
            for direction in dirs:
                if placed_count >= n_nbo_needed: break
                d = np.array(direction) + 0.08 * (rng.random(3) - 0.5)
                d = d / np.linalg.norm(d)
                o_pos = (pos + bond * d) % box
                
                nearby = nf_tree.query_ball_point(o_pos, NBO_SAFE_DIST)
                if any(idx != nf for idx in nearby): continue
                if o_tree is not None and len(o_tree.query_ball_point(o_pos, NBO_O_OVERLAP)) > 0: continue
                
                term_coords.append(o_pos); term_owners.append(nf)
                all_o.append(o_pos); placed_count += 1
                if len(all_o) % 100 == 0 and len(all_o) > 0:
                    o_tree = cKDTree(np.array(all_o), boxsize=box)
                    
        if placed_count < n_nbo_needed:
            failed_nf.append((nf, n_nbo_needed - placed_count))
            
    for nf, missing in failed_nf:
        pos = nf_coords[nf]
        bond = SI_O_BOND if nf_types[nf] == 0 else P_O_BOND
        for _ in range(missing):
            best_pos, best_d = None, 0
            for _ in range(5000):
                d = rng.normal(size=3); d /= np.linalg.norm(d)
                o_pos = (pos + bond * d) % box
                nearby = nf_tree.query_ball_point(o_pos, 5.0)
                nearby = [idx for idx in nearby if idx != nf]
                if not nearby:
                    best_pos = o_pos; break
                d_near = minimum_image(nf_coords[nearby] - o_pos, box)
                r_near = np.sqrt(np.sum(d_near*d_near, axis=1)).min()
                if r_near > best_d:
                    best_d, best_pos = r_near, o_pos
            term_coords.append(best_pos); term_owners.append(nf)
            all_o.append(best_pos)
            
    return np.array(term_coords, dtype=np.float64), np.array(term_owners, dtype=np.int32)


# ========================= STEP 5: PLACE MODIFIERS =========================
def place_modifiers(n_mod_list, sites, site_owners, nf_types, static_coords, box, rng):
    if len(sites) == 0:
        total = sum(n for _, n in n_mod_list)
        return rng.uniform(0.0, box, size=(total, 3))
    
    static_tree = cKDTree(static_coords, boxsize=box)
    target_dist = {"Na": NA_O_BOND, "Ca": CA_O_BOND, "Mg": MG_O_BOND}
    p_site_idx = [i for i, owner in enumerate(site_owners) if owner >= 0 and nf_types[owner] == 1]
    
    mod_coords = []
    for elem, n_mod in n_mod_list:
        if n_mod <= 0: continue
        for _ in range(n_mod):
            pos = None
            for _ in range(3000):
                if elem in ("Ca", "Mg") and p_site_idx and rng.random() < 0.35:
                    site = sites[p_site_idx[rng.integers(len(p_site_idx))]]
                else:
                    site = sites[rng.integers(len(sites))]
                
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                dist = target_dist[elem] + rng.uniform(-0.15, 0.25)
                cand = (site + direction * dist) % box
                
                if len(static_tree.query_ball_point(cand, 1.4)) > 0: continue
                ok = True
                if mod_coords:
                    for m in mod_coords:
                        if np.linalg.norm(minimum_image(cand - m, box)) < 2.0:
                            ok = False; break
                if ok:
                    pos = cand; break
            if pos is None: pos = rng.uniform(0.0, box, 3)
            mod_coords.append(pos)
            
    return np.array(mod_coords, dtype=np.float64)


# ========================= STEP 6: PLACE FREE OXYGENS NEAR MODIFIERS =========================
def place_free_oxygens_near_modifiers(n_fo, mod_coords, static_coords, box, rng):
    if n_fo <= 0: return np.empty((0, 3))
    static_tree = cKDTree(static_coords, boxsize=box)
    mod_tree = cKDTree(mod_coords, boxsize=box) if len(mod_coords) > 0 else None
    
    fo_coords = []
    for _ in range(n_fo):
        pos = None
        for _ in range(20000):
            if mod_tree is not None and rng.random() < 0.8:
                i = rng.integers(len(mod_coords))
                nb = mod_tree.query_ball_point(mod_coords[i], 4.5)
                nb = [j for j in nb if j != i]
                if not nb: continue
                j = nb[rng.integers(len(nb))]
                mid = 0.5 * (mod_coords[i] + mod_coords[j])
                cand = (mid + rng.normal(0, 0.2, 3)) % box
            else:
                cand = rng.uniform(0.0, box, 3)
            
            if len(static_tree.query_ball_point(cand, 2.0)) > 0: continue
            if mod_tree is not None:
                d = mod_coords - cand
                d = minimum_image(d, box)
                r = np.sqrt(np.sum(d*d, axis=1))
                if np.sum(r < 2.7) < 2: continue
            pos = cand; break
        if pos is None: pos = rng.uniform(0.0, box, 3)
        fo_coords.append(pos)
        
    return np.array(fo_coords, dtype=np.float64)


# ========================= STEP 7: SAFE RELAXATION =========================
def relax_o_overlaps_safe(coords, types, box, fixed_mask, iterations=200, step=0.05):
    new_coords = coords.copy()
    r_hard = np.full((6, 6), 0.9)
    r_hard[4, 4] = 1.95   # O-O
    r_hard[0, 4] = r_hard[4, 0] = 1.50  # Si-O
    r_hard[3, 4] = r_hard[4, 3] = 1.40  # P-O
    r_hard[1, 4] = r_hard[4, 1] = 1.50  # Ca-O
    r_hard[2, 4] = r_hard[4, 2] = 1.60  # Na-O
    r_hard[5, 4] = r_hard[4, 5] = 1.40  # Mg-O
    
    for it in range(iterations):
        tree = cKDTree(new_coords, boxsize=box)
        pairs = tree.query_pairs(2.5, output_type='ndarray')
        moved = False
        for i, j in pairs:
            if fixed_mask[i] and fixed_mask[j]: continue
            d = minimum_image(new_coords[i] - new_coords[j], box)
            r = np.linalg.norm(d)
            limit = r_hard[types[i], types[j]]
            if r < limit and r > 1e-6:
                moved = True
                f = d * (step * (limit - r) / r)
                if not fixed_mask[i]: new_coords[i] = (new_coords[i] + f) % box
                if not fixed_mask[j]: new_coords[j] = (new_coords[j] - f) % box
        if not moved:
            print(f"    Relaxation converged after {it+1} iterations")
            break
    return new_coords


# ========================= VERIFICATION =========================
def verify_structure(symbols, coords, box, target_nc):
    print("\n" + "=" * 60)
    print("  STRUCTURE VERIFICATION")
    print("=" * 60)
    tree = cKDTree(coords, boxsize=box)
    
    si_idx = [i for i, s in enumerate(symbols) if s == "Si"]
    p_idx = [i for i, s in enumerate(symbols) if s == "P"]
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]
    
    si_cn = [sum(1 for j in tree.query_ball_point(coords[si], 2.25) if j != si and symbols[j] == "O") for si in si_idx]
    p_cn = [sum(1 for j in tree.query_ball_point(coords[p], 2.25) if j != p and symbols[j] == "O") for p in p_idx]
    
    print(f"  Si-O CN: {np.mean(si_cn):.3f} (target: 4.0)")
    print(f"  P-O CN:  {np.mean(p_cn):.3f} (target: 4.0)")
    
    o_nf_count = []
    for o in o_idx:
        nf_count = sum(1 for j in tree.query_ball_point(coords[o], 2.25) if j != o and symbols[j] in ["Si", "P"] and np.linalg.norm(minimum_image(coords[o] - coords[j], box)) < 2.25)
        o_nf_count.append(nf_count)
        
    fo = sum(1 for c in o_nf_count if c == 0)
    nbo = sum(1 for c in o_nf_count if c == 1)
    bo = sum(1 for c in o_nf_count if c == 2)
    to = sum(1 for c in o_nf_count if c >= 3)
    n_o = max(1, len(o_idx))
    
    print(f"\n--- Oxygen Speciation ---")
    print(f"  FO:  {fo:5d} ({fo/n_o*100:5.2f}%)")
    print(f"  NBO: {nbo:5d} ({nbo/n_o*100:5.2f}%)")
    print(f"  BO:  {bo:5d} ({bo/n_o*100:5.2f}%)")
    print(f"  TO:  {to:5d} ({to/n_o*100:5.2f}%)")
    
    bridging_set = {o_idx[i] for i, c in enumerate(o_nf_count) if c >= 2}
    def count_bo_around(center_idx):
        total_bo = 0
        for c in center_idx:
            bo_count = sum(1 for j in tree.query_ball_point(coords[c], 2.25) if j != c and symbols[j] == "O" and np.linalg.norm(minimum_image(coords[c] - coords[j], box)) < 2.25 and j in bridging_set)
            total_bo += min(bo_count, 4)
        return total_bo
        
    nc_comb = (count_bo_around(si_idx) + count_bo_around(p_idx)) / max(1, len(si_idx) + len(p_idx))
    print(f"\n  NC Combined: {nc_comb:.3f} (target: {target_nc:.3f})")
    print("=" * 60)
    return nc_comb


# ========================= MAIN =========================
def generate_structure(mg_label, output_file, seed=42, scale=4):
    entry = PAPER_COMPOSITION[mg_label]
    rng = np.random.default_rng(seed)
    
    N_SI = 461 * scale
    N_P = 52 * scale
    N_NA = 488 * scale
    N_CA = entry["N_Ca"] * scale
    N_MG = entry["N_Mg"] * scale
    N_O = 1565 * scale
    
    print(f"\nComposition for {entry['label']} (scale={scale}x)")
    
    density = entry["density"]
    masses = {"Si": 28.0855, "Ca": 40.078, "Na": 22.98977, "P": 30.97376, "O": 15.999, "Mg": 24.305}
    total_mass = (N_SI * masses["Si"] + N_P * masses["P"] + N_NA * masses["Na"] + N_CA * masses["Ca"] + N_MG * masses["Mg"] + N_O * masses["O"])
    box = ((total_mass / NA) / density * 1e24) ** (1/3)
    
    N_NF = N_SI + N_P
    target_nc = PAPER_NC[mg_label]
    n_bo_target = int(round(target_nc * N_NF / 2.0))
    n_siop_max = min(int(0.02 * N_O), n_bo_target, 2 * N_P)
    qn_target = PAPER_QN[mg_label]
    
    print(f"  Box: {box:.4f} A, Target NC: {target_nc}, Target BOs: {n_bo_target}")
    
    print("\n[1/7] Placing Network Formers...")
    nf_coords, nf_types = place_network_formers(N_SI, N_P, box, rng)
    
    print("[2/7] Building Topology with Qn target...")
    edges, deg = build_topology_with_qn(nf_coords, nf_types, box, n_bo_target, n_siop_max, qn_target, rng)
    
    print("[3/7] Placing Bridging Oxygens...")
    bo_coords = place_bridging_oxygens(nf_coords, nf_types, edges, box, rng)
    
    print("[4/7] Placing Terminal Oxygens (NBO)...")
    term_coords, term_owners = place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng)
    
    print("[5/7] Placing Modifiers...")
    n_fo = N_O - len(term_coords) - len(bo_coords)
    if n_fo < 0:
        print("  WARNING: Too many O, truncating BO")
        bo_coords = bo_coords[:N_O - len(term_coords)]
        n_fo = 0
        
    o_coords_no_fo = np.vstack([term_coords, bo_coords])
    site_owners = np.concatenate([term_owners, -1 * np.ones(len(bo_coords), dtype=np.int32)])
    
    mod_coords = place_modifiers(
        [("Na", N_NA), ("Ca", N_CA), ("Mg", N_MG)],
        o_coords_no_fo, site_owners, nf_types, 
        np.vstack([nf_coords, o_coords_no_fo]), box, rng
    )
    
    print("[6/7] Placing Free Oxygens (FO) near Modifiers...")
    static_for_fo = np.vstack([nf_coords, o_coords_no_fo, mod_coords])
    fo_coords = place_free_oxygens_near_modifiers(n_fo, mod_coords, static_for_fo, box, rng)
    print(f"  NBO: {len(term_coords)}, BO: {len(bo_coords)}, FO: {len(fo_coords)}")
    
    print("[7/7] Safe Relaxation...")
    o_coords = np.vstack([term_coords, bo_coords, fo_coords])
    all_coords = np.vstack([nf_coords, o_coords, mod_coords])
    
    all_types = np.concatenate([
        np.zeros(N_SI + N_P, dtype=np.int32),
        np.full(len(o_coords), 4, dtype=np.int32),
        np.full(len(mod_coords), 1, dtype=np.int32)
    ])
    all_types[:N_SI + N_P] = nf_types
    mod_start = N_SI + N_P + len(o_coords)
    all_types[mod_start:mod_start + N_NA] = 2
    all_types[mod_start + N_NA:mod_start + N_NA + N_CA] = 1
    all_types[mod_start + N_NA + N_CA:] = 5
    
    fixed_mask = np.zeros(len(all_coords), dtype=bool)
    fixed_mask[:N_NF] = True
    fixed_mask[N_NF:N_NF + len(term_coords)] = False
    fixed_mask[N_NF + len(term_coords):N_NF + len(term_coords) + len(bo_coords)] = True
    
    relaxed_coords = relax_o_overlaps_safe(all_coords, all_types, box, fixed_mask)
    
    symbols = (["Si"] * N_SI + ["P"] * N_P + ["O"] * len(o_coords) + ["Na"] * N_NA + ["Ca"] * N_CA + ["Mg"] * N_MG)
    perm = rng.permutation(len(symbols))
    symbols = [symbols[i] for i in perm]
    final_coords = relaxed_coords[perm]
    
    verify_structure(symbols, final_coords, box, target_nc)
    
    out_path = Path(output_file)
    with open(out_path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"Mg-doped 45S5 PATCHED v2, label={entry['label']}, x={mg_label}, N={len(symbols)}, rho={density:.6f}, box={box:.6f}, seed={seed}\n")
        for i in range(len(symbols)):
            f.write(f"{symbols[i]:2s} {final_coords[i,0]:12.6f} {final_coords[i,1]:12.6f} {final_coords[i,2]:12.6f}\n")
    
    print(f"\n[OK] Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mg-doped 45S5 Structure Generator (FINAL PATCHED v2)")
    parser.add_argument("--mg", type=int, default=0, choices=sorted(PAPER_COMPOSITION.keys()))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    args = parser.parse_args()
    
    if args.out is None:
        args.out = f"initial_Mg{args.mg}.xyz"
    
    generate_structure(args.mg, args.out, args.seed, args.scale)
    
    
    