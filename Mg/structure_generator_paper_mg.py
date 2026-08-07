#!/usr/bin/env python3
"""
========================================================================
Paper-Matched Structure Generator for Mg-doped 45S5 (v5.3 - Exact BO Control)
Based on: Moghanian et al., Mg-doped 45S5 bioglass manuscript

v5.3 KEY FIX:
- EXACTLY builds target number of BO (1976 for 45-M0)
- Strict control on Si-Si and Si-P edge counts
- Simplified, predictable algorithm
========================================================================
"""
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial import cKDTree
from math import sqrt, sin, cos, pi

NA = 6.02214076e23

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

def minimum_image(dr, box):
    return dr - box * np.round(dr / box)

def random_rotation(rng):
    u1, u2, u3 = rng.random(3)
    q0 = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    q1 = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    q2 = sqrt(u1) * sin(2.0 * pi * u3)
    q3 = sqrt(u1) * cos(2.0 * pi * u3)
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (q2*q2 + q3*q3); R[0, 1] = 2.0 * (q1*q2 - q0*q3); R[0, 2] = 2.0 * (q1*q3 + q0*q2)
    R[1, 0] = 2.0 * (q1*q2 + q0*q3); R[1, 1] = 1.0 - 2.0 * (q1*q1 + q3*q3); R[1, 2] = 2.0 * (q2*q3 - q0*q1)
    R[2, 0] = 2.0 * (q1*q3 - q0*q2); R[2, 1] = 2.0 * (q2*q3 + q0*q1); R[2, 2] = 1.0 - 2.0 * (q1*q1 + q2*q2)
    return R

def place_network_formers(n_si, n_p, box, min_nf_dist, min_p_p_dist, rng):
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    types = np.zeros(n_total, dtype=np.int32)
    min_d_sq = min_nf_dist * min_nf_dist
    min_p_sq = min_p_p_dist * min_p_p_dist
    placed = 0
    
    for i in range(n_total):
        is_p = (i >= n_si)
        success = False
        
        for attempt in range(30000):
            pos = rng.uniform(0.0, box, 3)
            valid = True
            
            if placed > 0:
                d = coords[:placed] - pos
                d = minimum_image(d, box)
                r2 = np.sum(d * d, axis=1)
                
                if np.any(r2 < min_d_sq):
                    valid = False
                elif valid and is_p:
                    p_mask = (types[:placed] == 1)
                    if np.any(p_mask) and np.any(r2[p_mask] < min_p_sq):
                        valid = False
            
            if valid:
                coords[placed] = pos
                types[placed] = 1 if is_p else 0
                placed += 1
                success = True
                break
        
        if not success:
            coords[placed] = rng.uniform(0.0, box, 3)
            types[placed] = 1 if is_p else 0
            placed += 1
    
    return coords, types

def build_topology_exact(nf_coords, nf_types, box, n_siosi_target, n_siop_target, rng):
    """Build EXACTLY the target number of BO edges."""
    tree = cKDTree(nf_coords, boxsize=box)
    # Tighter distance range: 2.8 to 4.5 Å
    pairs = tree.query_pairs(4.5, output_type="ndarray")
    
    si_si_cand = []
    si_p_cand = []
    
    for i, j in pairs:
        ti, tj = nf_types[i], nf_types[j]
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        r = np.linalg.norm(d)
        
        # Tighter range: 2.8 to 4.5 Å
        if r < 2.8 or r > 4.5:
            continue
        
        if ti == 0 and tj == 0:
            si_si_cand.append((r + rng.uniform(0, 0.05), i, j))
        elif ti != tj:
            si_p_cand.append((r + rng.uniform(0, 0.05), i, j))
    
    si_si_cand.sort(key=lambda x: x[0])
    si_p_cand.sort(key=lambda x: x[0])
    
    deg = np.zeros(len(nf_coords), dtype=np.int32)
    selected = set()
    si_si_edges = []
    si_p_edges = []
    
    # EXACTLY n_siop_target Si-P edges
    for score, i, j in si_p_cand:
        if len(si_p_edges) >= n_siop_target:
            break
        
        p_idx = i if nf_types[i] == 1 else j
        si_idx = j if nf_types[i] == 1 else i
        
        # P should have max 2 BO, Si should have max 4
        if deg[si_idx] < 4 and deg[p_idx] < 2:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key)
                deg[i] += 1
                deg[j] += 1
                si_p_edges.append((i, j))
    
    # EXACTLY n_siosi_target Si-Si edges
    for score, i, j in si_si_cand:
        if len(si_si_edges) >= n_siosi_target:
            break
        
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key)
                deg[i] += 1
                deg[j] += 1
                si_si_edges.append((i, j))
    
    return si_si_edges, si_p_edges, deg

def place_bridging_oxygens(nf_coords, nf_types, edges, box, rng):
    bo_coords = []
    
    for i, j in edges:
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        r = np.linalg.norm(d)
        
        if r < 1e-6:
            continue
        
        o_pos = (nf_coords[i] + d * 0.5) % box
        
        if r > 2.8:
            perp = np.cross(d, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp) < 0.1:
                perp = np.cross(d, np.array([0.0, 1.0, 0.0]))
            perp = perp / np.linalg.norm(perp)
            offset = rng.uniform(-0.15, 0.15)
            o_pos = (o_pos + perp * offset) % box
        
        bo_coords.append(o_pos)
    
    return np.array(bo_coords, dtype=np.float64) if bo_coords else np.empty((0, 3), dtype=np.float64)

def place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng):
    term_coords = []
    term_owners = []
    
    o_tree = cKDTree(bo_coords, boxsize=box) if len(bo_coords) > 0 else None
    nf_tree = cKDTree(nf_coords, boxsize=box)
    
    TETRA = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=np.float64)
    for i in range(4):
        TETRA[i] /= np.linalg.norm(TETRA[i])
    
    for nf in range(len(nf_coords)):
        keep = max(0, 4 - int(deg[nf]))
        if keep == 0:
            continue
        
        pos = nf_coords[nf]
        bond = 1.61 if nf_types[nf] == 0 else 1.50
        R = random_rotation(rng)
        
        candidates = []
        for k in range(4):
            v = R @ TETRA[k]
            v += 0.05 * (rng.random(3) - 0.5)
            v /= np.linalg.norm(v)
            candidates.append(v)
        
        rng.shuffle(candidates)
        
        placed_count = 0
        for v in candidates:
            if placed_count >= keep:
                break
            
            o_pos = (pos + bond * v) % box
            
            if o_tree is not None and len(o_tree.query_ball_point(o_pos, 1.85)) > 0:
                continue
            
            overlap = False
            for prev_o in term_coords:
                if np.linalg.norm(minimum_image(o_pos - prev_o, box)) < 1.85:
                    overlap = True
                    break
            if overlap:
                continue
            
            nearby_nf = nf_tree.query_ball_point(o_pos, 2.30)
            too_close = False
            for nf_idx in nearby_nf:
                if nf_idx == nf:
                    continue
                if np.linalg.norm(minimum_image(o_pos - nf_coords[nf_idx], box)) < 2.30:
                    too_close = True
                    break
            if too_close:
                continue
            
            term_coords.append(o_pos)
            term_owners.append(nf)
            placed_count += 1
        
        while placed_count < keep:
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            o_pos = (pos + bond * v) % box
            term_coords.append(o_pos)
            term_owners.append(nf)
            placed_count += 1
    
    return (np.array(term_coords, dtype=np.float64) if term_coords 
            else np.empty((0, 3), dtype=np.float64), 
            np.array(term_owners, dtype=np.int32))

def place_free_oxygens(n_fo, static_coords, box, rng):
    if n_fo <= 0:
        return np.empty((0, 3), dtype=np.float64)
    
    tree = cKDTree(static_coords, boxsize=box) if len(static_coords) > 0 else None
    fo_coords = []
    
    for _ in range(n_fo):
        placed = False
        
        for attempt in range(8000):
            pos = rng.uniform(0.0, box, 3)
            
            if tree is not None and len(tree.query_ball_point(pos, 2.1)) > 0:
                continue
            
            ok = True
            for p in fo_coords:
                if np.linalg.norm(minimum_image(pos - p, box)) < 1.9:
                    ok = False
                    break
            
            if ok:
                fo_coords.append(pos)
                placed = True
                break
        
        if not placed:
            fo_coords.append(rng.uniform(0.0, box, 3))
    
    return np.array(fo_coords, dtype=np.float64)

def place_modifiers(n_mod_list, sites, site_owners, nf_types, static_coords, box, rng):
    if len(sites) == 0:
        total = sum(n for _, n in n_mod_list)
        return rng.uniform(0.0, box, size=(total, 3))
    
    static_tree = cKDTree(static_coords, boxsize=box)
    target_dist = {"Na": 2.40, "Ca": 2.36, "Mg": 2.00}
    
    p_site_idx = [
        i for i, owner in enumerate(site_owners) 
        if owner >= 0 and nf_types[owner] == 1
    ]
    
    mod_coords = []
    
    for elem, n_mod in n_mod_list:
        if n_mod <= 0:
            continue
        
        for _ in range(n_mod):
            pos = None
            
            for attempt in range(2000):
                if elem in ("Ca", "Mg") and len(p_site_idx) > 0 and rng.random() < 0.35:
                    site = sites[p_site_idx[rng.integers(len(p_site_idx))]]
                else:
                    site = sites[rng.integers(len(sites))]
                
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                dist = target_dist[elem] + rng.uniform(-0.15, 0.25)
                cand = (site + direction * dist) % box
                
                if len(static_tree.query_ball_point(cand, 1.4)) > 0:
                    continue
                
                ok = True
                if len(mod_coords) > 0:
                    arr = np.array(mod_coords)
                    d = minimum_image(arr - cand, box)
                    if np.any(np.sum(d * d, axis=1) < 4.0):
                        ok = False
                
                if ok:
                    pos = cand
                    break
            
            if pos is None:
                pos = rng.uniform(0.0, box, 3)
            
            mod_coords.append(pos)
    
    return np.array(mod_coords, dtype=np.float64)

def hard_sphere_relaxation_fast(coords, types, box, iterations=80, step=0.10, r_max=2.5):
    """Fast repulsive relaxation using KD-tree."""
    new_coords = coords.copy()
    
    r_hard = np.full((6, 6), 0.9, dtype=np.float64)
    r_hard[4, 4] = 1.90  # O-O
    r_hard[0, 4] = 1.30; r_hard[4, 0] = 1.30  # Si-O
    r_hard[3, 4] = 1.30; r_hard[4, 3] = 1.30  # P-O
    r_hard[5, 4] = 1.20; r_hard[4, 5] = 1.20  # Mg-O
    r_hard[1, 4] = 1.5; r_hard[4, 1] = 1.5    # Ca-O
    r_hard[2, 4] = 1.5; r_hard[4, 2] = 1.5    # Na-O
    
    for it in range(iterations):
        tree = cKDTree(new_coords, boxsize=box)
        pairs = tree.query_pairs(r_max, output_type='ndarray')
        
        moved = False
        for i, j in pairs:
            d = minimum_image(new_coords[i] - new_coords[j], box)
            r = np.linalg.norm(d)
            limit = r_hard[types[i], types[j]]
            
            if r < limit and r > 1e-6:
                moved = True
                force_mag = step * (limit - r) / r
                f = d * force_mag
                new_coords[i] = (new_coords[i] + f) % box
                new_coords[j] = (new_coords[j] - f) % box
        
        if not moved:
            print(f"    Converged after {it + 1} iterations")
            break
    
    return new_coords

def verify_structure(symbols, coords, box, targets):
    print("\n" + "=" * 70)
    print("  STRUCTURE VERIFICATION REPORT (v5.3 - Exact BO Control)")
    print("=" * 70)
    
    tree = cKDTree(coords, boxsize=box)
    
    print("\n--- Close Contact Check ---")
    checks = [
        ("Si-Si", "Si", "Si", 2.4),
        ("O-O", "O", "O", 1.7),
        ("Si-O", "Si", "O", 1.2),
        ("P-O", "P", "O", 1.2),
        ("Mg-O", "Mg", "O", 1.2),
        ("P-P", "P", "P", 3.0),
    ]
    
    for name, e1, e2, threshold in checks:
        close = 0
        pairs = tree.query_pairs(threshold, output_type="ndarray")
        
        for i, j in pairs:
            if (symbols[i] == e1 and symbols[j] == e2) or \
               (symbols[i] == e2 and symbols[j] == e1):
                close += 1
        
        status = "✅ PASS" if close == 0 else f"⚠️ {close} violations"
        print(f"  {name:8s} < {threshold:.1f} A: {status}")
    
    si_idx = [i for i, s in enumerate(symbols) if s == "Si"]
    p_idx = [i for i, s in enumerate(symbols) if s == "P"]
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]
    
    si_cn = [
        sum(1 for j in tree.query_ball_point(coords[si], 2.25) 
            if j != si and symbols[j] == "O") 
        for si in si_idx
    ]
    p_cn = [
        sum(1 for j in tree.query_ball_point(coords[p], 2.25) 
            if j != p and symbols[j] == "O") 
        for p in p_idx
    ]
    
    print("\n--- Network Former CN ---")
    print(f"  Si-O CN mean: {np.mean(si_cn):.3f} (Target: 4.0)")
    print(f"  P-O  CN mean: {np.mean(p_cn):.3f} (Target: 4.0)")
    
    o_nf_count = []
    for o in o_idx:
        nf_count = sum(
            1 for j in tree.query_ball_point(coords[o], 2.25)
            if j != o and symbols[j] in ["Si", "P"] and 
            np.linalg.norm(minimum_image(coords[o] - coords[j], box)) < 2.25
        )
        o_nf_count.append(nf_count)
    
    fo = sum(1 for c in o_nf_count if c == 0)
    nbo = sum(1 for c in o_nf_count if c == 1)
    bo = sum(1 for c in o_nf_count if c == 2)
    to = sum(1 for c in o_nf_count if c >= 3)
    n_o = max(1, len(o_idx))
    
    print("\n--- Oxygen Speciation ---")
    print(f"  FO : {fo:6d}  ({fo / n_o * 100.0:6.2f}%)   target FO : {targets['fo_pct']:.2f}%")
    print(f"  NBO: {nbo:6d}  ({nbo / n_o * 100.0:6.2f}%)   target NBO: {targets['nbo_pct']:.2f}%")
    print(f"  BO : {bo:6d}  ({bo / n_o * 100.0:6.2f}%)   target BO : {targets['bo_pct']:.2f}%")
    print(f"  TO : {to:6d}  ({to / n_o * 100.0:6.2f}%)")
    
    bridging_set = {o_idx[i] for i, c in enumerate(o_nf_count) if c >= 2}
    
    def qn_for(center_idx):
        return [
            sum(
                1 for j in tree.query_ball_point(coords[c], 2.25)
                if j != c and symbols[j] == "O" and 
                np.linalg.norm(minimum_image(coords[c] - coords[j], box)) < 2.25 and 
                j in bridging_set
            )
            for c in center_idx
        ]
    
    qn_si = qn_for(si_idx)
    qn_p = qn_for(p_idx)
    
    n_si, n_p = max(1, len(si_idx)), max(1, len(p_idx))
    n_net = n_si + n_p
    
    nc_si = sum(min(q, 4) for q in qn_si) / n_si
    nc_p = sum(min(q, 4) for q in qn_p) / n_p
    nc_comb = (sum(min(q, 4) for q in qn_si) + sum(min(q, 4) for q in qn_p)) / n_net
    
    print("\n--- Network Connectivity ---")
    print(f"  NC Si            : {nc_si:.3f}")
    print(f"  NC P             : {nc_p:.3f}")
    print(f"  NC Si-P Combined : {nc_comb:.3f}   target: {targets['nc']:.3f}")
    print("=" * 70)

def generate_structure(mg_label, density_override, output_file, seed, scale=4):
    if mg_label not in PAPER_COMPOSITION:
        raise ValueError(f"Invalid --mg {mg_label}")
    
    entry = PAPER_COMPOSITION[mg_label]
    rng = np.random.default_rng(seed)
    
    N_SI, N_P, N_NA = 461 * scale, 52 * scale, 488 * scale
    N_CA, N_MG, N_O = entry["N_Ca"] * scale, entry["N_Mg"] * scale, 1565 * scale
    
    counts = {
        "Si": N_SI, "P": N_P, "Na": N_NA, 
        "Ca": N_CA, "Mg": N_MG, "O": N_O
    }
    
    print(f"\nComposition for {entry['label']} (SCALE={scale}x): TOTAL: {sum(counts.values())}")
    
    density = entry["density"] if density_override is None else float(density_override)
    masses = {
        "Si": 28.0855, "Ca": 40.078, "Na": 22.98977, 
        "P": 30.97376, "O": 15.999, "Mg": 24.305
    }
    total_mass = sum(masses[s] * c for s, c in counts.items())
    box = ((total_mass / NA) / density * 1.0e24) ** (1.0 / 3.0)
    
    print(f"Box size: {box:.6f} A (Density: {density:.6f} g/cm3)")
    
    N_NF = N_SI + N_P
    target_nc = PAPER_NC[mg_label]
    
    # EXACT target counts
    n_bo_target = int(round(target_nc * N_NF / 2.0))
    n_siop_target = min(125, n_bo_target)  # Max 125 Si-O-P
    n_siosi_target = n_bo_target - n_siop_target
    
    print("\nTopology targets (EXACT):")
    print(f"  Target NC: {target_nc:.3f} | Target BO: {n_bo_target} | Si-O-Si: {n_siosi_target} | Si-O-P: {n_siop_target}")
    
    # Stage 1: Network formers
    print("\n[Stage 1/7] Placing network formers (P-P > 3.5 A)...")
    nf_coords, nf_types = place_network_formers(
        N_SI, N_P, box, min_nf_dist=2.7, min_p_p_dist=3.5, rng=rng
    )
    
    # Stage 2: Build EXACT topology
    print("[Stage 2/7] Building EXACT topology (target BO count)...")
    si_si_edges, si_p_edges, deg = build_topology_exact(
        nf_coords, nf_types, box, n_siosi_target, n_siop_target, rng
    )
    print(f"  Built Si-O-Si: {len(si_si_edges)} (target: {n_siosi_target})")
    print(f"  Built Si-O-P: {len(si_p_edges)} (target: {n_siop_target})")
    all_edges = si_si_edges + si_p_edges
    
    # Stage 3: Bridging oxygens
    print("[Stage 3/7] Placing Bridging Oxygens (BO)...")
    bo_coords = place_bridging_oxygens(nf_coords, nf_types, all_edges, box, rng)
    
    # Stage 4: Terminal oxygens
    print("[Stage 4/7] Placing Terminal Oxygens (NBO) with strict distance check...")
    term_coords, term_owners = place_terminal_oxygens(
        nf_coords, nf_types, deg, box, bo_coords, rng
    )
    
    # Stage 5: Free oxygens
    print("[Stage 5/7] Placing Free Oxygens (FO)...")
    n_fo = N_O - len(term_coords) - len(bo_coords)
    if n_fo < 0:
        print(f"  WARNING: Too many NBO+BO, truncating BO")
        bo_coords = bo_coords[:N_O - len(term_coords)]
        n_fo = 0
    
    static_for_fo = (np.vstack([nf_coords, term_coords, bo_coords]) 
                     if len(term_coords) > 0 
                     else np.vstack([nf_coords, bo_coords]))
    fo_coords = place_free_oxygens(n_fo, static_for_fo, box, rng)
    
    o_coords = (np.vstack([term_coords, bo_coords, fo_coords]) 
                if len(term_coords) > 0 
                else np.vstack([bo_coords, fo_coords]))
    
    sites = o_coords.copy()
    site_owners = np.concatenate([
        term_owners, 
        -1 * np.ones(len(bo_coords) + len(fo_coords), dtype=np.int32)
    ])
    
    print(f"  Terminal/NBO: {len(term_coords)} | Bridging: {len(bo_coords)} | Free: {n_fo}")
    
    # Stage 6: Modifiers
    print("[Stage 6/7] Placing modifiers near NBO/FO sites...")
    static_coords = np.vstack([nf_coords, o_coords])
    mod_coords = place_modifiers(
        [("Na", N_NA), ("Ca", N_CA), ("Mg", N_MG)], 
        sites, site_owners, nf_types, static_coords, box, rng
    )
    
    # Assembly
    symbols_list = (["Si"] * N_SI + ["P"] * N_P + ["O"] * N_O + 
                    ["Na"] * N_NA + ["Ca"] * N_CA + ["Mg"] * N_MG)
    
    all_coords_pre = np.vstack([nf_coords, o_coords, mod_coords])
    
    # Hard sphere relaxation
    print("\n[Stage 7/7] Applying hard sphere relaxation to fix O-O overlap...")
    type_map_int = {"Si": 0, "Ca": 1, "Na": 2, "P": 3, "O": 4, "Mg": 5}
    all_types = np.array([type_map_int[s] for s in symbols_list], dtype=np.int32)
    
    relaxed_coords = hard_sphere_relaxation_fast(all_coords_pre, all_types, box, iterations=80, step=0.10, r_max=2.5)
    
    # Shuffle
    final_coords = relaxed_coords
    symbols = symbols_list.copy()
    perm = rng.permutation(len(symbols))
    symbols = [symbols[i] for i in perm]
    final_coords = final_coords[perm]
    
    # Verification
    targets = {
        "nc": target_nc, "fo_pct": 0.45, 
        "nbo_pct": 67.99, "bo_pct": 31.57
    }
    verify_structure(symbols, final_coords, box, targets)
    
    # Write XYZ
    out_path = Path(output_file)
    with open(out_path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(
            f"Mg-doped 45S5 v5.3 Exact BO, label={entry['label']}, "
            f"x={mg_label}, N={len(symbols)}, rho={density:.6f} g/cm3, "
            f"box={box:.6f} A, seed={seed}\n"
        )
        for i in range(len(symbols)):
            f.write(
                f"{symbols[i]:2s} {final_coords[i, 0]:12.6f} "
                f"{final_coords[i, 1]:12.6f} {final_coords[i, 2]:12.6f}\n"
            )
    
    print(f"\n[OK] Paper-matched initial structure saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper-matched structure generator v5.3 (Exact BO Control)"
    )
    parser.add_argument("--mg", type=int, default=0, 
                        choices=sorted(PAPER_COMPOSITION.keys()))
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    args = parser.parse_args()
    
    if args.out is None:
        args.out = f"initial_Mg{args.mg}_v5.xyz"
    
    generate_structure(args.mg, args.density, args.out, args.seed, args.scale)
    
    