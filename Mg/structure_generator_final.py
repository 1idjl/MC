#!/usr/bin/env python3
"""
========================================================================
Structure Generator for Mg-doped 45S5 Bioactive Glass (FINAL)
Based on: Moghanian et al. manuscript

Strategy:
1. Place Network Formers (Si, P) with proper spacing
2. Build exact number of BOs from paper NC target
3. Place NBOs at safe distance from ALL other NFs (prevents fake BOs)
4. Place FOs
5. Place Modifiers
6. Relax O-O overlaps

Usage:
python structure_generator_final.py --mg 0 --scale 4
python structure_generator_final.py --mg 5 --scale 4
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

# Network Connectivity targets from paper
PAPER_NC = {
    0: 1.926, 1: 1.922, 3: 1.938, 5: 1.953,
    8: 1.953, 10: 1.981, 15: 1.973, 20: 2.035,
}

# Bond lengths (Angstrom)
SI_O_BOND = 1.61
P_O_BOND = 1.50
NA_O_BOND = 2.40
CA_O_BOND = 2.36
MG_O_BOND = 2.00

# Safety distance: NBO must be farther than this from other NFs
NBO_SAFE_DIST = 2.30  # > 2.25 cutoff used in analysis


def minimum_image(dr, box):
    return dr - box * np.round(dr / box)


def random_rotation(rng):
    """Generate a random 3D rotation matrix."""
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
    """Place Si and P atoms with minimum distance constraints."""
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    types = np.zeros(n_total, dtype=np.int32)  # 0=Si, 1=P
    
    min_si_dist = 2.8   # Minimum Si-Si distance
    min_p_dist = 3.5    # Minimum P-P distance (P isolation)
    min_si_p_dist = 2.9 # Minimum Si-P distance
    
    placed = 0
    
    for i in range(n_total):
        is_p = (i >= n_si)
        success = False
        
        for attempt in range(50000):
            pos = rng.uniform(0.0, box, 3)
            valid = True
            
            if placed > 0:
                d = coords[:placed] - pos
                d = minimum_image(d, box)
                r = np.sqrt(np.sum(d * d, axis=1))
                
                if is_p:
                    # P must be far from all other P atoms
                    p_mask = types[:placed] == 1
                    if np.any(p_mask) and np.min(r[p_mask]) < min_p_dist:
                        valid = False
                    # P must be reasonably far from Si
                    si_mask = types[:placed] == 0
                    if np.any(si_mask) and np.min(r[si_mask]) < min_si_p_dist:
                        valid = False
                else:
                    # Si must be far from other Si
                    si_mask = types[:placed] == 0
                    if np.any(si_mask) and np.min(r[si_mask]) < min_si_dist:
                        valid = False
            
            if valid:
                coords[placed] = pos
                types[placed] = 1 if is_p else 0
                placed += 1
                success = True
                break
        
        if not success:
            # Fallback: place randomly
            coords[placed] = rng.uniform(0.0, box, 3)
            types[placed] = 1 if is_p else 0
            placed += 1
    
    return coords, types


# ========================= STEP 2: BUILD TOPOLOGY =========================
def build_topology(nf_coords, nf_types, box, n_bo_target, n_siop_max, rng):
    """
    Build BO topology.
    - n_bo_target: total number of BOs needed (from NC target)
    - n_siop_max: maximum Si-O-P bonds (keep P mostly isolated)
    """
    n_nf = len(nf_coords)
    tree = cKDTree(nf_coords, boxsize=box)
    
    # Find all pairs within bonding distance
    pairs = tree.query_pairs(4.5, output_type="ndarray")
    
    si_si_cand = []
    si_p_cand = []
    
    for i, j in pairs:
        ti, tj = nf_types[i], nf_types[j]
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        r = np.linalg.norm(d)
        
        # Only consider pairs in valid bonding range
        if r < 2.8 or r > 4.2:
            continue
        
        if ti == 0 and tj == 0:
            # Si-O-Si: ideal distance ~3.22 (2 x 1.61)
            score = abs(r - 3.22) + rng.uniform(0, 0.1)
            si_si_cand.append((score, i, j))
        elif ti != tj:
            # Si-O-P: ideal distance ~3.11 (1.61 + 1.50)
            score = abs(r - 3.11) + rng.uniform(0, 0.1)
            si_p_cand.append((score, i, j))
    
    # Sort by score (prefer ideal distances)
    si_si_cand.sort(key=lambda x: x[0])
    si_p_cand.sort(key=lambda x: x[0])
    
    # Select edges
    deg = np.zeros(n_nf, dtype=np.int32)
    selected = set()
    edges = []
    
    # First: Si-O-P (limited)
    n_siop = 0
    for score, i, j in si_p_cand:
        if n_siop >= n_siop_max:
            break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key)
                deg[i] += 1
                deg[j] += 1
                edges.append((i, j))
                n_siop += 1
    
    # Then: Si-O-Si (fill remaining)
    n_siosi = 0
    n_siosi_target = n_bo_target - n_siop
    
    for score, i, j in si_si_cand:
        if n_siosi >= n_siosi_target:
            break
        if deg[i] < 4 and deg[j] < 4:
            key = (min(i, j), max(i, j))
            if key not in selected:
                selected.add(key)
                deg[i] += 1
                deg[j] += 1
                edges.append((i, j))
                n_siosi += 1
    
    print(f"    Built {n_siosi} Si-O-Si + {n_siop} Si-O-P = {len(edges)} BOs")
    
    return edges, deg


# ========================= STEP 3: PLACE BRIDGING OXYGENS =========================
def place_bridging_oxygens(nf_coords, nf_types, edges, box, rng):
    """Place BO between each bonded pair."""
    bo_coords = []
    
    for i, j in edges:
        ti, tj = nf_types[i], nf_types[j]
        
        # Determine bond lengths
        r_i = SI_O_BOND if ti == 0 else P_O_BOND
        r_j = SI_O_BOND if tj == 0 else P_O_BOND
        
        d = minimum_image(nf_coords[j] - nf_coords[i], box)
        dist = np.linalg.norm(d)
        
        if dist < 1e-6:
            continue
        
        # Place BO at weighted position
        frac = r_i / (r_i + r_j)
        o_pos = nf_coords[i] + d * frac
        
        # Small perpendicular offset for realistic geometry
        if dist > r_i + r_j:
            perp = np.cross(d, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp) < 0.1:
                perp = np.cross(d, np.array([0.0, 1.0, 0.0]))
            perp = perp / np.linalg.norm(perp)
            o_pos = o_pos + perp * rng.uniform(-0.1, 0.1)
        
        bo_coords.append(o_pos % box)
    
    return np.array(bo_coords, dtype=np.float64) if bo_coords else np.empty((0, 3))


# ========================= STEP 4: PLACE TERMINAL OXYGENS (NBO) =========================
def place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng):
    """
    Place NBOs for each NF to complete CN=4.
    CRITICAL: NBOs must be > NBO_SAFE_DIST from ALL other NFs.
    This prevents fake BO formation during MC.
    """
    term_coords = []
    term_owners = []
    
    n_nf = len(nf_coords)
    nf_tree = cKDTree(nf_coords, boxsize=box)
    
    # Combine BOs for overlap checking
    all_o = list(bo_coords) if len(bo_coords) > 0 else []
    o_tree = cKDTree(np.array(all_o), boxsize=box) if all_o else None
    
    # Tetrahedral directions
    TETRA = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=np.float64)
    for i in range(4):
        TETRA[i] /= np.linalg.norm(TETRA[i])
    
    for nf in range(n_nf):
        n_bo = int(deg[nf])
        n_nbo_needed = max(0, 4 - n_bo)
        
        if n_nbo_needed == 0:
            continue
        
        pos = nf_coords[nf]
        bond = SI_O_BOND if nf_types[nf] == 0 else P_O_BOND
        
        # Random rotation for tetrahedron
        R = random_rotation(rng)
        dirs = [(R @ TETRA[k]).tolist() for k in range(4)]
        rng.shuffle(dirs)
        
        placed_count = 0
        
        for direction in dirs:
            if placed_count >= n_nbo_needed:
                break
            
            d = np.array(direction)
            d += 0.05 * (rng.random(3) - 0.5)
            d = d / np.linalg.norm(d)
            
            o_pos = (pos + bond * d) % box
            
            # CHECK 1: Must be far from ALL other NFs
            nearby_nf = nf_tree.query_ball_point(o_pos, NBO_SAFE_DIST)
            too_close = False
            for nf_idx in nearby_nf:
                if nf_idx != nf:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # CHECK 2: Must not overlap with existing O
            if o_tree is not None:
                if len(o_tree.query_ball_point(o_pos, 1.85)) > 0:
                    continue
            
            # Place it
            term_coords.append(o_pos)
            term_owners.append(nf)
            all_o.append(o_pos)
            placed_count += 1
            
            # Rebuild tree periodically
            if len(all_o) % 200 == 0:
                o_tree = cKDTree(np.array(all_o), boxsize=box)
        
        # Fallback: if couldn't place all NBOs with safety check,
        # place remaining without NF check (MC will fix)
        while placed_count < n_nbo_needed:
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            o_pos = (pos + bond * d) % box
            term_coords.append(o_pos)
            term_owners.append(nf)
            all_o.append(o_pos)
            placed_count += 1
    
    return np.array(term_coords, dtype=np.float64), np.array(term_owners, dtype=np.int32)


# ========================= STEP 5: PLACE FREE OXYGENS =========================
def place_free_oxygens(n_fo, static_coords, box, rng):
    """Place FOs away from network."""
    if n_fo <= 0:
        return np.empty((0, 3))
    
    tree = cKDTree(static_coords, boxsize=box)
    fo_coords = []
    
    for _ in range(n_fo):
        for attempt in range(10000):
            pos = rng.uniform(0.0, box, 3)
            if len(tree.query_ball_point(pos, 2.1)) == 0:
                fo_coords.append(pos)
                break
        else:
            fo_coords.append(rng.uniform(0.0, box, 3))
    
    return np.array(fo_coords, dtype=np.float64)


# ========================= STEP 6: PLACE MODIFIERS =========================
def place_modifiers(n_mod_list, sites, site_owners, nf_types, static_coords, box, rng):
    """Place modifier cations near NBO/FO sites."""
    if len(sites) == 0:
        total = sum(n for _, n in n_mod_list)
        return rng.uniform(0.0, box, size=(total, 3))
    
    static_tree = cKDTree(static_coords, boxsize=box)
    target_dist = {"Na": NA_O_BOND, "Ca": CA_O_BOND, "Mg": MG_O_BOND}
    
    # P-associated sites
    p_site_idx = [i for i, owner in enumerate(site_owners) 
                  if owner >= 0 and nf_types[owner] == 1]
    
    mod_coords = []
    
    for elem, n_mod in n_mod_list:
        if n_mod <= 0:
            continue
        
        for _ in range(n_mod):
            pos = None
            
            for attempt in range(3000):
                # Ca and Mg prefer P sites
                if elem in ("Ca", "Mg") and p_site_idx and rng.random() < 0.35:
                    site = sites[p_site_idx[rng.integers(len(p_site_idx))]]
                else:
                    site = sites[rng.integers(len(sites))]
                
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                dist = target_dist[elem] + rng.uniform(-0.15, 0.25)
                cand = (site + direction * dist) % box
                
                # Check overlap with existing atoms
                if len(static_tree.query_ball_point(cand, 1.4)) > 0:
                    continue
                
                # Check overlap with other modifiers
                ok = True
                if mod_coords:
                    for m in mod_coords:
                        if np.linalg.norm(minimum_image(cand - m, box)) < 2.0:
                            ok = False
                            break
                
                if ok:
                    pos = cand
                    break
            
            if pos is None:
                pos = rng.uniform(0.0, box, 3)
            
            mod_coords.append(pos)
    
    return np.array(mod_coords, dtype=np.float64)


# ========================= STEP 7: RELAX O-O OVERLAPS =========================
def relax_o_overlaps(coords, types, box, iterations=100, step=0.08):
    """Simple repulsive relaxation for O-O overlaps."""
    new_coords = coords.copy()
    
    # Hard core distances
    r_hard = np.full((6, 6), 0.9)
    r_hard[4, 4] = 1.90  # O-O
    r_hard[0, 4] = r_hard[4, 0] = 1.30  # Si-O
    r_hard[3, 4] = r_hard[4, 3] = 1.30  # P-O
    r_hard[1, 4] = r_hard[4, 1] = 1.50  # Ca-O
    r_hard[2, 4] = r_hard[4, 2] = 1.50  # Na-O
    r_hard[5, 4] = r_hard[4, 5] = 1.20  # Mg-O
    
    for it in range(iterations):
        tree = cKDTree(new_coords, boxsize=box)
        pairs = tree.query_pairs(2.5, output_type='ndarray')
        
        moved = False
        for i, j in pairs:
            d = minimum_image(new_coords[i] - new_coords[j], box)
            r = np.linalg.norm(d)
            limit = r_hard[types[i], types[j]]
            
            if r < limit and r > 1e-6:
                moved = True
                f = d * (step * (limit - r) / r)
                new_coords[i] = (new_coords[i] + f) % box
                new_coords[j] = (new_coords[j] - f) % box
        
        if not moved:
            print(f"    Relaxation converged after {it + 1} iterations")
            break
    
    return new_coords


# ========================= VERIFICATION =========================
def verify_structure(symbols, coords, box, target_nc):
    """Verify structure quality."""
    print("\n" + "=" * 60)
    print("  STRUCTURE VERIFICATION")
    print("=" * 60)
    
    tree = cKDTree(coords, boxsize=box)
    
    # Close contacts
    checks = [
        ("O-O", "O", "O", 1.7),
        ("Si-O", "Si", "O", 1.2),
        ("P-O", "P", "O", 1.2),
    ]
    
    print("\n--- Close Contact Check ---")
    for name, e1, e2, threshold in checks:
        close = 0
        pairs = tree.query_pairs(threshold, output_type="ndarray")
        for i, j in pairs:
            if (symbols[i] == e1 and symbols[j] == e2) or \
               (symbols[i] == e2 and symbols[j] == e1):
                close += 1
        status = "PASS" if close == 0 else f"{close} violations"
        print(f"  {name}: {status}")
    
    # CN check
    si_idx = [i for i, s in enumerate(symbols) if s == "Si"]
    p_idx = [i for i, s in enumerate(symbols) if s == "P"]
    o_idx = [i for i, s in enumerate(symbols) if s == "O"]
    
    si_cn = [sum(1 for j in tree.query_ball_point(coords[si], 2.25) 
                 if j != si and symbols[j] == "O") for si in si_idx]
    p_cn = [sum(1 for j in tree.query_ball_point(coords[p], 2.25) 
                if j != p and symbols[j] == "O") for p in p_idx]
    
    print(f"\n--- CN Check ---")
    print(f"  Si-O CN: {np.mean(si_cn):.3f} (target: 4.0)")
    print(f"  P-O CN:  {np.mean(p_cn):.3f} (target: 4.0)")
    
    # Oxygen speciation
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
    
    print(f"\n--- Oxygen Speciation ---")
    print(f"  FO:  {fo:5d} ({fo/n_o*100:5.2f}%)")
    print(f"  NBO: {nbo:5d} ({nbo/n_o*100:5.2f}%)")
    print(f"  BO:  {bo:5d} ({bo/n_o*100:5.2f}%)")
    print(f"  TO:  {to:5d} ({to/n_o*100:5.2f}%)")
    
    # NC calculation
    bridging_set = {o_idx[i] for i, c in enumerate(o_nf_count) if c >= 2}
    
    def count_bo_around(center_idx):
        total_bo = 0
        for c in center_idx:
            bo_count = sum(
                1 for j in tree.query_ball_point(coords[c], 2.25)
                if j != c and symbols[j] == "O" and
                np.linalg.norm(minimum_image(coords[c] - coords[j], box)) < 2.25 and
                j in bridging_set
            )
            total_bo += min(bo_count, 4)
        return total_bo
    
    nc_si = count_bo_around(si_idx) / max(1, len(si_idx))
    nc_p = count_bo_around(p_idx) / max(1, len(p_idx))
    nc_comb = (count_bo_around(si_idx) + count_bo_around(p_idx)) / max(1, len(si_idx) + len(p_idx))
    
    print(f"\n--- Network Connectivity ---")
    print(f"  NC Si:       {nc_si:.3f}")
    print(f"  NC P:        {nc_p:.3f}")
    print(f"  NC Combined: {nc_comb:.3f} (target: {target_nc:.3f})")
    print("=" * 60)
    
    return nc_comb


# ========================= MAIN =========================
def generate_structure(mg_label, output_file, seed=42, scale=4):
    """Main structure generation function."""
    
    entry = PAPER_COMPOSITION[mg_label]
    rng = np.random.default_rng(seed)
    
    # Composition
    N_SI = 461 * scale
    N_P = 52 * scale
    N_NA = 488 * scale
    N_CA = entry["N_Ca"] * scale
    N_MG = entry["N_Mg"] * scale
    N_O = 1565 * scale
    
    print(f"\nComposition for {entry['label']} (scale={scale}x)")
    print(f"  Si={N_SI}, P={N_P}, Na={N_NA}, Ca={N_CA}, Mg={N_MG}, O={N_O}")
    print(f"  Total: {N_SI + N_P + N_NA + N_CA + N_MG + N_O} atoms")
    
    # Box size from density
    density = entry["density"]
    masses = {"Si": 28.0855, "Ca": 40.078, "Na": 22.98977, 
              "P": 30.97376, "O": 15.999, "Mg": 24.305}
    total_mass = (N_SI * masses["Si"] + N_P * masses["P"] + 
                  N_NA * masses["Na"] + N_CA * masses["Ca"] + 
                  N_MG * masses["Mg"] + N_O * masses["O"])
    box = ((total_mass / NA) / density * 1e24) ** (1/3)
    
    print(f"  Box: {box:.4f} A, Density: {density:.4f} g/cm3")
    
    # Calculate targets
    N_NF = N_SI + N_P
    target_nc = PAPER_NC[mg_label]
    n_bo_target = int(round(target_nc * N_NF / 2.0))
    n_siop_max = min(int(0.02 * N_O), n_bo_target, 2 * N_P)  # ~2% Si-O-P
    
    print(f"\nTargets:")
    print(f"  NC target: {target_nc}")
    print(f"  BO target: {n_bo_target}")
    print(f"  Si-O-P max: {n_siop_max}")
    
    # STEP 1: Place Network Formers
    print("\n[Step 1/7] Placing Network Formers...")
    nf_coords, nf_types = place_network_formers(N_SI, N_P, box, rng)
    
    # STEP 2: Build Topology
    print("[Step 2/7] Building topology...")
    edges, deg = build_topology(nf_coords, nf_types, box, n_bo_target, n_siop_max, rng)
    
    # STEP 3: Place BOs
    print("[Step 3/7] Placing Bridging Oxygens...")
    bo_coords = place_bridging_oxygens(nf_coords, nf_types, edges, box, rng)
    
    # STEP 4: Place NBOs
    print("[Step 4/7] Placing Terminal Oxygens (NBO)...")
    term_coords, term_owners = place_terminal_oxygens(nf_coords, nf_types, deg, box, bo_coords, rng)
    
    # STEP 5: Place FOs
    print("[Step 5/7] Placing Free Oxygens...")
    n_fo = N_O - len(term_coords) - len(bo_coords)
    if n_fo < 0:
        print(f"  WARNING: Too many O, truncating BO")
        bo_coords = bo_coords[:N_O - len(term_coords)]
        n_fo = 0
    
    static_for_fo = np.vstack([nf_coords, term_coords, bo_coords])
    fo_coords = place_free_oxygens(n_fo, static_for_fo, box, rng)
    
    print(f"  NBO: {len(term_coords)}, BO: {len(bo_coords)}, FO: {n_fo}")
    
    # STEP 6: Place Modifiers
    print("[Step 6/7] Placing Modifiers...")
    o_coords = np.vstack([term_coords, bo_coords, fo_coords])
    site_owners = np.concatenate([
        term_owners,
        -1 * np.ones(len(bo_coords) + len(fo_coords), dtype=np.int32)
    ])
    
    mod_coords = place_modifiers(
        [("Na", N_NA), ("Ca", N_CA), ("Mg", N_MG)],
        o_coords, site_owners, nf_types, 
        np.vstack([nf_coords, o_coords]), box, rng
    )
    
    # STEP 7: Relax O-O overlaps
    print("[Step 7/7] Relaxing O-O overlaps...")
    all_coords = np.vstack([nf_coords, o_coords, mod_coords])
    
    # Type array: Si=0, Ca=1, Na=2, P=3, O=4, Mg=5
    all_types = np.concatenate([
        np.zeros(N_SI + N_P, dtype=np.int32),  # NF (will fix below)
        np.full(len(o_coords), 4, dtype=np.int32),
        np.full(len(mod_coords), 1, dtype=np.int32),  # placeholder
    ])
    
    # Fix NF types
    all_types[:N_SI + N_P] = nf_types
    
    # Fix modifier types
    mod_start = N_SI + N_P + len(o_coords)
    all_types[mod_start:mod_start + N_NA] = 2  # Na
    all_types[mod_start + N_NA:mod_start + N_NA + N_CA] = 1  # Ca
    all_types[mod_start + N_NA + N_CA:] = 5  # Mg
    
    relaxed_coords = relax_o_overlaps(all_coords, all_types, box)
    
    # Build final structure
    symbols = (["Si"] * N_SI + ["P"] * N_P + ["O"] * len(o_coords) +
               ["Na"] * N_NA + ["Ca"] * N_CA + ["Mg"] * N_MG)
    
    # Shuffle
    perm = rng.permutation(len(symbols))
    symbols = [symbols[i] for i in perm]
    final_coords = relaxed_coords[perm]
    
    # Verify
    nc_result = verify_structure(symbols, final_coords, box, target_nc)
    
    # Save
    out_path = Path(output_file)
    with open(out_path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"Mg-doped 45S5 FINAL, label={entry['label']}, x={mg_label}, "
                f"N={len(symbols)}, rho={density:.6f}, box={box:.6f}, seed={seed}\n")
        for i in range(len(symbols)):
            f.write(f"{symbols[i]:2s} {final_coords[i,0]:12.6f} "
                    f"{final_coords[i,1]:12.6f} {final_coords[i,2]:12.6f}\n")
    
    print(f"\n[OK] Saved to: {out_path}")
    
    return nc_result


# ========================= CLI =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mg-doped 45S5 Structure Generator (FINAL)")
    parser.add_argument("--mg", type=int, default=0, choices=sorted(PAPER_COMPOSITION.keys()))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4, choices=[1, 2, 4])
    args = parser.parse_args()
    
    if args.out is None:
        args.out = f"initial_Mg{args.mg}.xyz"
    
    generate_structure(args.mg, args.out, args.seed, args.scale)
    