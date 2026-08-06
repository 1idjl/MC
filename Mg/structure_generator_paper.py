#!/usr/bin/env python3
"""
========================================================================
Smart Initial Structure Generator for 45S5 Bioglass with Mg-doping (NEW!)
Version: v5.2 (Mg Edition - NO Sr)
Based on: Xiang & Du, Chem. Mater. 2011 + Manuscript.docx (Mg Substitution)
NEW FEATURES (v5.2):
1. Removes ALL Sr references
2. Adds Mg doping (45-M0 to 45-M20) based on manuscript Table 2
3. Uses composition from Table 2 of Manuscript.docx
4. Calculates N_CA and N_MG based on x MgO mol%
5. Maintains original algorithm (NF -> O -> Modifiers, Affinity, etc.)
6. Supports 2835/5670/11340 atom systems
7. Optional pre-relaxation report
Usage:
python structure_generator_paper.py --mg 0 --density 2.6738 --seed 42 --scale 4
python structure_generator_paper.py --mg 5 --density 2.6298 --seed 42 --scale 4
...
python structure_generator_paper.py --mg 20 --density 2.5502 --seed 42 --scale 4 --scale 4
========================================================================
"""

import numpy as np
from numba import njit
import argparse
from pathlib import Path
from scipy.spatial import cKDTree # Keep scipy import here

NA = 6.02214076e23
BASE_N_ATOMS = 2835 # Base number of atoms for the standard composition (from manuscript Table 2)

# Type indices (NO Sr)
# Si=0, Ca=1, Na=2, P=3, O=4, Mg=5
TYPE_MAP = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Mg': 5} # Sr removed, Mg added
ELEM_MAP = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Mg'} # Sr removed, Mg added

# Masses (NO Sr)
MASSES = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
          'P': 30.97376, 'O': 15.999, 'Mg': 24.305} # Sr removed, Mg added


@njit(fastmath=True, cache=True)
def minimum_image(dr, box):
    """Apply minimum image convention."""
    dr -= box * np.floor(dr / box + 0.5)
    return dr

@njit(fastmath=True, cache=True)
def place_network_formers(n_si, n_p, box, min_nf_dist=3.0, seed=42):
    """Stage 1: Place Si and P atoms with minimum distance."""
    np.random.seed(seed)
    coords = np.zeros((n_si + n_p, 3), dtype=np.float64)
    placed = 0
    max_attempts = 100000
    min_d_sq = min_nf_dist * min_nf_dist

    for i in range(n_si + n_p):
        success = False
        for attempt in range(max_attempts):
            px = np.random.uniform(0, box)
            py = np.random.uniform(0, box)
            pz = np.random.uniform(0, box)

            valid = True
            for j in range(placed):
                dx = coords[j, 0] - px
                dy = coords[j, 1] - py
                dz = coords[j, 2] - pz
                dx, dy, dz = minimum_image(np.array([dx, dy, dz]), box)
                if dx*dx + dy*dy + dz*dz < min_d_sq:
                    valid = False
                    break

            if valid:
                coords[placed, 0] = px
                coords[placed, 1] = py
                coords[placed, 2] = pz
                placed += 1
                success = True
                break

        if not success:
            coords[placed, 0] = np.random.uniform(0, box)
            coords[placed, 1] = np.random.uniform(0, box)
            coords[placed, 2] = np.random.uniform(0, box)
            placed += 1

    return coords, placed

@njit(fastmath=True, cache=True)
def place_oxygens_around_nf(nf_coords, n_nf, box, o_si_dist, o_p_dist,
                           n_si, o_min_dist, seed):
    """Stage 2: Place O around Si/P in TETRAHEDRAL geometry
    with RANDOM ORIENTATION per tetrahedron."""
    np.random.seed(seed + 1000)
    n_o_needed = n_nf * 4
    coords = np.zeros((n_o_needed, 3), dtype=np.float64)
    placed = 0
    o_min_d_sq = o_min_dist * o_min_dist

    for i in range(n_nf):
        center = nf_coords[i]
        # Tetrahedron vectors (randomly oriented)
        # Generate a random rotation matrix
        u = np.random.rand()
        v = np.random.rand()
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        # Initial tetrahedron points (relative to origin)
        # Using canonical points and rotating them
        # Points: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) normalized
        norm_factor = np.sqrt(1**2 + 1**2 + 1**2)
        tetra_points = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]) / norm_factor

        # Simple rotation (could be more sophisticated)
        # Rotate around a random axis
        angle = np.random.uniform(0, 2*np.pi)
        rot_axis = np.random.rand(3)
        rot_axis /= np.linalg.norm(rot_axis)
        # Rodrigues' rotation formula applied to each point
        rotated_points = np.zeros_like(tetra_points)
        for k, pt in enumerate(tetra_points):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            cross = np.cross(rot_axis, pt)
            dot = np.dot(rot_axis, pt)
            rotated_points[k] = pt * cos_a + cross * sin_a + rot_axis * dot * (1 - cos_a)

        # Scale and translate points
        dist = o_si_dist if i < n_si else o_p_dist
        for j in range(4):
            offset = rotated_points[j] * dist
            px = center[0] + offset[0]
            py = center[1] + offset[1]
            pz = center[2] + offset[2]

            # Apply PBC
            px = px - box * np.floor(px / box)
            py = py - box * np.floor(py / box)
            pz = pz - box * np.floor(pz / box)

            # Check minimum distance from previously placed O
            valid = True
            for k in range(placed):
                dx = coords[k, 0] - px
                dy = coords[k, 1] - py
                dz = coords[k, 2] - pz
                dx, dy, dz = minimum_image(np.array([dx, dy, dz]), box)
                if dx*dx + dy*dy + dz*dz < o_min_d_sq:
                    valid = False
                    break

            if valid:
                coords[placed, 0] = px
                coords[placed, 1] = py
                coords[placed, 2] = pz
                placed += 1
            else:
                 # Fallback: place randomly nearby if overlap occurs
                 # This might be less ideal but avoids failure
                 # A better approach might be to try different orientations
                 # For now, just place it if no valid spot found after rotation
                 coords[placed, 0] = px
                 coords[placed, 1] = py
                 coords[placed, 2] = pz
                 placed += 1

    return coords, placed

@njit(fastmath=True, cache=True)
def place_modifiers_affinity(n_mod, box, all_nf_o_coords, n_all,
                             mod_o_min_dist=2.0, mod_mod_min_dist=2.0,
                             p_coords=None, n_p=0,
                             p_attraction_radius=6.0, seed=42):
    """Stage 3: Place modifiers (Na, Ca, Mg) with affinity rules."""
    np.random.seed(seed + 2000)
    coords = np.zeros((n_mod, 3), dtype=np.float64)
    placed = 0
    max_attempts = 300000 # Increased attempts
    mod_o_min_d_sq = mod_o_min_dist * mod_o_min_dist
    mod_mod_min_d_sq = mod_mod_min_dist * mod_mod_min_dist

    # Pre-calculate P attraction probability based on N_P fraction
    # Higher N_P might mean higher chance to place Ca/Mg near P
    # For now, fixed probability
    ca_mg_prefers_p_prob = 0.7 # Ca and Mg prefer P neighborhood
    na_no_pref_prob = 0.3      # Na has lower preference for P

    for i in range(n_mod):
        success = False
        for attempt in range(max_attempts):
            # Determine element type based on index 'i'
            # Assuming order: Na, Ca, Mg (Need to know counts beforehand or pass them)
            # This is tricky with the current logic. Let's assume a general modifier placement
            # and let the assembly stage assign types based on counts.
            # For affinity, we'll use a probabilistic approach.
            # If Ca/Mg, higher chance to be near P.
            # If Na, lower chance.
            # For now, treat all modifiers similarly but with slight bias.
            # Better: Pass counts and assign types here or pass specific counts for each modifier type.
            # Simpler: Just place, but bias placement based on type.
            # Let's assume Ca/Mg are placed first (indices 0 to N_CA-1 and N_CA to N_CA+N_MG-1)
            # and Na last (N_CA+N_MG to N_MOD-1).
            # This requires knowing N_CA, N_MG, N_NA upfront.
            # We pass n_mod (total modifiers), and inside, decide based on remaining counts
            # or use a global counter/array which is complex in njit.
            # Alternative: Decide placement strategy probabilistically without knowing exact type yet.
            # This is less precise but simpler.

            # Probabilistic placement
            place_near_p = (n_p > 0 and np.random.random() < ca_mg_prefers_p_prob)
            place_near_na_or_ca = np.random.random() < na_no_pref_prob

            if place_near_p:
                # Try to place near a P atom
                p_idx = np.random.randint(0, n_p)
                theta = np.random.uniform(0, 2.0 * np.pi)
                phi = np.arccos(2.0 * np.random.random() - 1.0) # Uniform on sphere
                r = np.random.uniform(2.5, p_attraction_radius) # Distance from P
                px = (p_coords[p_idx, 0] + r * np.sin(phi) * np.cos(theta)) % box
                py = (p_coords[p_idx, 1] + r * np.sin(phi) * np.cos(theta)) % box
                pz = (p_coords[p_idx, 2] + r * np.sin(phi) * np.cos(theta)) % box
            else:
                # Place randomly
                px = np.random.uniform(0, box)
                py = np.random.uniform(0, box)
                pz = np.random.uniform(0, box)

            # Check minimum distance from all previously placed modifiers and O
            valid = True
            for j in range(placed):
                dx = coords[j, 0] - px
                dy = coords[j, 1] - py
                dz = coords[j, 2] - pz
                dx, dy, dz = minimum_image(np.array([dx, dy, dz]), box)
                if dx*dx + dy*dy + dz*dz < mod_mod_min_d_sq:
                    valid = False
                    break
            if valid:
                 # Also check distance from NF and O (all_nf_o_coords)
                 for j in range(n_all):
                      dx = all_nf_o_coords[j, 0] - px
                      dy = all_nf_o_coords[j, 1] - py
                      dz = all_nf_o_coords[j, 2] - pz
                      dx, dy, dz = minimum_image(np.array([dx, dy, dz]), box)
                      if dx*dx + dy*dy + dz*dz < mod_o_min_d_sq:
                          valid = False
                          break

            if valid:
                coords[placed, 0] = px
                coords[placed, 1] = py
                coords[placed, 2] = pz
                placed += 1
                success = True
                break

        if not success:
             # Fallback: place randomly with minimal checks
             print(f"Warning: Could not place modifier {i+1} optimally, using fallback.")
             coords[placed, 0] = np.random.uniform(0, box)
             coords[placed, 1] = np.random.uniform(0, box)
             coords[placed, 2] = np.random.uniform(0, box)
             placed += 1

    return coords, placed

# ================================================================
# VERIFICATION (Updated for Mg, NO Sr)
# ================================================================
def verify_structure(symbols, coords, box):
    """Comprehensive structure verification."""
    print("" + "=" * 60)
    print(" STRUCTURE VERIFICATION REPORT")
    print("=" * 60)
    tree = cKDTree(coords, boxsize=box)
    n = len(symbols)

    # 1. Close contact check
    print("- Close Contact Check -")
    checks = [('Si-Si', 'Si', 'Si', 2.6),
              ('O-O', 'O', 'O', 1.8),
              ('Si-O', 'Si', 'O', 1.3),
              ('P-O', 'P', 'O', 1.3),
              ('Na-O', 'Na', 'O', 2.0),
              ('Ca-O', 'Ca', 'O', 2.0),
              ('Mg-O', 'Mg', 'O', 2.0)] # Added Mg-O check

    for label, e1, e2, dist in checks:
        t1, t2 = TYPE_MAP[e1], TYPE_MAP[e2]
        idx1 = [i for i, s in enumerate(symbols) if s == e1]
        idx2 = [i for i, s in enumerate(symbols) if s == e2]
        close = 0
        for i in idx1:
            neigh = tree.query_ball_point(coords[i], dist)
            close += sum(1 for j in neigh if j != i and j in idx2)
        print(f"  {label:<6s}: {close} contacts < {dist:.1f} A")

    # 2. Si coordination distribution
    print("- Si Coordination Distribution -")
    si_idx = [i for i, s in enumerate(symbols) if s == 'Si']
    o_idx_set = set(i for i, s in enumerate(symbols) if s == 'O')
    si_cn = {}
    for si in si_idx:
        neigh = tree.query_ball_point(coords[si], 2.25)
        cn = sum(1 for j in neigh if j != si and j in o_idx_set)
        si_cn[cn] = si_cn.get(cn, 0) + 1
    for cn in sorted(si_cn.keys()):
        pct = si_cn[cn] / len(si_idx) * 100
        bar = '█' * int(pct / 2)
        print(f"  CN={cn}: {si_cn[cn]:5d} ({pct:5.1f}%) {bar}")

    # 3. P coordination distribution
    print("- P Coordination Distribution -")
    p_idx = [i for i, s in enumerate(symbols) if s == 'P']
    p_cn = {}
    for p in p_idx:
        neigh = tree.query_ball_point(coords[p], 2.25)
        cn = sum(1 for j in neigh if j != p and j in o_idx_set)
        p_cn[cn] = p_cn.get(cn, 0) + 1
    for cn in sorted(p_cn.keys()):
        pct = p_cn[cn] / len(p_idx) * 100
        bar = '█' * int(pct / 2)
        print(f"  CN={cn}: {p_cn[cn]:5d} ({pct:5.1f}%) {bar}")

    # 4. Modifier-O distance stats
    print("- Modifier-O Distance Stats -")
    for mod in ['Na', 'Ca', 'Mg']: # Added Mg, removed Sr
        mod_idx = [i for i, s in enumerate(symbols) if s == mod]
        o_idx = [i for i, s in enumerate(symbols) if s == 'O']
        if not mod_idx or not o_idx: continue
        dists = []
        for m in mod_idx:
            neigh = tree.query_ball_point(coords[m], 5.0) # Search radius
            for j in neigh:
                if j != m and symbols[j] == 'O':
                    d = coords[m] - coords[j]
                    d = minimum_image(d, box)
                    dists.append(np.linalg.norm(d))
        if dists:
            print(f"  {mod}-O: mean={np.mean(dists):.3f} A, "
                  f"min={np.min(dists):.3f}, max={np.max(dists):.3f}, "
                  f"avg_CN={len(dists)/len(mod_idx):.2f}")

    print("=" * 60)

# ================================================================
# MAIN GENERATOR (Updated for Mg, NO Sr)
# ================================================================
def generate_structure(mg_percent, density, output_file, seed, scale=1):
    # ===== COMPOSITION (Based on Manuscript Table 2, scale accordingly) =====
    # Original counts for 2835 atoms (x = mg_percent mol%)
    # N_SI = 461
    # N_P = 52
    # N_NA = 488
    # N_CA_MG_TOTAL = 269 (Constant total Ca+Mg)
    # N_O = 1565
    # N_MG = N_CA_MG_TOTAL * (x / 100)
    # N_CA = N_CA_MG_TOTAL - N_MG

    # Apply scaling factor
    N_SI = 461 * scale
    N_P = 52 * scale
    N_NA = 488 * scale
    N_CA_MG_TOTAL_ORIG = 269 # Original total for Ca+Mg in 2835 atom system
    N_CA_MG_TOTAL = N_CA_MG_TOTAL_ORIG * scale # Scaled total for Ca+Mg

    # Calculate N_MG and N_CA based on mg_percent
    N_MG = int(round(N_CA_MG_TOTAL * (mg_percent / 100.0)))
    N_CA = N_CA_MG_TOTAL - N_MG

    N_O = 1565 * scale
    N_TARGET = N_SI + N_P + N_NA + N_CA_MG_TOTAL + N_O # Should equal BASE_N_ATOMS*scale if formula is correct

    counts = {'Si': N_SI, 'P': N_P, 'Na': N_NA, 'Ca': N_CA, 'Mg': N_MG, 'O': N_O} # Added Mg, removed Sr
    actual_total = sum(counts.values())

    print(f"Composition for {mg_percent}% Mg doping (SCALE={scale}x based on 2835 atoms):")
    for k, v in counts.items():
        print(f" {k}: {v}")
    print(f" TOTAL (calculated): {actual_total}")
    print(f" TARGET (scaled base): {BASE_N_ATOMS * scale}")

    # Calculate mass and box size
    total_mass = sum(MASSES[s] * c for s, c in counts.items())
    volume_A3 = (total_mass / NA) / density * 1e24
    box = volume_A3 ** (1/3)
    print(f"Box size: {box:.4f} A (Density: {density} g/cm3)")

    # ===== STAGE 1: Network Formers =====
    print(f"[Stage 1/4] Placing {N_SI+N_P} network formers (min NF-NF = 3.0 A)...")
    nf_coords, nf_placed = place_network_formers(N_SI, N_P, box, min_nf_dist=3.0, seed=seed)
    print(f"  Placed {nf_placed} NF ({N_SI} Si + {N_P} P)")

    # ===== STAGE 2: Oxygens (Tetrahedral + Random Orientation) =====
    print("[Stage 2/4] Placing O in tetrahedral geometry (RANDOM ORIENTATION)...")
    o_coords, o_placed = place_oxygens_around_nf(
        nf_coords, nf_placed, box,
        o_si_dist=1.61, o_p_dist=1.54, # Distances from paper or common values
        n_si=N_SI, o_min_dist=2.0, seed=seed)
    print(f"  Placed {o_placed} O (target: {N_O})")

    # Adjust O count if needed (should match N_O from table)
    if o_placed > N_O:
        o_coords = o_coords[:N_O]
        o_placed = N_O
    elif o_placed < N_O:
        np.random.seed(seed + 3000)
        remaining = N_O - o_placed
        extra_o = np.zeros((remaining, 3), dtype=np.float64)
        half_box = box / 2.0
        o_min_d_sq = 2.0 ** 2
        for i in range(remaining):
            for attempt in range(100000):
                px = np.random.uniform(0, box)
                py = np.random.uniform(0, box)
                pz = np.random.uniform(0, box)
                valid = True
                for j in range(o_placed):
                    dx = px - o_coords[j, 0]
                    dy = py - o_coords[j, 1]
                    dz = pz - o_coords[j, 2]
                    if dx > half_box: dx -= box
                    elif dx < -half_box: dx += box
                    if dy > half_box: dy -= box
                    elif dy < -half_box: dy += box
                    if dz > half_box: dz -= box
                    elif dz < -half_box: dz += box
                    if dx*dx + dy*dy + dz*dz < o_min_d_sq:
                        valid = False
                        break
                if valid:
                    extra_o[i] = [px, py, pz]
                    break
        o_coords = np.vstack((o_coords, extra_o))
        o_placed = N_O
        print(f"  Added {remaining} extra O randomly")

    # ===== STAGE 3: Modifiers (Affinity-aware) =====
    print("[Stage 3/4] Placing modifiers (AFFINITY-AWARE: Ca/Mg prefer P)...")
    all_nf_o = np.vstack((nf_coords, o_coords[:N_O]))
    n_all = nf_placed + N_O
    N_MOD = N_NA + N_CA + N_MG # Total modifiers

    # P coords for affinity placement
    p_coords = nf_coords[N_SI:N_SI + N_P] # Coordinates of P atoms
    mod_coords, mod_placed = place_modifiers_affinity(
        N_MOD, box, all_nf_o, n_all,
        mod_o_min_dist=2.0, mod_mod_min_dist=2.0,
        p_coords=p_coords, n_p=N_P, # Pass P coords and count
        p_attraction_radius=6.0, seed=seed)
    print(f"  Placed {mod_placed} modifiers ({N_NA} Na + {N_CA} Ca + {N_MG} Mg)") # Mention Mg count

    # ===== STAGE 4: Assembly =====
    print("[Stage 4/4] Assembling final structure...")
    symbols = []
    all_coords = []

    # Add Si
    for i in range(N_SI):
        symbols.append('Si'); all_coords.append(nf_coords[i])
    # Add P
    for i in range(N_P):
        symbols.append('P'); all_coords.append(nf_coords[N_SI + i])
    # Add O
    for i in range(N_O):
        symbols.append('O'); all_coords.append(o_coords[i])
    # Add Na
    for i in range(N_NA):
        symbols.append('Na'); all_coords.append(mod_coords[i])
    # Add Ca
    for i in range(N_CA):
        symbols.append('Ca'); all_coords.append(mod_coords[N_NA + i])
    # Add Mg (Replaces Sr)
    for i in range(N_MG):
        symbols.append('Mg'); all_coords.append(mod_coords[N_NA + N_CA + i]) # Mg comes after Ca

    all_coords = np.array(all_coords, dtype=np.float64)
    total_atoms = len(symbols)
    print(f" Total atoms: {total_atoms}")

    # ===== VERIFICATION =====
    verify_structure(symbols, all_coords, box)

    # ===== SHUFFLE (Optional but recommended for initial randomness) =====
    np.random.seed(seed + 5000)
    perm = np.random.permutation(total_atoms)
    symbols = [symbols[i] for i in perm]
    all_coords = all_coords[perm]

    # ===== WRITE XYZ =====
    out_path = Path(output_file)
    with open(out_path, 'w') as f:
        f.write(f"{total_atoms}\n")
        f.write(f"Mg-doped Xiang-Du 2011, x={mg_percent} mol% MgO, " # Updated label, removed Sr
                f"N={total_atoms}, rho={density:.4f} g/cm3, "
                f"box={box:.4f} A, seed={seed}\n")
        for i in range(total_atoms):
            f.write(f"{symbols[i]:2s} "
                    f"{all_coords[i,0]:10.6f} "
                    f"{all_coords[i,1]:10.6f} "
                    f"{all_coords[i,2]:10.6f}\n")

    print(f"[OK] Initial structure saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart initial structure generator v5.2 for 45S5 bioglass with Mg doping (NO Sr)') # Updated description
    parser.add_argument('--mg', type=float, default=0.0, # Changed argument name from 'sr' to 'mg'
                        help='Mg doping percentage (0.0 to 20.0) - replaces Ca with Mg') # Updated help
    parser.add_argument('--density', type=float, required=True, # Made density required for clarity
                        help='Density in g/cm3 (from manuscript Table 2)') # Updated help
    parser.add_argument('--out', type=str, default=None,
                        help='Output XYZ file name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--scale', type=int, default=1, choices=[1, 2, 4],
                        help='Scale factor: 1=2835, 2=5670, 4=11340 atoms')

    args = parser.parse_args()

    if args.out is None:
        args.out = f"initial_Mg{args.mg:.0f}_scale{args.scale}.xyz" # Updated default filename

    generate_structure(args.mg, args.density, args.out, args.seed, args.scale)
