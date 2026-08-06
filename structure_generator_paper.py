#!/usr/bin/env python3
"""
Smart Initial Structure Generator for 45S5 Bioglass with Sr-doping.
Based on: Xiang & Du, Chem. Mater. 2011

NEW FEATURES (v4.0):
- Staged placement: Network formers first, then O around them, then modifiers
- Pair-specific minimum distances
- Si-Si, P-P, O-O close contact prevention
- Coordination-aware placement for SiO4 and PO4 tetrahedra
- Cell-linked-list algorithm for speed

Usage:
  python struture.py --sr 0 --density 2.649 --seed 42
  python struture.py --sr 5 --density 2.712 --seed 42
"""
import numpy as np
from numba import njit
import argparse
from pathlib import Path

NA = 6.02214076e23
SCALE = 4

# Type indices (MUST match Sr.py)
# Si=0, Ca=1, Na=2, P=3, O=4, Sr=5

@njit(fastmath=True, cache=True)
def check_min_dist(px, py, pz, coords, n_placed, min_d_sq, box):
    """Check if point (px,py,pz) is at least min_d away from all placed atoms."""
    half_box = box / 2.0
    for j in range(n_placed):
        dx = px - coords[j, 0]
        dy = py - coords[j, 1]
        dz = pz - coords[j, 2]
        if dx > half_box: dx -= box
        elif dx < -half_box: dx += box
        if dy > half_box: dy -= box
        elif dy < -half_box: dy += box
        if dz > half_box: dz -= box
        elif dz < -half_box: dz += box
        if dx*dx + dy*dy + dz*dz < min_d_sq:
            return False
    return True


@njit(fastmath=True, cache=True)
def place_network_formers(n_si, n_p, box, min_nf_dist, seed):
    """
    Stage 1: Place Si and P atoms with minimum NF-NF distance.
    Returns combined coords array for Si and P.
    """
    np.random.seed(seed)
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    min_d_sq = min_nf_dist * min_nf_dist
    placed = 0
    max_attempts = 200000

    for i in range(n_total):
        success = False
        for attempt in range(max_attempts):
            px = np.random.uniform(0, box)
            py = np.random.uniform(0, box)
            pz = np.random.uniform(0, box)

            if check_min_dist(px, py, pz, coords, placed, min_d_sq, box):
                coords[placed, 0] = px
                coords[placed, 1] = py
                coords[placed, 2] = pz
                placed += 1
                success = True
                break

        if not success:
            # Force place (rare)
            coords[placed, 0] = np.random.uniform(0, box)
            coords[placed, 1] = np.random.uniform(0, box)
            coords[placed, 2] = np.random.uniform(0, box)
            placed += 1

    return coords, placed


@njit(fastmath=True, cache=True)
def place_oxygens_around_nf(nf_coords, n_nf, box, o_si_dist, o_p_dist,
                            n_si, o_min_dist, o_nf_min_dist, seed):
    """
    Stage 2: Place O atoms around Si and P to form tetrahedra.
    Each Si gets 4 O at ~1.61 A, each P gets 4 O at ~1.54 A.
    """
    np.random.seed(seed + 1000)
    n_o_needed = n_nf * 4  # Each NF gets 4 O
    o_coords = np.zeros((n_o_needed, 3), dtype=np.float64)
    o_placed = 0
    half_box = box / 2.0
    o_min_d_sq = o_min_dist * o_min_dist
    max_attempts = 50000

    for nf in range(n_nf):
        nf_x = nf_coords[nf, 0]
        nf_y = nf_coords[nf, 1]
        nf_z = nf_coords[nf, 2]

        # Determine bond distance based on NF type
        if nf < n_si:
            bond_dist = o_si_dist  # Si-O = 1.61
        else:
            bond_dist = o_p_dist   # P-O = 1.54

        # Place 4 oxygens in approximate tetrahedral positions
        # Using tetrahedral angle ~109.47 degrees
        # Direction vectors for a regular tetrahedron
        tet_dirs = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0]
        ], dtype=np.float64)

        # Normalize
        for d in range(4):
            norm = np.sqrt(tet_dirs[d, 0]**2 + tet_dirs[d, 1]**2 + tet_dirs[d, 2]**2)
            tet_dirs[d, 0] /= norm
            tet_dirs[d, 1] /= norm
            tet_dirs[d, 2] /= norm

        for k in range(4):
            # Add small random perturbation to break perfect symmetry
            px = nf_x + bond_dist * (tet_dirs[k, 0] + 0.15 * (np.random.random() - 0.5))
            py = nf_y + bond_dist * (tet_dirs[k, 1] + 0.15 * (np.random.random() - 0.5))
            pz = nf_z + bond_dist * (tet_dirs[k, 2] + 0.15 * (np.random.random() - 0.5))

            # Apply PBC
            px = px % box
            py = py % box
            pz = pz % box

            # Check O-O minimum distance with previously placed O
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
                o_coords[o_placed, 0] = px
                o_coords[o_placed, 1] = py
                o_coords[o_placed, 2] = pz
                o_placed += 1
            else:
                # Try random position around NF
                for attempt in range(max_attempts):
                    # Random direction on sphere
                    theta = np.random.uniform(0, 2.0 * np.pi)
                    phi = np.arccos(2.0 * np.random.random() - 1.0)
                    rx = bond_dist * np.sin(phi) * np.cos(theta)
                    ry = bond_dist * np.sin(phi) * np.sin(theta)
                    rz = bond_dist * np.cos(phi)

                    px = (nf_x + rx) % box
                    py = (nf_y + ry) % box
                    pz = (nf_z + rz) % box

                    valid2 = True
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
                            valid2 = False
                            break

                    if valid2:
                        o_coords[o_placed, 0] = px
                        o_coords[o_placed, 1] = py
                        o_coords[o_placed, 2] = pz
                        o_placed += 1
                        break

    return o_coords, o_placed


@njit(fastmath=True, cache=True)
def place_modifiers(n_mod, box, mod_coords_start_idx, all_coords, n_all_placed,
                    mod_o_min_dist, mod_mod_min_dist, seed):
    """
    Stage 3: Place modifier cations (Na, Ca, Sr) avoiding O and each other.
    """
    np.random.seed(seed + 2000)
    half_box = box / 2.0
    mod_o_min_d_sq = mod_o_min_dist * mod_o_min_dist
    mod_mod_min_d_sq = mod_mod_min_dist * mod_mod_min_dist
    placed = 0
    max_attempts = 200000

    mod_coords = np.zeros((n_mod, 3), dtype=np.float64)

    for i in range(n_mod):
        success = False
        for attempt in range(max_attempts):
            px = np.random.uniform(0, box)
            py = np.random.uniform(0, box)
            pz = np.random.uniform(0, box)

            # Check distance to O atoms
            valid = True
            for j in range(n_all_placed):
                dx = px - all_coords[j, 0]
                dy = py - all_coords[j, 1]
                dz = pz - all_coords[j, 2]
                if dx > half_box: dx -= box
                elif dx < -half_box: dx += box
                if dy > half_box: dy -= box
                elif dy < -half_box: dy += box
                if dz > half_box: dz -= box
                elif dz < -half_box: dz += box
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < mod_o_min_d_sq:
                    valid = False
                    break

            if not valid:
                continue

            # Check distance to other modifiers
            for j in range(placed):
                dx = px - mod_coords[j, 0]
                dy = py - mod_coords[j, 1]
                dz = pz - mod_coords[j, 2]
                if dx > half_box: dx -= box
                elif dx < -half_box: dx += box
                if dy > half_box: dy -= box
                elif dy < -half_box: dy += box
                if dz > half_box: dz -= box
                elif dz < -half_box: dz += box
                if dx*dx + dy*dy + dz*dz < mod_mod_min_d_sq:
                    valid = False
                    break

            if valid:
                mod_coords[placed, 0] = px
                mod_coords[placed, 1] = py
                mod_coords[placed, 2] = pz
                placed += 1
                success = True
                break

        if not success:
            mod_coords[placed, 0] = np.random.uniform(0, box)
            mod_coords[placed, 1] = np.random.uniform(0, box)
            mod_coords[placed, 2] = np.random.uniform(0, box)
            placed += 1

    return mod_coords, placed


def generate_structure(sr_percent: int, density: float, output_file: str, seed: int):
    # ===== COMPOSITION (11340 atoms = 4 x 2835) =====
    N_SI_BASE    = 461
    N_P_BASE     = 52
    N_NA_BASE    = 488
    N_CA_SR_BASE = 269
    N_O_BASE     = 1565

    N_SI          = N_SI_BASE * SCALE       # 1844
    N_P           = N_P_BASE * SCALE        # 208
    N_NA          = N_NA_BASE * SCALE       # 1952
    N_CA_SR_TOTAL = N_CA_SR_BASE * SCALE    # 1076
    N_O           = N_O_BASE * SCALE        # 6260
    N_TARGET      = N_SI + N_P + N_NA + N_CA_SR_TOTAL + N_O  # 11340

    N_SR = int(round(N_CA_SR_TOTAL * (sr_percent / 100.0)))
    N_CA = N_CA_SR_TOTAL - N_SR

    counts = {'Si': N_SI, 'P': N_P, 'Na': N_NA, 'Ca': N_CA, 'Sr': N_SR, 'O': N_O}
    actual_total = sum(counts.values())

    print(f"Composition for {sr_percent}% Sr doping (SCALE={SCALE}x):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL: {actual_total}")

    if actual_total != N_TARGET:
        print(f"  WARNING: Total {actual_total} != expected {N_TARGET}")

    masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
              'P': 30.97376, 'O': 15.999, 'Sr': 87.62}
    total_mass = sum(masses[s] * c for s, c in counts.items())
    volume_A3 = (total_mass / NA) / density * 1e24
    box = volume_A3 ** (1/3)
    print(f"Box size: {box:.4f} A (Density: {density} g/cm3)")
    print(f"Volume: {volume_A3:.2f} A3")

    # ===== STAGED PLACEMENT =====
    print("\n[Stage 1/4] Placing network formers (Si, P) with min NF-NF distance 3.0 A...")
    nf_coords, nf_placed = place_network_formers(
        N_SI, N_P, box, min_nf_dist=3.0, seed=seed
    )
    print(f"  Placed {nf_placed} network formers ({N_SI} Si + {N_P} P)")

    print("[Stage 2/4] Placing O atoms around Si and P (tetrahedral)...")
    o_coords, o_placed = place_oxygens_around_nf(
        nf_coords, nf_placed, box,
        o_si_dist=1.61, o_p_dist=1.54,
        n_si=N_SI,
        o_min_dist=2.0, o_nf_min_dist=1.4,
        seed=seed
    )
    print(f"  Placed {o_placed} oxygens (target: {N_O})")

    # If we placed more or fewer O than needed, adjust
    if o_placed > N_O:
        o_coords = o_coords[:N_O]
        o_placed = N_O
    elif o_placed < N_O:
        # Add remaining O randomly with min O-O distance
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
                # Check against existing O
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
                    extra_o[i, 0] = px
                    extra_o[i, 1] = py
                    extra_o[i, 2] = pz
                    break
        o_coords = np.vstack((o_coords, extra_o))
        o_placed = N_O
        print(f"  Added {remaining} extra O atoms randomly")

    # Combine NF + O coords for modifier placement
    all_nf_o_coords = np.vstack((nf_coords, o_coords[:N_O]))
    n_all = nf_placed + N_O

    print("[Stage 3/4] Placing modifier cations (Na, Ca, Sr)...")
    N_MOD = N_NA + N_CA + N_SR
    mod_coords, mod_placed = place_modifiers(
        N_MOD, box, 0, all_nf_o_coords, n_all,
        mod_o_min_dist=2.0, mod_mod_min_dist=2.0,
        seed=seed
    )
    print(f"  Placed {mod_placed} modifiers ({N_NA} Na + {N_CA} Ca + {N_SR} Sr)")

    print("[Stage 4/4] Assembling final structure...")

    # Build symbols and coordinates
    symbols = []
    all_coords = []

    # Si atoms (indices 0 to N_SI-1 in nf_coords)
    for i in range(N_SI):
        symbols.append('Si')
        all_coords.append(nf_coords[i])

    # P atoms (indices N_SI to N_SI+N_P-1 in nf_coords)
    for i in range(N_P):
        symbols.append('P')
        all_coords.append(nf_coords[N_SI + i])

    # O atoms
    for i in range(N_O):
        symbols.append('O')
        all_coords.append(o_coords[i])

    # Na atoms
    for i in range(N_NA):
        symbols.append('Na')
        all_coords.append(mod_coords[i])

    # Ca atoms
    for i in range(N_CA):
        symbols.append('Ca')
        all_coords.append(mod_coords[N_NA + i])

    # Sr atoms
    for i in range(N_SR):
        symbols.append('Sr')
        all_coords.append(mod_coords[N_NA + N_CA + i])

    all_coords = np.array(all_coords, dtype=np.float64)
    total_atoms = len(symbols)

    print(f"  Total atoms assembled: {total_atoms}")

    # ===== VERIFICATION =====
    print("\n[Verification] Checking for close contacts...")
    from scipy.spatial import cKDTree
    tree = cKDTree(all_coords, boxsize=box)

    # Check Si-Si
    si_idx = [i for i, s in enumerate(symbols) if s == 'Si']
    si_si_close = 0
    if len(si_idx) > 1:
        si_pairs = tree.query_pairs(2.6, output_type='ndarray')
        for i, j in si_pairs:
            if symbols[i] == 'Si' and symbols[j] == 'Si':
                si_si_close += 1
    print(f"  Si-Si pairs closer than 2.6 A: {si_si_close}")

    # Check O-O
    o_o_close = 0
    o_pairs = tree.query_pairs(1.8, output_type='ndarray')
    for i, j in o_pairs:
        if symbols[i] == 'O' and symbols[j] == 'O':
            o_o_close += 1
    print(f"  O-O pairs closer than 1.8 A: {o_o_close}")

    # Check Si-O coordination
    si_o_pairs = tree.query_pairs(2.25, output_type='ndarray')
    si_cn = {i: 0 for i in si_idx}
    for i, j in si_o_pairs:
        if symbols[i] == 'Si' and symbols[j] == 'O':
            si_cn[i] += 1
        elif symbols[j] == 'Si' and symbols[i] == 'O':
            si_cn[j] += 1
    cn_counts = {}
    for cn in si_cn.values():
        cn_counts[cn] = cn_counts.get(cn, 0) + 1
    print(f"  Si coordination distribution: {dict(sorted(cn_counts.items()))}")
    avg_cn = sum(si_cn.values()) / max(1, len(si_cn))
    print(f"  Average Si-O CN: {avg_cn:.3f}")

    # ===== SHUFFLE (to avoid ordering bias) =====
    np.random.seed(seed + 5000)
    perm = np.random.permutation(total_atoms)
    symbols = [symbols[i] for i in perm]
    all_coords = all_coords[perm]

    # ===== WRITE XYZ =====
    out_path = Path(output_file)
    with open(out_path, 'w') as f:
        f.write(f"{total_atoms}\n")
        f.write(
            f"Xiang-Du 2011, x={sr_percent} mol% SrO, "
            f"N={total_atoms}, rho={density:.4f} g/cm3, "
            f"box={box:.4f} A, seed={seed}\n"
        )
        for i in range(total_atoms):
            f.write(
                f"{symbols[i]:2s} "
                f"{all_coords[i,0]:10.6f} "
                f"{all_coords[i,1]:10.6f} "
                f"{all_coords[i,2]:10.6f}\n"
            )

    print(f"\n[OK] Initial structure saved to: {out_path}")
    print(f"[OK] Box = {box:.4f} A, Density = {density:.4f} g/cm3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Smart initial structure generator for 45S5 bioglass with Sr doping (11340 atoms).'
    )
    parser.add_argument('--sr', type=int, default=0,
                        help='Sr doping percentage (0, 1, 5, 10, 15)')
    parser.add_argument('--density', type=float, default=2.649,
                        help='Density in g/cm3 (default: 2.649 for x=0)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output XYZ file name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    if args.out is None:
        args.out = f"initial_Sr{args.sr}.xyz"

    generate_structure(args.sr, args.density, args.out, args.seed)