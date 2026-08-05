#!/usr/bin/env python3
"""
Optimized Initial Structure Generator for 45S5 Bioglass with Sr-doping.
Uses Numba (@njit) for ultra-fast non-overlapping coordinate generation.
Pair-specific minimum distances to prevent Si-Si close contacts.
Total atoms: 11340 (4x base cell of 2835)
Based on: Xiang & Du, Chem. Mater. 2011

Composition (11340 atoms):
  Si=1844, P=208, Na=1952, Ca+Sr=1076, O=6260
"""
import numpy as np
from numba import njit
import argparse
from pathlib import Path

NA = 6.02214076e23

# Scale factor: 4x the base 2835-atom cell from Xiang & Du 2011
SCALE = 4


@njit(fastmath=True, cache=True)
def generate_coords_numba(n_atoms, box, min_dist_matrix, type_indices, seed):
    """
    Generate non-overlapping coordinates with PAIR-SPECIFIC minimum distances.
    min_dist_matrix[i,j] = minimum distance between atom types i and j.
    """
    np.random.seed(seed)
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    half_box = box / 2.0
    placed = 0
    max_attempts = 50000  # Increased for larger system

    for i in range(n_atoms):
        success = False
        p = np.zeros(3, dtype=np.float64)
        ti = type_indices[i]

        for attempt in range(max_attempts):
            p[0] = np.random.uniform(0, box)
            p[1] = np.random.uniform(0, box)
            p[2] = np.random.uniform(0, box)

            valid = True

            for j in range(placed):
                tj = type_indices[j]
                min_d = min_dist_matrix[ti, tj]
                min_d_sq = min_d * min_d

                dx = p[0] - coords[j, 0]
                dy = p[1] - coords[j, 1]
                dz = p[2] - coords[j, 2]

                if dx > half_box: dx -= box
                elif dx < -half_box: dx += box
                if dy > half_box: dy -= box
                elif dy < -half_box: dy += box
                if dz > half_box: dz -= box
                elif dz < -half_box: dz += box

                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < min_d_sq:
                    valid = False
                    break

            if valid:
                coords[i, 0] = p[0]
                coords[i, 1] = p[1]
                coords[i, 2] = p[2]
                placed += 1
                success = True
                break

        if not success:
            coords[i, 0] = p[0]
            coords[i, 1] = p[1]
            coords[i, 2] = p[2]
            placed += 1

    return coords


def generate_structure(sr_percent: int, density: float, output_file: str, seed: int):
    # ===== BASE COMPOSITION (2835 atoms, Xiang & Du 2011) =====
    N_SI_BASE    = 461
    N_P_BASE     = 52
    N_NA_BASE    = 488
    N_CA_SR_BASE = 269
    N_O_BASE     = 1565

    assert (N_SI_BASE + N_P_BASE + N_NA_BASE + N_CA_SR_BASE + N_O_BASE) == 2835

    # ===== SCALED COMPOSITION (11340 atoms = 4 × 2835) =====
    N_SI          = N_SI_BASE * SCALE       # 1844
    N_P           = N_P_BASE * SCALE        # 208
    N_NA          = N_NA_BASE * SCALE       # 1952
    N_CA_SR_TOTAL = N_CA_SR_BASE * SCALE    # 1076
    N_O           = N_O_BASE * SCALE        # 6260
    N_TARGET      = N_SI + N_P + N_NA + N_CA_SR_TOTAL + N_O  # 11340

    # Sr doping: replace Ca with Sr
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

    # Create symbol list
    symbols = []
    for elem, count in counts.items():
        symbols.extend([elem] * count)

    np.random.seed(seed)
    np.random.shuffle(symbols)

    # Type map (MUST match Sr.py type_map)
    type_map = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Sr': 5}
    type_indices = np.array([type_map[s] for s in symbols], dtype=np.int32)

    # ===== PAIR-SPECIFIC MINIMUM DISTANCES (Angstrom) =====
    # Type order: Si=0, Ca=1, Na=2, P=3, O=4, Sr=5
    min_dist_matrix = np.array([
        # Si    Ca    Na    P     O     Sr
        [3.0,  2.5,  2.5,  3.0,  1.5,  2.5],  # Si
        [2.5,  2.5,  2.0,  2.5,  1.5,  2.5],  # Ca
        [2.5,  2.0,  2.0,  2.5,  1.5,  2.0],  # Na
        [3.0,  2.5,  2.5,  3.0,  1.5,  2.5],  # P
        [1.5,  1.5,  1.5,  1.5,  1.4,  1.5],  # O
        [2.5,  2.5,  2.0,  2.5,  1.5,  2.5],  # Sr
    ], dtype=np.float64)
    # ========================================================

    masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
              'P': 30.97376, 'O': 15.999, 'Sr': 87.62}
    total_mass = sum(masses[s] * c for s, c in counts.items())

    volume_A3 = (total_mass / NA) / density * 1e24
    box = volume_A3 ** (1/3)
    print(f"Box size: {box:.4f} Å (Density: {density} g/cm³)")
    print(f"Volume: {volume_A3:.2f} Å³")

    print(f"Generating {N_TARGET} coordinates with pair-specific minimum distances...")
    coords = generate_coords_numba(N_TARGET, box, min_dist_matrix, type_indices, seed)
    print("Coordinate generation complete.")

    # Write XYZ with metadata compatible with Sr.py parse_xyz_header
    out_path = Path(output_file)
    with open(out_path, 'w') as f:
        f.write(f"{N_TARGET}\n")
        f.write(
            f"Xiang-Du 2011, x={sr_percent} mol% SrO, "
            f"N={N_TARGET}, rho={density:.4f} g/cm3, "
            f"box={box:.4f} A, seed={seed}\n"
        )
        for i in range(N_TARGET):
            f.write(
                f"{symbols[i]:2s} "
                f"{coords[i,0]:10.6f} "
                f"{coords[i,1]:10.6f} "
                f"{coords[i,2]:10.6f}\n"
            )

    print(f"\n[OK] Initial structure saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate initial 45S5 glass structure with Sr doping (11340 atoms).'
    )
    parser.add_argument('--sr', type=int, default=0,
                        help='Sr doping percentage')
    parser.add_argument('--density', type=float, default=2.649,
                        help='Density in g/cm³ (default: 2.649 for 45S5)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output XYZ file name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    if args.out is None:
        args.out = f"initial_Sr{args.sr}.xyz"

    generate_structure(args.sr, args.density, args.out, args.seed)