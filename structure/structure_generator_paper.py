#!/usr/bin/env python3
"""
Initial structure generator for Sr-substituted 45S5 bioglass
Based on:

    Ye Xiang and Jincheng Du,
    "Effect of Strontium Substitution on the Structure of 45S5 Bioglasses",
    Chem. Mater. 2011, 23, 2703-2717.

Base composition:
    46.1 SiO2 - 24.4 Na2O - (26.9-x) CaO - 2.6 P2O5 - x SrO
    x = 0, 1, 5, 10, 15 mol%

Base cell:
    2835 atoms

Default target:
    11340 atoms = 4 * 2835

Usage examples:
    python structure_generator_paper.py --x 0 --n-atoms 11340
    python structure_generator_paper.py --x 5 --n-atoms 11340
    python structure_generator_paper.py --x 10 --scale 4
"""

import argparse
import numpy as np
from pathlib import Path
from numba import njit

# Avogadro constant
NA = 6.02214076e23

# =========================
# Paper base cell: 2835 atoms
# =========================
BASE_N_ATOMS = 2835

BASE_SI = 461
BASE_P = 52
BASE_NA = 488
BASE_CA_SR = 269
BASE_O = 1565

# Number of Sr atoms in the 2835-atom base cell for each x in paper
# x is mol% SrO in the oxide composition.
SR_ATOMS_BASE = {
    0: 0,
    1: 10,
    5: 50,
    10: 100,
    15: 150
}

# Final densities from Table 1 of the paper
PAPER_DENSITY = {
    0: 2.649,
    1: 2.646,
    5: 2.712,
    10: 2.791,
    15: 2.845
}

# Paper cubic cell sizes for 2835 atoms, Table 1
PAPER_BOX_2835 = {
    0: 33.81,
    1: 33.91,
    5: 33.97,
    10: 34.06,
    15: 34.24
}

# Atomic masses
MASSES = {
    'Si': 28.0855,
    'P': 30.97376,
    'Na': 22.98977,
    'Ca': 40.078,
    'Sr': 87.62,
    'O': 15.999
}

# Partial charges from paper
CHARGES = {
    'Si': 2.4,
    'P': 3.0,
    'Na': 0.6,
    'Ca': 1.2,
    'Sr': 1.2,
    'O': -1.2
}


@njit(fastmath=True, cache=True)
def generate_coords_cell(n_atoms, box, min_dist, seed, max_attempts):
    """
    Generate non-overlapping random coordinates using a cell-linked list.

    This is much faster than O(N^2) checking for large systems such as 11340 atoms.
    Periodic boundary conditions are applied for distance checks.
    """

    np.random.seed(seed)

    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    min_dist_sq = min_dist * min_dist

    # Number of cells per dimension.
    # cell_size will be >= min_dist, so checking 27 neighbor cells is sufficient.
    nx = int(box / min_dist)
    if nx < 1:
        nx = 1

    cell_size = box / nx
    inv_cell = 1.0 / cell_size
    n_cells = nx * nx * nx

    head = np.full(n_cells, -1, dtype=np.int64)
    next_idx = np.full(n_atoms, -1, dtype=np.int64)

    fallback_count = 0
    half_box = box * 0.5

    for i in range(n_atoms):
        success = False

        p0 = 0.0
        p1 = 0.0
        p2 = 0.0

        cx = 0
        cy = 0
        cz = 0

        for attempt in range(max_attempts):
            p0 = np.random.uniform(0.0, box)
            p1 = np.random.uniform(0.0, box)
            p2 = np.random.uniform(0.0, box)

            cx = int(p0 * inv_cell)
            cy = int(p1 * inv_cell)
            cz = int(p2 * inv_cell)

            if cx >= nx:
                cx = nx - 1
            if cy >= nx:
                cy = nx - 1
            if cz >= nx:
                cz = nx - 1

            valid = True

            # Check 27 neighboring cells
            for ox in range(-1, 2):
                nx_cell = cx + ox
                if nx_cell < 0:
                    nx_cell += nx
                elif nx_cell >= nx:
                    nx_cell -= nx

                base_x = nx_cell * nx * nx

                for oy in range(-1, 2):
                    ny_cell = cy + oy
                    if ny_cell < 0:
                        ny_cell += nx
                    elif ny_cell >= nx:
                        ny_cell -= nx

                    base_xy = base_x + ny_cell * nx

                    for oz in range(-1, 2):
                        nz_cell = cz + oz
                        if nz_cell < 0:
                            nz_cell += nx
                        elif nz_cell >= nx:
                            nz_cell -= nx

                        cell = base_xy + nz_cell
                        j = head[cell]

                        while j != -1:
                            dx = p0 - coords[j, 0]
                            dy = p1 - coords[j, 1]
                            dz = p2 - coords[j, 2]

                            # Minimum image convention
                            if dx > half_box:
                                dx -= box
                            elif dx < -half_box:
                                dx += box

                            if dy > half_box:
                                dy -= box
                            elif dy < -half_box:
                                dy += box

                            if dz > half_box:
                                dz -= box
                            elif dz < -half_box:
                                dz += box

                            dist_sq = dx * dx + dy * dy + dz * dz

                            if dist_sq < min_dist_sq:
                                valid = False
                                break

                            j = next_idx[j]

                        if not valid:
                            break

                    if not valid:
                        break

                if not valid:
                    break

            if valid:
                coords[i, 0] = p0
                coords[i, 1] = p1
                coords[i, 2] = p2

                cell = cx * nx * nx + cy * nx + cz
                next_idx[i] = head[cell]
                head[cell] = i

                success = True
                break

        if not success:
            # Fallback: place the last attempted point anyway.
            # This should be rare. If many fallbacks occur, reduce min_dist
            # or increase max_attempts.
            coords[i, 0] = p0
            coords[i, 1] = p1
            coords[i, 2] = p2

            cell = cx * nx * nx + cy * nx + cz
            next_idx[i] = head[cell]
            head[cell] = i

            fallback_count += 1

    return coords, fallback_count


def get_counts(x: int, scale: float):
    """
    Return atom counts for the requested x and scaling factor.

    x:
        mol% SrO in the paper formula:
        0, 1, 5, 10, 15

    scale:
        Scaling factor relative to the 2835-atom base cell.
        For 11340 atoms, scale = 4.
    """

    if x not in SR_ATOMS_BASE:
        raise ValueError("x must be one of: 0, 1, 5, 10, 15")

    sr_base = SR_ATOMS_BASE[x]
    ca_base = BASE_CA_SR - sr_base

    counts = {
        'Si': int(round(BASE_SI * scale)),
        'P': int(round(BASE_P * scale)),
        'Na': int(round(BASE_NA * scale)),
        'Ca': int(round(ca_base * scale)),
        'Sr': int(round(sr_base * scale)),
        'O': int(round(BASE_O * scale)),
    }

    target_total = int(round(BASE_N_ATOMS * scale))
    current_total = sum(counts.values())

    # Correct possible rounding errors by adjusting oxygen.
    if current_total != target_total:
        counts['O'] += target_total - current_total

    return counts, target_total


def check_charge_neutrality(counts):
    """
    Compute required oxygen charge for charge neutrality.
    For exact paper stoichiometry it should be -1.2.
    """

    total_positive = 0.0

    for elem in ['Si', 'P', 'Na', 'Ca', 'Sr']:
        total_positive += CHARGES[elem] * counts.get(elem, 0)

    n_o = counts.get('O', 0)

    if n_o <= 0:
        raise ValueError("Oxygen count must be positive.")

    q_o = -total_positive / n_o

    return q_o


def main():
    parser = argparse.ArgumentParser(
        description="Generate initial Sr-doped 45S5 bioglass structure "
                    "based on Xiang & Du, Chem. Mater. 2011."
    )

    parser.add_argument(
        "--x",
        type=int,
        default=0,
        choices=[0, 1, 5, 10, 15],
        help="mol%% SrO in the paper composition: 0, 1, 5, 10, 15"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="Scaling factor relative to 2835-atom base cell. Default: 4 => 11340 atoms."
    )

    parser.add_argument(
        "--n-atoms",
        type=int,
        default=None,
        help="Target total number of atoms. Overrides --scale if provided."
    )

    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Density in g/cm^3. If not given, paper Table 1 density is used."
    )

    parser.add_argument(
        "--min-dist",
        type=float,
        default=1.2,
        help="Minimum allowed distance between atoms in Angstrom during random placement."
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=200000,
        help="Maximum random attempts per atom."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output XYZ file name."
    )

    args = parser.parse_args()

    # Determine scale
    if args.n_atoms is not None:
        scale = args.n_atoms / BASE_N_ATOMS
    else:
        scale = args.scale

    # If scale is very close to an integer, use exact integer scaling
    scale_rounded = round(scale)
    if abs(scale - scale_rounded) < 1.0e-9:
        scale = int(scale_rounded)
    else:
        print("\nWARNING:")
        print(f"Scale factor {scale:.6f} is not an integer multiple of the base cell.")
        print("Composition will be approximate due to rounding.\n")

    counts, target_total = get_counts(args.x, scale)

    if args.n_atoms is not None and target_total != args.n_atoms:
        print("\nWARNING:")
        print(f"Requested {args.n_atoms} atoms, but adjusted to {target_total} atoms")
        print("to preserve integer stoichiometry.\n")

    # Density
    if args.density is not None:
        density = args.density
        density_source = "user-provided"
    else:
        density = PAPER_DENSITY[args.x]
        density_source = "paper Table 1"

    # Mass and box
    total_mass = sum(MASSES[elem] * counts[elem] for elem in counts)

    volume_A3 = (total_mass / NA) / density * 1.0e24
    box = volume_A3 ** (1.0 / 3.0)

    # Charge neutrality
    q_o = check_charge_neutrality(counts)

    # Expected box size from paper cell size, if using paper density
    expected_box = None
    if args.density is None:
        expected_box = PAPER_BOX_2835[args.x] * (target_total / BASE_N_ATOMS) ** (1.0 / 3.0)

    # Print summary
    print("=" * 70)
    print("Initial structure generator based on Xiang & Du, Chem. Mater. 2011")
    print("=" * 70)
    print(f"Composition series : 46.1SiO2-24.4Na2O-(26.9-x)CaO-2.6P2O5-xSrO")
    print(f"x (mol% SrO)       : {args.x}")
    print(f"Base atoms         : {BASE_N_ATOMS}")
    print(f"Scale factor       : {scale}")
    print(f"Target atoms       : {target_total}")
    print(f"Density source     : {density_source}")
    print(f"Density            : {density:.4f} g/cm^3")
    print(f"Box size           : {box:.4f} Å")

    if expected_box is not None:
        print(f"Expected paper box : {expected_box:.4f} Å")

    print("\nAtom counts:")
    for elem in ['Si', 'P', 'Na', 'Ca', 'Sr', 'O']:
        print(f"  {elem:3s}: {counts[elem]:6d}")

    print("\nCharge check:")
    print(f"  Required O charge: {q_o:.6f}")

    if abs(q_o - (-1.2)) < 1.0e-6:
        print("  Charge neutrality consistent with paper O charge = -1.2")
    else:
        print("  WARNING: Oxygen charge is not exactly -1.2.")
        print("           This may be due to non-integer scaling or rounding.")

    # Build symbol list
    symbols = []
    for elem in ['Si', 'P', 'Na', 'Ca', 'Sr', 'O']:
        symbols.extend([elem] * counts[elem])

    symbols = np.array(symbols)

    # Shuffle chemical species
    rng = np.random.default_rng(args.seed)
    rng.shuffle(symbols)

    # Generate coordinates
    print("\nGenerating coordinates using cell-linked-list algorithm...")
    coords, fallback_count = generate_coords_cell(
        n_atoms=target_total,
        box=box,
        min_dist=args.min_dist,
        seed=args.seed + 10000,
        max_attempts=args.max_attempts
    )

    print(f"Coordinate generation completed.")
    print(f"Fallback placements: {fallback_count}")

    if fallback_count > 0:
        print("\nWARNING:")
        print(f"{fallback_count} atoms were placed without satisfying min_dist.")
        print("If this number is large, reduce --min-dist or increase --max-attempts.")

    # Output file
    if args.out is None:
        out_file = f"initial_x{args.x}_N{target_total}_rho{density:.3f}_seed{args.seed}.xyz"
    else:
        out_file = args.out

    out_path = Path(out_file)

    comment = (
        f"Xiang-Du 2011, x={args.x} mol% SrO, "
        f"N={target_total}, rho={density:.4f} g/cm3, box={box:.4f} A, "
        f"seed={args.seed}"
    )

    with open(out_path, 'w') as f:
        f.write(f"{target_total}\n")
        f.write(f"{comment}\n")

        for i in range(target_total):
            f.write(
                f"{symbols[i]:2s} "
                f"{coords[i, 0]:12.6f} "
                f"{coords[i, 1]:12.6f} "
                f"{coords[i, 2]:12.6f}\n"
            )

    print(f"\n[OK] Structure saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()