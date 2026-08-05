#!/usr/bin/env python3
"""
Optimized Initial Structure Generator for 45S5 Bioglass with Sr-doping.
Based on: Xiang & Du, Chem. Mater. 2011, 23, 2703-2717

Features:
- Total atoms: 11340 (4x base cell of 2835)
- Pair-specific minimum distances to prevent Si-Si, P-P, O-O close contacts
- Cell-linked-list algorithm for fast generation of large systems
- Output format compatible with Sr.py v8.1

Composition (11340 atoms):
  Si=1844, P=208, Na=1952, Ca+Sr=1076, O=6260

Usage:
  python struture.py --sr 0 --density 2.649 --seed 42
  python struture.py --sr 1 --density 2.646 --seed 42
  python struture.py --sr 5 --density 2.712 --seed 42
  python struture.py --sr 10 --density 2.791 --seed 42
  python struture.py --sr 15 --density 2.845 --seed 42
"""
import numpy as np
from numba import njit
import argparse
from pathlib import Path

NA = 6.02214076e23
SCALE = 4  # 4x the base 2835-atom cell from Xiang & Du 2011


@njit(fastmath=True, cache=True)
def generate_coords_cell(n_atoms, box, min_dist_matrix, type_indices, seed):
    """
    Generate non-overlapping coordinates using cell-linked-list algorithm.
    PAIR-SPECIFIC minimum distances are enforced.
    min_dist_matrix[i,j] = minimum distance between atom types i and j.
    
    This is much faster than O(N^2) for large systems (N > 5000).
    """
    np.random.seed(seed)
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    half_box = box / 2.0
    placed = 0
    max_attempts = 100000
    
    # Determine the largest minimum distance to set cell size
    max_min_dist = 0.0
    for i in range(6):
        for j in range(6):
            if min_dist_matrix[i, j] > max_min_dist:
                max_min_dist = min_dist_matrix[i, j]
    
    # Cell size should be at least the maximum minimum distance
    cell_size = max_min_dist
    n_cells_1d = int(box / cell_size)
    if n_cells_1d < 1:
        n_cells_1d = 1
    cell_size = box / n_cells_1d
    n_cells = n_cells_1d * n_cells_1d * n_cells_1d
    
    # Cell linked list arrays
    head = np.full(n_cells, -1, dtype=np.int64)
    next_atom = np.full(n_atoms, -1, dtype=np.int64)
    
    for i in range(n_atoms):
        success = False
        p = np.zeros(3, dtype=np.float64)
        ti = type_indices[i]
        
        for attempt in range(max_attempts):
            p[0] = np.random.uniform(0, box)
            p[1] = np.random.uniform(0, box)
            p[2] = np.random.uniform(0, box)
            
            valid = True
            
            # Determine the cell of the new atom
            cx = int(p[0] / cell_size)
            cy = int(p[1] / cell_size)
            cz = int(p[2] / cell_size)
            if cx >= n_cells_1d: cx = n_cells_1d - 1
            if cy >= n_cells_1d: cy = n_cells_1d - 1
            if cz >= n_cells_1d: cz = n_cells_1d - 1
            
            # Check neighboring cells (3x3x3 = 27 cells)
            for dx in range(-1, 2):
                nx = (cx + dx + n_cells_1d) % n_cells_1d
                for dy in range(-1, 2):
                    ny = (cy + dy + n_cells_1d) % n_cells_1d
                    for dz in range(-1, 2):
                        nz = (cz + dz + n_cells_1d) % n_cells_1d
                        
                        cell_idx = nx * n_cells_1d * n_cells_1d + ny * n_cells_1d + nz
                        j = head[cell_idx]
                        
                        while j != -1:
                            tj = type_indices[j]
                            min_d = min_dist_matrix[ti, tj]
                            min_d_sq = min_d * min_d
                            
                            ddx = p[0] - coords[j, 0]
                            ddy = p[1] - coords[j, 1]
                            ddz = p[2] - coords[j, 2]
                            
                            if ddx > half_box: ddx -= box
                            elif ddx < -half_box: ddx += box
                            if ddy > half_box: ddy -= box
                            elif ddy < -half_box: ddy += box
                            if ddz > half_box: ddz -= box
                            elif ddz < -half_box: ddz += box
                            
                            dist_sq = ddx*ddx + ddy*ddy + ddz*ddz
                            if dist_sq < min_d_sq:
                                valid = False
                                break
                            
                            j = next_atom[j]
                        
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                coords[i, 0] = p[0]
                coords[i, 1] = p[1]
                coords[i, 2] = p[2]
                
                # Add to cell linked list
                cell_idx = cx * n_cells_1d * n_cells_1d + cy * n_cells_1d + cz
                next_atom[i] = head[cell_idx]
                head[cell_idx] = i
                
                placed += 1
                success = True
                break
        
        if not success:
            # Fallback: place anyway (should rarely happen)
            coords[i, 0] = p[0]
            coords[i, 1] = p[1]
            coords[i, 2] = p[2]
            
            cx = int(p[0] / cell_size)
            cy = int(p[1] / cell_size)
            cz = int(p[2] / cell_size)
            if cx >= n_cells_1d: cx = n_cells_1d - 1
            if cy >= n_cells_1d: cy = n_cells_1d - 1
            if cz >= n_cells_1d: cz = n_cells_1d - 1
            cell_idx = cx * n_cells_1d * n_cells_1d + cy * n_cells_1d + cz
            next_atom[i] = head[cell_idx]
            head[cell_idx] = i
            
            placed += 1
    
    return coords


def generate_structure(sr_percent: int, density: float, output_file: str, seed: int):
    # ===== BASE COMPOSITION (2835 atoms, Xiang & Du 2011) =====
    N_SI_BASE    = 461
    N_P_BASE     = 52
    N_NA_BASE    = 488
    N_CA_SR_BASE = 269
    N_O_BASE     = 1565
    
    # ===== SCALED COMPOSITION (11340 atoms = 4 x 2835) =====
    N_SI          = N_SI_BASE * SCALE       # 1844
    N_P           = N_P_BASE * SCALE        # 208
    N_NA          = N_NA_BASE * SCALE       # 1952
    N_CA_SR_TOTAL = N_CA_SR_BASE * SCALE    # 1076
    N_O           = N_O_BASE * SCALE        # 6260
    N_TARGET      = N_SI + N_P + N_NA + N_CA_SR_TOTAL + N_O  # 11340
    
    # Sr doping: replace Ca with Sr
    # Based on Xiang & Du 2011, Table 1:
    # x=0: Sr=0, x=1: Sr=40, x=5: Sr=200, x=10: Sr=400, x=15: Sr=600
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
    # These values prevent unphysical close contacts in the initial structure.
    min_dist_matrix = np.array([
        # Si    Ca    Na    P     O     Sr
        [3.0,  2.5,  2.5,  3.0,  1.5,  2.5],  # Si - Si-Si=3.0 prevents Si-Si bonds
        [2.5,  2.5,  2.0,  2.5,  1.5,  2.5],  # Ca
        [2.5,  2.0,  2.0,  2.5,  1.5,  2.0],  # Na
        [3.0,  2.5,  2.5,  3.0,  1.5,  2.5],  # P - P-P=3.0 prevents P-P bonds
        [1.5,  1.5,  1.5,  1.5,  2.2,  1.5],  # O - O-O=2.2 prevents O-O bonds
        [2.5,  2.5,  2.0,  2.5,  1.5,  2.5],  # Sr
    ], dtype=np.float64)
    # ========================================================
    
    masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
              'P': 30.97376, 'O': 15.999, 'Sr': 87.62}
    total_mass = sum(masses[s] * c for s, c in counts.items())
    
    volume_A3 = (total_mass / NA) / density * 1e24
    box = volume_A3 ** (1/3)
    print(f"Box size: {box:.4f} A (Density: {density} g/cm3)")
    print(f"Volume: {volume_A3:.2f} A3")
    
    print(f"Generating {N_TARGET} coordinates with cell-linked-list algorithm...")
    coords = generate_coords_cell(N_TARGET, box, min_dist_matrix, type_indices, seed)
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