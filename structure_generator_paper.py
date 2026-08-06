#!/usr/bin/env python3
"""
========================================================================
Smart Initial Structure Generator for 45S5 Bioglass with Sr-doping
Version: v5.0 (Professional Edition)
Based on: Xiang & Du, Chem. Mater. 2011

NEW FEATURES (v5.0):
  1. Staged placement: NF -> O (tetrahedral) -> Modifiers
  2. RANDOM TETRAHEDRON ORIENTATION (prevents artificial ordering)
  3. Affinity-aware modifier placement (Ca/Sr prefer P neighborhood)
  4. Comprehensive verification with CN distribution
  5. Support for 2835/5670/11340 atom systems
  6. Optional pre-relaxation report

Usage:
  python struture.py --sr 0  --density 2.649 --seed 42
  python struture.py --sr 5  --density 2.712 --seed 42
  python struture.py --sr 10 --density 2.791 --seed 42
  python struture.py --sr 15 --density 2.845 --seed 42
  python struture.py --sr 0  --density 2.649 --scale 2  (5670 atoms)
========================================================================
"""
import numpy as np
from numba import njit
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

NA = 6.02214076e23
BASE_N_ATOMS = 2835

# Type indices (MUST match Sr.py)
# Si=0, Ca=1, Na=2, P=3, O=4, Sr=5


# ================================================================
# NUMBA KERNELS
# ================================================================

@njit(fastmath=True, cache=True)
def _random_rotation_matrix(seed_val):
    """Generate a random 3D rotation matrix (uniform SO(3))."""
    np.random.seed(seed_val)
    # Random unit quaternion
    u1 = np.random.random()
    u2 = np.random.random()
    u3 = np.random.random()
    q0 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q1 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    # Convert quaternion to rotation matrix
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2*(q2*q2 + q3*q3)
    R[0, 1] = 2*(q1*q2 - q0*q3)
    R[0, 2] = 2*(q1*q3 + q0*q2)
    R[1, 0] = 2*(q1*q2 + q0*q3)
    R[1, 1] = 1 - 2*(q1*q1 + q3*q3)
    R[1, 2] = 2*(q2*q3 - q0*q1)
    R[2, 0] = 2*(q1*q3 - q0*q2)
    R[2, 1] = 2*(q2*q3 + q0*q1)
    R[2, 2] = 1 - 2*(q1*q1 + q2*q2)
    return R


@njit(fastmath=True, cache=True)
def place_network_formers(n_si, n_p, box, min_nf_dist, seed):
    """Stage 1: Place Si and P with minimum NF-NF distance."""
    np.random.seed(seed)
    n_total = n_si + n_p
    coords = np.zeros((n_total, 3), dtype=np.float64)
    min_d_sq = min_nf_dist * min_nf_dist
    placed = 0
    half_box = box / 2.0
    max_attempts = 300000

    for i in range(n_total):
        success = False
        for attempt in range(max_attempts):
            px = np.random.uniform(0, box)
            py = np.random.uniform(0, box)
            pz = np.random.uniform(0, box)
            valid = True
            for j in range(placed):
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
    """
    Stage 2: Place O around Si/P in TETRAHEDRAL geometry
    with RANDOM ORIENTATION per tetrahedron.
    """
    np.random.seed(seed + 1000)
    n_o_needed = n_nf * 4
    o_coords = np.zeros((n_o_needed, 3), dtype=np.float64)
    o_placed = 0
    half_box = box / 2.0
    o_min_d_sq = o_min_dist * o_min_dist
    max_attempts = 100000

    # Reference tetrahedron directions (before rotation)
    tet_ref = np.array([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0]
    ], dtype=np.float64)
    # Normalize
    for d in range(4):
        norm = np.sqrt(tet_ref[d, 0]**2 + tet_ref[d, 1]**2 + tet_ref[d, 2]**2)
        tet_ref[d, 0] /= norm
        tet_ref[d, 1] /= norm
        tet_ref[d, 2] /= norm

    for nf in range(n_nf):
        nf_x = nf_coords[nf, 0]
        nf_y = nf_coords[nf, 1]
        nf_z = nf_coords[nf, 2]

        bond_dist = o_si_dist if nf < n_si else o_p_dist

        # === NEW: Random rotation per tetrahedron ===
        R = _random_rotation_matrix(seed + 2000 + nf)

        for k in range(4):
            # Rotate tetrahedron direction
            dx_t = R[0,0]*tet_ref[k,0] + R[0,1]*tet_ref[k,1] + R[0,2]*tet_ref[k,2]
            dy_t = R[1,0]*tet_ref[k,0] + R[1,1]*tet_ref[k,1] + R[1,2]*tet_ref[k,2]
            dz_t = R[2,0]*tet_ref[k,0] + R[2,1]*tet_ref[k,1] + R[2,2]*tet_ref[k,2]

            # Add small perturbation (thermal disorder)
            dx_t += 0.10 * (np.random.random() - 0.5)
            dy_t += 0.10 * (np.random.random() - 0.5)
            dz_t += 0.10 * (np.random.random() - 0.5)

            px = (nf_x + bond_dist * dx_t) % box
            py = (nf_y + bond_dist * dy_t) % box
            pz = (nf_z + bond_dist * dz_t) % box

            # Check O-O minimum distance
            valid = True
            for j in range(o_placed):
                ddx = px - o_coords[j, 0]
                ddy = py - o_coords[j, 1]
                ddz = pz - o_coords[j, 2]
                if ddx > half_box: ddx -= box
                elif ddx < -half_box: ddx += box
                if ddy > half_box: ddy -= box
                elif ddy < -half_box: ddy += box
                if ddz > half_box: ddz -= box
                elif ddz < -half_box: ddz += box
                if ddx*ddx + ddy*ddy + ddz*ddz < o_min_d_sq:
                    valid = False
                    break

            if valid:
                o_coords[o_placed, 0] = px
                o_coords[o_placed, 1] = py
                o_coords[o_placed, 2] = pz
                o_placed += 1
            else:
                # Fallback: random direction
                for attempt in range(max_attempts):
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
                        ddx = px - o_coords[j, 0]
                        ddy = py - o_coords[j, 1]
                        ddz = pz - o_coords[j, 2]
                        if ddx > half_box: ddx -= box
                        elif ddx < -half_box: ddx += box
                        if ddy > half_box: ddy -= box
                        elif ddy < -half_box: ddy += box
                        if ddz > half_box: ddz -= box
                        elif ddz < -half_box: ddz += box
                        if ddx*ddx + ddy*ddy + ddz*ddz < o_min_d_sq:
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
def place_modifiers_affinity(n_mod, box, all_coords, n_all,
                             mod_o_min_dist, mod_mod_min_dist,
                             p_coords, n_p, p_attraction_radius, seed):
    """
    Stage 3: Place modifiers with AFFINITY-AWARE placement.
    Ca/Sr prefer to be near P (modifier scavenger effect).
    Na distributes more uniformly.
    """
    np.random.seed(seed + 3000)
    half_box = box / 2.0
    mod_o_min_d_sq = mod_o_min_dist * mod_o_min_dist
    mod_mod_min_d_sq = mod_mod_min_dist * mod_mod_min_dist
    placed = 0
    max_attempts = 300000
    mod_coords = np.zeros((n_mod, 3), dtype=np.float64)

    for i in range(n_mod):
        success = False
        for attempt in range(max_attempts):
            # 30% chance: try to place near a P atom (affinity)
            if n_p > 0 and np.random.random() < 0.30:
                p_idx = np.random.randint(0, n_p)
                theta = np.random.uniform(0, 2.0 * np.pi)
                phi = np.arccos(2.0 * np.random.random() - 1.0)
                r = np.random.uniform(3.0, p_attraction_radius)
                px = (p_coords[p_idx, 0] + r * np.sin(phi) * np.cos(theta)) % box
                py = (p_coords[p_idx, 1] + r * np.sin(phi) * np.sin(theta)) % box
                pz = (p_coords[p_idx, 2] + r * np.cos(phi)) % box
            else:
                px = np.random.uniform(0, box)
                py = np.random.uniform(0, box)
                pz = np.random.uniform(0, box)

            # Check distance to NF+O
            valid = True
            for j in range(n_all):
                dx = px - all_coords[j, 0]
                dy = py - all_coords[j, 1]
                dz = pz - all_coords[j, 2]
                if dx > half_box: dx -= box
                elif dx < -half_box: dx += box
                if dy > half_box: dy -= box
                elif dy < -half_box: dy += box
                if dz > half_box: dz -= box
                elif dz < -half_box: dz += box
                if dx*dx + dy*dy + dz*dz < mod_o_min_d_sq:
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


# ================================================================
# VERIFICATION
# ================================================================
def verify_structure(symbols, coords, box):
    """Comprehensive structure verification."""
    print("\n" + "=" * 60)
    print("  STRUCTURE VERIFICATION REPORT")
    print("=" * 60)

    tree = cKDTree(coords, boxsize=box)
    n = len(symbols)

    # 1. Close contact check
    print("\n--- Close Contact Check ---")
    checks = [
        ('Si-Si', 'Si', 'Si', 2.6),
        ('O-O',   'O', 'O',   1.8),
        ('Si-O',  'Si', 'O',  1.3),
        ('P-O',   'P', 'O',   1.3),
    ]
    for name, e1, e2, threshold in checks:
        idx1 = [i for i, s in enumerate(symbols) if s == e1]
        idx2 = [i for i, s in enumerate(symbols) if s == e2]
        if not idx1 or not idx2:
            continue
        close = 0
        pairs = tree.query_pairs(threshold, output_type='ndarray')
        for i, j in pairs:
            if symbols[i] == e1 and symbols[j] == e2:
                close += 1
            elif symbols[i] == e2 and symbols[j] == e1:
                close += 1
        status = "✅ PASS" if close == 0 else f"⚠️  {close} violations"
        print(f"  {name:8s} < {threshold:.1f} A: {status}")

    # 2. Si coordination distribution
    print("\n--- Si Coordination Distribution ---")
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
    print("\n--- P Coordination Distribution ---")
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

    # 4. Modifier-O distances
    print("\n--- Modifier-O Average Distances ---")
    for mod, cut in [('Na', 3.34), ('Ca', 3.14), ('Sr', 3.35)]:
        mod_idx = [i for i, s in enumerate(symbols) if s == mod]
        if not mod_idx:
            continue
        dists = []
        for m in mod_idx:
            neigh = tree.query_ball_point(coords[m], cut)
            for j in neigh:
                if j != m and j in o_idx_set:
                    d = coords[m] - coords[j]
                    d -= box * np.round(d / box)
                    dists.append(np.linalg.norm(d))
        if dists:
            print(f"  {mod}-O: mean={np.mean(dists):.3f} A, "
                  f"min={np.min(dists):.3f}, max={np.max(dists):.3f}, "
                  f"avg_CN={len(dists)/len(mod_idx):.2f}")

    print("=" * 60)


# ================================================================
# MAIN GENERATOR
# ================================================================
def generate_structure(sr_percent, density, output_file, seed, scale=4):
    # ===== COMPOSITION =====
    N_SI          = 461 * scale
    N_P           = 52 * scale
    N_NA          = 488 * scale
    N_CA_SR_TOTAL = 269 * scale
    N_O           = 1565 * scale
    N_TARGET      = N_SI + N_P + N_NA + N_CA_SR_TOTAL + N_O

    N_SR = int(round(N_CA_SR_TOTAL * (sr_percent / 100.0)))
    N_CA = N_CA_SR_TOTAL - N_SR

    counts = {'Si': N_SI, 'P': N_P, 'Na': N_NA, 'Ca': N_CA, 'Sr': N_SR, 'O': N_O}
    actual_total = sum(counts.values())

    print(f"\nComposition for {sr_percent}% Sr doping (SCALE={scale}x):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL: {actual_total}")

    masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
              'P': 30.97376, 'O': 15.999, 'Sr': 87.62}
    total_mass = sum(masses[s] * c for s, c in counts.items())
    volume_A3 = (total_mass / NA) / density * 1e24
    box = volume_A3 ** (1/3)
    print(f"Box size: {box:.4f} A (Density: {density} g/cm3)")

    # ===== STAGE 1: Network Formers =====
    print(f"\n[Stage 1/4] Placing {N_SI+N_P} network formers (min NF-NF = 3.0 A)...")
    nf_coords, nf_placed = place_network_formers(
        N_SI, N_P, box, min_nf_dist=3.0, seed=seed
    )
    print(f"  Placed {nf_placed} NF ({N_SI} Si + {N_P} P)")

    # ===== STAGE 2: Oxygens (Tetrahedral + Random Orientation) =====
    print("[Stage 2/4] Placing O in tetrahedral geometry (RANDOM ORIENTATION)...")
    o_coords, o_placed = place_oxygens_around_nf(
        nf_coords, nf_placed, box,
        o_si_dist=1.61, o_p_dist=1.54,
        n_si=N_SI, o_min_dist=2.0, seed=seed
    )
    print(f"  Placed {o_placed} O (target: {N_O})")

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
    print("[Stage 3/4] Placing modifiers (AFFINITY-AWARE: Ca/Sr prefer P)...")
    all_nf_o = np.vstack((nf_coords, o_coords[:N_O]))
    n_all = nf_placed + N_O
    N_MOD = N_NA + N_CA + N_SR
    # P coords for affinity placement
    p_coords = nf_coords[N_SI:N_SI + N_P]

    mod_coords, mod_placed = place_modifiers_affinity(
        N_MOD, box, all_nf_o, n_all,
        mod_o_min_dist=2.0, mod_mod_min_dist=2.0,
        p_coords=p_coords, n_p=N_P,
        p_attraction_radius=6.0, seed=seed
    )
    print(f"  Placed {mod_placed} modifiers ({N_NA} Na + {N_CA} Ca + {N_SR} Sr)")

    # ===== STAGE 4: Assembly =====
    print("[Stage 4/4] Assembling final structure...")
    symbols = []
    all_coords = []

    for i in range(N_SI):
        symbols.append('Si'); all_coords.append(nf_coords[i])
    for i in range(N_P):
        symbols.append('P'); all_coords.append(nf_coords[N_SI + i])
    for i in range(N_O):
        symbols.append('O'); all_coords.append(o_coords[i])
    for i in range(N_NA):
        symbols.append('Na'); all_coords.append(mod_coords[i])
    for i in range(N_CA):
        symbols.append('Ca'); all_coords.append(mod_coords[N_NA + i])
    for i in range(N_SR):
        symbols.append('Sr'); all_coords.append(mod_coords[N_NA + N_CA + i])

    all_coords = np.array(all_coords, dtype=np.float64)
    total_atoms = len(symbols)
    print(f"  Total atoms: {total_atoms}")

    # ===== VERIFICATION =====
    verify_structure(symbols, all_coords, box)

    # ===== SHUFFLE =====
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
            f.write(f"{symbols[i]:2s} "
                    f"{all_coords[i,0]:10.6f} "
                    f"{all_coords[i,1]:10.6f} "
                    f"{all_coords[i,2]:10.6f}\n")

    print(f"\n[OK] Initial structure saved to: {out_path}")
    print(f"[OK] Box = {box:.4f} A, Density = {density:.4f} g/cm3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Smart initial structure generator v5.0 for 45S5 bioglass'
    )
    parser.add_argument('--sr', type=int, default=0,
                        help='Sr doping percentage (0, 1, 5, 10, 15)')
    parser.add_argument('--density', type=float, default=2.649,
                        help='Density in g/cm3')
    parser.add_argument('--out', type=str, default=None,
                        help='Output XYZ file name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--scale', type=int, default=4, choices=[1, 2, 4],
                        help='Scale factor: 1=2835, 2=5670, 4=11340 atoms')
    args = parser.parse_args()

    if args.out is None:
        args.out = f"initial_Sr{args.sr}.xyz"

    generate_structure(args.sr, args.density, args.out, args.seed, args.scale)