#!/usr/bin/env python3
"""
========================================================================
Smart Initial Structure Generator for Mg-doped 45S5 Bioglass
Based on: Moghanian et al., Mg-doped 45S5 bioglass manuscript
Version: Mg-paper-4x

IMPORTANT:
- Compositions and densities are taken exactly from Table 2 of the paper.
- Default scale = 4, therefore total atoms = 4 * 2835 = 11340.
- Partial charges are fixed according to the paper:
  Si +2.4, Ca +1.2, Na +0.6, P +3.0, Mg +1.2, O -1.2

Example:
python structure_generator_paper_mg.py --mg 0  --seed 42
python structure_generator_paper_mg.py --mg 5  --seed 42
python structure_generator_paper_mg.py --mg 20 --seed 42

--mg values correspond to paper labels:
0  -> 45-M0
1  -> 45-M1
3  -> 45-M3
5  -> 45-M5
8  -> 45-M8
10 -> 45-M10
15 -> 45-M15
20 -> 45-M20
========================================================================
"""

import numpy as np
from numba import njit
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

NA = 6.02214076e23
BASE_N_ATOMS = 2835

# ================================================================
# PAPER TABLE 2: composition for 2835 atoms and densities
# ================================================================
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


# ================================================================
# NUMBA KERNELS
# ================================================================
@njit(fastmath=True, cache=True)
def _random_rotation_matrix(seed_val):
    """Generate a random 3D rotation matrix (uniform SO(3))."""
    np.random.seed(seed_val)

    u1 = np.random.random()
    u2 = np.random.random()
    u3 = np.random.random()

    q0 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q1 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q2 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q3 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    R[0, 1] = 2.0 * (q1 * q2 - q0 * q3)
    R[0, 2] = 2.0 * (q1 * q3 + q0 * q2)

    R[1, 0] = 2.0 * (q1 * q2 + q0 * q3)
    R[1, 1] = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    R[1, 2] = 2.0 * (q2 * q3 - q0 * q1)

    R[2, 0] = 2.0 * (q1 * q3 - q0 * q2)
    R[2, 1] = 2.0 * (q2 * q3 + q0 * q1)
    R[2, 2] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)

    return R


@njit(fastmath=True, cache=True)
def place_network_formers(n_si, n_p, box, min_nf_dist, seed):
    """
    Stage 1:
    Place Si and P network formers with a minimum NF-NF distance.
    """
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
            px = np.random.uniform(0.0, box)
            py = np.random.uniform(0.0, box)
            pz = np.random.uniform(0.0, box)

            valid = True
            for j in range(placed):
                dx = px - coords[j, 0]
                dy = py - coords[j, 1]
                dz = pz - coords[j, 2]

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

                if dx * dx + dy * dy + dz * dz < min_d_sq:
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
            coords[placed, 0] = np.random.uniform(0.0, box)
            coords[placed, 1] = np.random.uniform(0.0, box)
            coords[placed, 2] = np.random.uniform(0.0, box)
            placed += 1

    return coords, placed


@njit(fastmath=True, cache=True)
def place_oxygens_around_nf_target(
    nf_coords,
    n_nf,
    n_o_target,
    box,
    o_si_dist,
    o_p_dist,
    n_si,
    o_min_dist,
    seed
):
    """
    Stage 2:
    Place exactly n_o_target oxygens around Si/P network formers.

    For 45S5:
    O / (Si+P) ~= 1565 / 513 ~= 3.05

    Therefore most network formers receive 3 oxygens and a smaller
    fraction receives 4 oxygens. This avoids the artificial problem
    of assigning 4 oxygens to every tetrahedron and then truncating
    the last oxygens.
    """
    np.random.seed(seed + 1000)

    o_coords = np.zeros((n_o_target, 3), dtype=np.float64)
    o_placed = 0

    half_box = box / 2.0
    o_min_d_sq = o_min_dist * o_min_dist
    max_attempts = 100000

    # Reference tetrahedron directions
    tet_ref = np.zeros((4, 3), dtype=np.float64)
    tet_ref[0, 0] =  1.0
    tet_ref[0, 1] =  1.0
    tet_ref[0, 2] =  1.0

    tet_ref[1, 0] =  1.0
    tet_ref[1, 1] = -1.0
    tet_ref[1, 2] = -1.0

    tet_ref[2, 0] = -1.0
    tet_ref[2, 1] =  1.0
    tet_ref[2, 2] = -1.0

    tet_ref[3, 0] = -1.0
    tet_ref[3, 1] = -1.0
    tet_ref[3, 2] =  1.0

    for d in range(4):
        norm = np.sqrt(
            tet_ref[d, 0] * tet_ref[d, 0] +
            tet_ref[d, 1] * tet_ref[d, 1] +
            tet_ref[d, 2] * tet_ref[d, 2]
        )
        tet_ref[d, 0] /= norm
        tet_ref[d, 1] /= norm
        tet_ref[d, 2] /= norm

    # Distribute target oxygens among network formers:
    # base = 3 for 45S5, remainder receives +1
    base = n_o_target // n_nf
    rem = n_o_target - base * n_nf

    o_per_nf = np.empty(n_nf, dtype=np.int64)
    for i in range(n_nf):
        o_per_nf[i] = base

    placed_extra = 0
    while placed_extra < rem:
        idx = np.random.randint(0, n_nf)
        if o_per_nf[idx] == base:
            o_per_nf[idx] = base + 1
            placed_extra += 1

    for nf in range(n_nf):
        nf_x = nf_coords[nf, 0]
        nf_y = nf_coords[nf, 1]
        nf_z = nf_coords[nf, 2]

        bond_dist = o_si_dist if nf < n_si else o_p_dist

        # Random orientation for each local tetrahedral unit
        R = _random_rotation_matrix(seed + 2000 + nf)

        n_local = o_per_nf[nf]

        for k in range(n_local):
            if k < 4:
                dx_t = (
                    R[0, 0] * tet_ref[k, 0] +
                    R[0, 1] * tet_ref[k, 1] +
                    R[0, 2] * tet_ref[k, 2]
                )
                dy_t = (
                    R[1, 0] * tet_ref[k, 0] +
                    R[1, 1] * tet_ref[k, 1] +
                    R[1, 2] * tet_ref[k, 2]
                )
                dz_t = (
                    R[2, 0] * tet_ref[k, 0] +
                    R[2, 1] * tet_ref[k, 1] +
                    R[2, 2] * tet_ref[k, 2]
                )
            else:
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                phi = np.arccos(2.0 * np.random.random() - 1.0)
                dx_t = np.sin(phi) * np.cos(theta)
                dy_t = np.sin(phi) * np.sin(theta)
                dz_t = np.cos(phi)

            # Small thermal perturbation
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

                if ddx > half_box:
                    ddx -= box
                elif ddx < -half_box:
                    ddx += box

                if ddy > half_box:
                    ddy -= box
                elif ddy < -half_box:
                    ddy += box

                if ddz > half_box:
                    ddz -= box
                elif ddz < -half_box:
                    ddz += box

                if ddx * ddx + ddy * ddy + ddz * ddz < o_min_d_sq:
                    valid = False
                    break

            if not valid:
                # Fallback: random direction at the same bond distance
                for attempt in range(max_attempts):
                    theta = np.random.uniform(0.0, 2.0 * np.pi)
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

                        if ddx > half_box:
                            ddx -= box
                        elif ddx < -half_box:
                            ddx += box

                        if ddy > half_box:
                            ddy -= box
                        elif ddy < -half_box:
                            ddy += box

                        if ddz > half_box:
                            ddz -= box
                        elif ddz < -half_box:
                            ddz += box

                        if ddx * ddx + ddy * ddy + ddz * ddz < o_min_d_sq:
                            valid2 = False
                            break

                    if valid2:
                        break

            o_coords[o_placed, 0] = px
            o_coords[o_placed, 1] = py
            o_coords[o_placed, 2] = pz
            o_placed += 1

    return o_coords, o_placed


@njit(fastmath=True, cache=True)
def place_modifiers_affinity(
    n_mod,
    box,
    all_coords,
    n_all,
    mod_o_min_dist,
    mod_mod_min_dist,
    p_coords,
    n_p,
    p_attraction_radius,
    seed
):
    """
    Stage 3:
    Place modifiers.

    Ca/Mg have affinity for phosphate environments in many 45S5-type
    models; therefore a fraction of modifier atoms is placed near P.
    Na is placed more uniformly, but here the same generic routine is
    used for all modifiers for robust initial packing.
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
            # 30% chance: place near a P atom
            if n_p > 0 and np.random.random() < 0.30:
                p_idx = np.random.randint(0, n_p)

                theta = np.random.uniform(0.0, 2.0 * np.pi)
                phi = np.arccos(2.0 * np.random.random() - 1.0)
                r = np.random.uniform(3.0, p_attraction_radius)

                px = (p_coords[p_idx, 0] + r * np.sin(phi) * np.cos(theta)) % box
                py = (p_coords[p_idx, 1] + r * np.sin(phi) * np.sin(theta)) % box
                pz = (p_coords[p_idx, 2] + r * np.cos(phi)) % box
            else:
                px = np.random.uniform(0.0, box)
                py = np.random.uniform(0.0, box)
                pz = np.random.uniform(0.0, box)

            # Check distance to already placed NF + O atoms
            valid = True
            for j in range(n_all):
                dx = px - all_coords[j, 0]
                dy = py - all_coords[j, 1]
                dz = pz - all_coords[j, 2]

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

                if dx * dx + dy * dy + dz * dz < mod_o_min_d_sq:
                    valid = False
                    break

            if not valid:
                continue

            # Check distance to already placed modifiers
            for j in range(placed):
                dx = px - mod_coords[j, 0]
                dy = py - mod_coords[j, 1]
                dz = pz - mod_coords[j, 2]

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

                if dx * dx + dy * dy + dz * dz < mod_mod_min_d_sq:
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
            mod_coords[placed, 0] = np.random.uniform(0.0, box)
            mod_coords[placed, 1] = np.random.uniform(0.0, box)
            mod_coords[placed, 2] = np.random.uniform(0.0, box)
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

    # 1. Close contact check
    print("\n--- Close Contact Check ---")
    checks = [
        ("Si-Si", "Si", "Si", 2.6),
        ("O-O",   "O",   "O",   1.8),
        ("Si-O",  "Si",  "O",   1.3),
        ("P-O",   "P",   "O",   1.3),
        ("Mg-O",  "Mg",  "O",   1.3),
    ]

    for name, e1, e2, threshold in checks:
        idx1 = [i for i, s in enumerate(symbols) if s == e1]
        idx2 = [i for i, s in enumerate(symbols) if s == e2]

        if not idx1 or not idx2:
            continue

        close = 0
        pairs = tree.query_pairs(threshold, output_type="ndarray")

        for i, j in pairs:
            if symbols[i] == e1 and symbols[j] == e2:
                close += 1
            elif symbols[i] == e2 and symbols[j] == e1:
                close += 1

        status = "PASS" if close == 0 else f"{close} violations"
        print(f"  {name:8s} < {threshold:.1f} A: {status}")

    # 2. Si coordination distribution
    print("\n--- Si Coordination Distribution ---")
    si_idx = [i for i, s in enumerate(symbols) if s == "Si"]
    o_idx_set = set(i for i, s in enumerate(symbols) if s == "O")

    si_cn = {}
    for si in si_idx:
        neigh = tree.query_ball_point(coords[si], 2.25)
        cn = sum(1 for j in neigh if j != si and j in o_idx_set)
        si_cn[cn] = si_cn.get(cn, 0) + 1

    for cn in sorted(si_cn.keys()):
        pct = si_cn[cn] / max(1, len(si_idx)) * 100.0
        bar = "#" * int(pct / 2)
        print(f"  CN={cn}: {si_cn[cn]:5d} ({pct:5.1f}%) {bar}")

    # 3. P coordination distribution
    print("\n--- P Coordination Distribution ---")
    p_idx = [i for i, s in enumerate(symbols) if s == "P"]

    p_cn = {}
    for p in p_idx:
        neigh = tree.query_ball_point(coords[p], 2.25)
        cn = sum(1 for j in neigh if j != p and j in o_idx_set)
        p_cn[cn] = p_cn.get(cn, 0) + 1

    for cn in sorted(p_cn.keys()):
        pct = p_cn[cn] / max(1, len(p_idx)) * 100.0
        bar = "#" * int(pct / 2)
        print(f"  CN={cn}: {p_cn[cn]:5d} ({pct:5.1f}%) {bar}")

    # 4. Modifier-O distances
    print("\n--- Modifier-O Average Distances ---")
    for mod, cut in [("Na", 3.34), ("Ca", 3.14), ("Mg", 2.60)]:
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
            print(
                f"  {mod}-O: mean={np.mean(dists):.3f} A, "
                f"min={np.min(dists):.3f}, max={np.max(dists):.3f}, "
                f"avg_CN={len(dists) / len(mod_idx):.2f}"
            )

    print("=" * 60)


# ================================================================
# MAIN GENERATOR
# ================================================================
def generate_structure(mg_label, density_override, output_file, seed, scale=4):
    if mg_label not in PAPER_COMPOSITION:
        raise ValueError(
            f"Invalid --mg {mg_label}. Valid values: {sorted(PAPER_COMPOSITION.keys())}"
        )

    entry = PAPER_COMPOSITION[mg_label]

    # ===== COMPOSITION FROM PAPER TABLE 2 =====
    N_SI = 461 * scale
    N_P = 52 * scale
    N_NA = 488 * scale
    N_CA = entry["N_Ca"] * scale
    N_MG = entry["N_Mg"] * scale
    N_O = 1565 * scale

    N_TARGET = N_SI + N_P + N_NA + N_CA + N_MG + N_O

    counts = {
        "Si": N_SI,
        "P": N_P,
        "Na": N_NA,
        "Ca": N_CA,
        "Mg": N_MG,
        "O": N_O,
    }

    actual_total = sum(counts.values())

    print(f"\nComposition for paper label {entry['label']} (SCALE={scale}x):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  TOTAL: {actual_total}")

    # ===== DENSITY FROM PAPER TABLE 2 =====
    density = entry["density"] if density_override is None else float(density_override)

    masses = {
        "Si": 28.0855,
        "Ca": 40.078,
        "Na": 22.98977,
        "P": 30.97376,
        "O": 15.999,
        "Mg": 24.305,
    }

    total_mass = sum(masses[s] * c for s, c in counts.items())
    volume_A3 = (total_mass / NA) / density * 1.0e24
    box = volume_A3 ** (1.0 / 3.0)

    print(f"Box size: {box:.6f} A (Density: {density:.6f} g/cm3)")

    # ===== STAGE 1: Network Formers =====
    print(f"\n[Stage 1/4] Placing {N_SI + N_P} network formers (min NF-NF = 3.0 A)...")
    nf_coords, nf_placed = place_network_formers(
        N_SI,
        N_P,
        box,
        min_nf_dist=3.0,
        seed=seed
    )
    print(f"  Placed {nf_placed} NF ({N_SI} Si + {N_P} P)")

    # ===== STAGE 2: Oxygens =====
    print("[Stage 2/4] Placing O atoms with target O/(Si+P) ratio from paper...")
    o_coords, o_placed = place_oxygens_around_nf_target(
        nf_coords,
        nf_placed,
        N_O,
        box,
        o_si_dist=1.61,   # paper Si-O ~ 1.606 A
        o_p_dist=1.50,    # paper P-O ~ 1.49-1.50 A
        n_si=N_SI,
        o_min_dist=2.0,
        seed=seed
    )
    print(f"  Placed {o_placed} O (target: {N_O})")

    if o_placed > N_O:
        o_coords = o_coords[:N_O]
        o_placed = N_O
    elif o_placed < N_O:
        # Very unlikely, but keep a robust fallback
        np.random.seed(seed + 3000)
        remaining = N_O - o_placed
        extra_o = np.zeros((remaining, 3), dtype=np.float64)
        half_box = box / 2.0
        o_min_d_sq = 2.0 ** 2

        for i in range(remaining):
            px = np.random.uniform(0.0, box)
            py = np.random.uniform(0.0, box)
            pz = np.random.uniform(0.0, box)

            for attempt in range(100000):
                px = np.random.uniform(0.0, box)
                py = np.random.uniform(0.0, box)
                pz = np.random.uniform(0.0, box)

                valid = True
                for j in range(o_placed):
                    dx = px - o_coords[j, 0]
                    dy = py - o_coords[j, 1]
                    dz = pz - o_coords[j, 2]

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

                    if dx * dx + dy * dy + dz * dz < o_min_d_sq:
                        valid = False
                        break

                if valid:
                    break

            extra_o[i, 0] = px
            extra_o[i, 1] = py
            extra_o[i, 2] = pz

        o_coords = np.vstack((o_coords[:o_placed], extra_o))
        o_placed = N_O
        print(f"  Added {remaining} extra O randomly")

    # ===== STAGE 3: Modifiers =====
    print("[Stage 3/4] Placing modifiers (affinity-aware, Ca/Mg near P possible)...")
    all_nf_o = np.vstack((nf_coords[:nf_placed], o_coords[:N_O]))
    n_all = nf_placed + N_O

    N_MOD = N_NA + N_CA + N_MG

    p_coords = nf_coords[N_SI:N_SI + N_P]

    mod_coords, mod_placed = place_modifiers_affinity(
        N_MOD,
        box,
        all_nf_o,
        n_all,
        mod_o_min_dist=2.0,
        mod_mod_min_dist=2.0,
        p_coords=p_coords,
        n_p=N_P,
        p_attraction_radius=6.0,
        seed=seed
    )
    print(f"  Placed {mod_placed} modifiers ({N_NA} Na + {N_CA} Ca + {N_MG} Mg)")

    # ===== STAGE 4: Assembly =====
    print("[Stage 4/4] Assembling final structure...")

    symbols = []
    all_coords = []

    for i in range(N_SI):
        symbols.append("Si")
        all_coords.append(nf_coords[i])

    for i in range(N_P):
        symbols.append("P")
        all_coords.append(nf_coords[N_SI + i])

    for i in range(N_O):
        symbols.append("O")
        all_coords.append(o_coords[i])

    for i in range(N_NA):
        symbols.append("Na")
        all_coords.append(mod_coords[i])

    for i in range(N_CA):
        symbols.append("Ca")
        all_coords.append(mod_coords[N_NA + i])

    for i in range(N_MG):
        symbols.append("Mg")
        all_coords.append(mod_coords[N_NA + N_CA + i])

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
    with open(out_path, "w") as f:
        f.write(f"{total_atoms}\n")
        f.write(
            f"Mg-doped 45S5 paper, label={entry['label']}, "
            f"x={mg_label}, N={total_atoms}, "
            f"rho={density:.6f} g/cm3, box={box:.6f} A, seed={seed}\n"
        )
        for i in range(total_atoms):
            f.write(
                f"{symbols[i]:2s} "
                f"{all_coords[i, 0]:12.6f} "
                f"{all_coords[i, 1]:12.6f} "
                f"{all_coords[i, 2]:12.6f}\n"
            )

    print(f"\n[OK] Initial structure saved to: {out_path}")
    print(f"[OK] Box = {box:.6f} A, Density = {density:.6f} g/cm3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initial structure generator for Mg-doped 45S5 bioglass (paper parameters, 4x atoms)"
    )

    parser.add_argument(
        "--mg",
        type=int,
        default=0,
        choices=sorted(PAPER_COMPOSITION.keys()),
        help="Paper Mg label: 0, 1, 3, 5, 8, 10, 15, 20"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Optional density override. If omitted, paper density is used."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output XYZ file name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[1, 2, 4],
        help="Scale factor: 1=2835, 2=5670, 4=11340 atoms"
    )

    args = parser.parse_args()

    if args.out is None:
        args.out = f"initial_Mg{args.mg}.xyz"

    generate_structure(
        args.mg,
        args.density,
        args.out,
        args.seed,
        args.scale
    )
    