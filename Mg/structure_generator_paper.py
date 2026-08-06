"""
Modified structure_generator_paper.py for Mg-doped compositions based on the manuscript.
Generates structures with 4x the original size (11340 atoms).
Usage:
python structure_generator_paper.py --mg 0 --density 2.592 --seed 42 --scale 4
python structure_generator_paper.py --mg 1 --density 2.598 --seed 42 --scale 4
...
python structure_generator_paper.py --mg 20 --density 2.711 --seed 42 --scale 4
"""

import numpy as np
from numba import njit
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

NA = 6.02214076e23
# Original base number of atoms for the composition in the Xiang & Du paper
BASE_N_ATOMS_ORIGINAL = 2835 # Keep this for reference calculation
# New desired base number of atoms (4x original)
BASE_N_ATOMS = 11340 # 4 * 2835

# Type indices (Updated to include Mg)
# Si=0, Ca=1, Na=2, P=3, O=4, Mg=5 (or any other unused index)
TYPE_MAP = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Mg': 5}
ELEM_MAP = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Mg'}

# Masses
MASS = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
        'P': 30.97376, 'O': 15.999, 'Mg': 24.305} # Added Mg mass

# Base composition in mol% (Based on manuscript: 45SiO2 - 24.1CaO - 0.9MgO - 6Na2O - 24.1P2O5 -> Total ~ 100.1)
# Adjusting slightly for exact 100: 45SiO2 - 24.075CaO - 0.9MgO - 6Na2O - 24.025P2O5
# Or use: 45SiO2 - 24.1CaO - xMgO - 6Na2O - 24.1P2O5 (where x varies)
# Let's define it as: 45SiO2 - (24.1 - x)CaO - xMgO - 6Na2O - 24.1P2O5
# For x=0: 45SiO2 - 24.1CaO - 0MgO - 6Na2O - 24.1P2O5
# For x=0.9: 45SiO2 - 23.2CaO - 0.9MgO - 6Na2O - 24.1P2O5
# We need to calculate atom counts based on this.
# Moles per unit: SiO2 (1 Si + 2 O), Na2O (2 Na + 1 O), P2O5 (2 P + 5 O), CaO (1 Ca + 1 O), MgO (1 Mg + 1 O)
# Total moles of "formula units" needed to make up BASE_N_ATOMS
# Approximate calculation: Let f be the number of formula units scaling factor.
# Atoms from SiO2: 1 Si + 2 O => 3*f*45/100
# Atoms from Na2O: 2 Na + 1 O => 3*f*6/100
# Atoms from P2O5: 2 P + 5 O => 7*f*24.1/100
# Atoms from CaO: 1 Ca + 1 O => 2*f*((24.1 - x)/100)
# Atoms from MgO: 1 Mg + 1 O => 2*f*(x/100)
# Total Atoms = f * (3*45 + 3*6 + 7*24.1 + 2*(24.1 - x) + 2*x) / 100
# Total Atoms = f * (135 + 18 + 168.7 + 48.2 - 2x + 2x) / 100 = f * 369.9 / 100
# f = BASE_N_ATOMS * 100 / 369.9
# f = 11340 * 100 / 369.9 ~ 3065.96
# f = 2835 * 100 / 369.9 ~ 766.49 for original size

def calculate_atom_counts(mg_percent, base_n_atoms):
    """Calculates atom counts based on mol% composition."""
    x = mg_percent
    # Calculate scaling factor f based on base_n_atoms
    f = base_n_atoms * 100.0 / 369.9 # Using the derived formula above

    # Calculate approximate counts
    N_SI = round(f * 45.0 / 100.0)
    N_P = round(f * 24.1 / 100.0) * 2  # 2 P per P2O5
    N_NA = round(f * 6.0 / 100.0) * 2   # 2 Na per Na2O
    N_CA_MG_TOTAL = round(f * 24.1 / 100.0) # Total Ca+Mg from CaO+MgO part
    N_CA = round(N_CA_MG_TOTAL * (1 - x / 24.1)) if x <= 24.1 else 0 # Adjust based on x
    N_MG = round(N_CA_MG_TOTAL * (x / 24.1)) if x <= 24.1 else N_CA_MG_TOTAL # Adjust based on x
    # Ensure N_CA + N_MG = N_CA_MG_TOTAL
    if N_CA + N_MG != N_CA_MG_TOTAL:
        diff = N_CA_MG_TOTAL - (N_CA + N_MG)
        if diff > 0:
            N_MG += diff # Add difference to Mg if needed
        elif diff < 0:
             # Redistribute negative difference, prioritize keeping N_CA non-negative
             if N_CA > abs(diff):
                 N_CA += diff
             else:
                 N_MG += N_CA + diff # Transfer excess from Ca to Mg if Ca becomes negative
                 N_CA = 0


    # Calculate O atoms: 2 per SiO2 + 1 per Na2O + 5 per P2O5 + 1 per CaO + 1 per MgO
    N_O_SI = N_SI * 2
    N_O_NA = N_NA // 2  # Since 2 Na per Na2O -> 1 O per Na2O unit
    N_O_P = N_P // 2 * 5 # Since 2 P per P2O5 -> 5 O per P2O5 unit
    N_O_CA = N_CA # 1 O per CaO
    N_O_MG = N_MG # 1 O per MgO
    N_O = N_O_SI + N_O_NA + N_O_P + N_O_CA + N_O_MG

    counts = {'Si': N_SI, 'P': N_P, 'Na': N_NA, 'Ca': N_CA, 'Mg': N_MG, 'O': N_O}
    actual_total = sum(counts.values())
    print(f"Target base atoms: {base_n_atoms}")
    print(f"Calculated total atoms: {actual_total}")
    print(f"Composition for {mg_percent}% Mg doping:")
    for k, v in counts.items():
        print(f" {k}: {v}")
    return counts

@njit()
def place_atoms_nucleation_framework(coords, placed, n_si, n_p, box, type_indices, SI_TID, P_TID):
    """Place Si and P atoms in the nucleation framework."""
    max_attempts = 100000
    si_placed = 0
    p_placed = 0
    min_dist_sq = 1.8**2  # Minimum distance squared (e.g., 1.8 Angstrom)

    for i in range(n_si):
        attempts = 0
        while attempts < max_attempts:
            px, py, pz = np.random.random(3) * box
            valid = True
            # Check distance to already placed Si/P atoms
            for j in range(si_placed + p_placed):
                dx = coords[j, 0] - px
                dy = coords[j, 1] - py
                dz = coords[j, 2] - pz
                if dx > box/2.0: dx -= box
                elif dx < -box/2.0: dx += box
                if dy > box/2.0: dy -= box
                elif dy < -box/2.0: dy += box
                if dz > box/2.0: dz -= box
                elif dz < -box/2.0: dz += box
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < min_dist_sq:
                    valid = False
                    break
            if valid:
                coords[si_placed + p_placed] = [px, py, pz]
                type_indices[si_placed + p_placed] = SI_TID
                si_placed += 1
                placed[si_placed + p_placed - 1] = True
                break
            attempts += 1
        if attempts >= max_attempts:
            print(f"Warning: Could not place Si atom {i+1}, max attempts reached.")
            break

    for i in range(n_p):
        attempts = 0
        while attempts < max_attempts:
            px, py, pz = np.random.random(3) * box
            valid = True
            # Check distance to already placed Si/P atoms
            for j in range(si_placed + p_placed):
                dx = coords[j, 0] - px
                dy = coords[j, 1] - py
                dz = coords[j, 2] - pz
                if dx > box/2.0: dx -= box
                elif dx < -box/2.0: dx += box
                if dy > box/2.0: dy -= box
                elif dy < -box/2.0: dy += box
                if dz > box/2.0: dz -= box
                elif dz < -box/2.0: dy += box
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < min_dist_sq:
                    valid = False
                    break
            if valid:
                coords[si_placed + p_placed] = [px, py, pz]
                type_indices[si_placed + p_placed] = P_TID
                p_placed += 1
                placed[si_placed + p_placed - 1] = True
                break
            attempts += 1
        if attempts >= max_attempts:
            print(f"Warning: Could not place P atom {i+1}, max attempts reached.")
            break
    return si_placed, p_placed

@njit()
def place_oxygens_near_nucleation(coords, placed, n_o, si_placed, p_placed, box, type_indices, O_TID):
    """Place O atoms near Si and P atoms."""
    max_attempts_per_o = 1000
    o_placed = 0
    target_bond_length = 1.6  # Approximate Si-O/P-O bond length
    std_dev_bond = 0.1         # Standard deviation for bond length variation

    tree = cKDTree(coords[:si_placed+p_placed], boxsize=box)
    for i in range(n_o):
        attempts = 0
        placed_successfully = False
        while attempts < max_attempts_per_o and not placed_successfully:
            # Pick a random Si or P atom
            center_idx = np.random.randint(0, si_placed + p_placed)
            center_pos = coords[center_idx]
            # Generate random direction vector
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)
            # Generate random bond length (normal distribution around target)
            bond_len = np.random.normal(target_bond_length, std_dev_bond)
            bond_len = max(1.2, bond_len) # Ensure minimum distance
            new_pos = center_pos + direction * bond_len
            # Apply periodic boundary conditions
            new_pos = new_pos - box * np.floor(new_pos / box)

            # Check minimum distance from *all* currently placed atoms (including other O)
            valid = True
            for j in range(si_placed + p_placed + o_placed):
                 dx = coords[j, 0] - new_pos[0]
                 dy = coords[j, 1] - new_pos[1]
                 dz = coords[j, 2] - new_pos[2]
                 if dx > box/2.0: dx -= box
                 elif dx < -box/2.0: dx += box
                 if dy > box/2.0: dy -= box
                 elif dy < -box/2.0: dy += box
                 if dz > box/2.0: dz -= box
                 elif dz < -box/2.0: dz += box
                 dist_sq = dx*dx + dy*dy + dz*dz
                 if dist_sq < (1.2)**2: # Check against 1.2 Angstrom
                     valid = False
                     break

            if valid:
                coords[si_placed + p_placed + o_placed] = new_pos
                type_indices[si_placed + p_placed + o_placed] = O_TID
                o_placed += 1
                placed[si_placed + p_placed + o_placed - 1] = True
                placed_successfully = True
            attempts += 1

        if not placed_successfully:
             print(f"Warning: Could not place O atom {i+1}, max attempts per O reached.")
             # Fallback: Place randomly with basic overlap check
             fallback_attempts = 100
             for _ in range(fallback_attempts):
                 px, py, pz = np.random.random(3) * box
                 new_pos_fallback = np.array([px, py, pz])
                 valid_fallback = True
                 for j in range(si_placed + p_placed + o_placed):
                      dx = coords[j, 0] - new_pos_fallback[0]
                      dy = coords[j, 1] - new_pos_fallback[1]
                      dz = coords[j, 2] - new_pos_fallback[2]
                      if dx > box/2.0: dx -= box
                      elif dx < -box/2.0: dx += box
                      if dy > box/2.0: dy -= box
                      elif dy < -box/2.0: dy += box
                      if dz > box/2.0: dz -= box
                      elif dz < -box/2.0: dz += box
                      dist_sq = dx*dx + dy*dy + dz*dz
                      if dist_sq < (1.2)**2:
                          valid_fallback = False
                          break
                 if valid_fallback:
                     coords[si_placed + p_placed + o_placed] = new_pos_fallback
                     type_indices[si_placed + p_placed + o_placed] = O_TID
                     o_placed += 1
                     placed[si_placed + p_placed + o_placed - 1] = True
                     placed_successfully = True
                     print(f"  Placed O {i+1} via fallback method.")
                     break
             if not placed_successfully:
                 print(f"  ERROR: Could not place O atom {i+1} even with fallback.")


    return o_placed

@njit()
def place_modifiers_affinity_based(coords, placed, n_na, n_ca, n_mg, si_placed, p_placed, o_placed, box, type_indices, NA_TID, CA_TID, MG_TID):
    """Place modifier ions (Na, Ca, Mg) with affinity rules (e.g., Ca/Mg prefer P sites)."""
    n_mod_total = n_na + n_ca + n_mg
    mod_placed = 0
    mod_coords = np.empty((n_mod_total, 3), dtype=np.float64)
    mod_types = np.empty(n_mod_total, dtype=np.int32)

    # Create KDTree for P atoms (preferable sites for Ca/Mg)
    p_indices = np.where(type_indices[:si_placed+p_placed] == P_TID)[0]
    p_coords = coords[p_indices]

    # Affinity probabilities (adjustable)
    ca_mg_prefers_p_prob = 0.7 # Probability that Ca/Mg goes near P
    na_no_pref_prob = 0.5      # Probability that Na goes near Si/P (rest goes anywhere)

    # Place Ca and Mg first (based on affinity)
    n_ca_placed = 0
    n_mg_placed = 0
    for i in range(n_ca + n_mg):
        if i < n_ca:
            elem_tid = CA_TID
        else:
            elem_tid = MG_TID

        attempts = 0
        max_mod_attempts = 1000
        placed_successfully = False
        while attempts < max_mod_attempts and not placed_successfully:
            if np.random.random() < ca_mg_prefers_p_prob and len(p_coords) > 0:
                # Try placing near P
                p_idx = np.random.choice(len(p_coords))
                center_pos = p_coords[p_idx]
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                bond_len = np.random.normal(2.4, 0.2) # Approx Ca-O/Mg-O distance
                new_pos = center_pos + direction * bond_len
            else:
                # Place randomly
                new_pos = np.random.random(3) * box

            new_pos = new_pos - box * np.floor(new_pos / box)

            # Check minimum distance from *all* placed atoms
            valid = True
            for j in range(si_placed + p_placed + o_placed + mod_placed):
                 dx = coords[j, 0] - new_pos[0]
                 dy = coords[j, 1] - new_pos[1]
                 dz = coords[j, 2] - new_pos[2]
                 if dx > box/2.0: dx -= box
                 elif dx < -box/2.0: dx += box
                 if dy > box/2.0: dy -= box
                 elif dy < -box/2.0: dy += box
                 if dz > box/2.0: dz -= box
                 elif dz < -box/2.0: dz += box
                 dist_sq = dx*dx + dy*dy + dz*dz
                 if dist_sq < (2.0)**2: # Check against 2.0 Angstrom
                     valid = False
                     break

            if valid:
                mod_coords[mod_placed] = new_pos
                mod_types[mod_placed] = elem_tid
                mod_placed += 1
                placed_successfully = True
            attempts += 1
        if not placed_successfully:
             print(f"Warning: Could not place modifier (Ca/Mg #{i+1}), max attempts reached.")
             # Fallback: Place randomly
             px, py, pz = np.random.random(3) * box
             mod_coords[mod_placed] = [px, py, pz]
             mod_types[mod_placed] = elem_tid
             mod_placed += 1
             print(f"  Placed modifier {i+1} via fallback method.")
             placed_successfully = True # Mark as placed for loop logic

    # Now place Na (can go anywhere, maybe slightly prefer Si)
    for i in range(n_na):
        attempts = 0
        max_mod_attempts = 1000
        placed_successfully = False
        while attempts < max_mod_attempts and not placed_successfully:
            if np.random.random() < na_no_pref_prob:
                # Try placing near Si
                si_idx = np.random.choice(si_placed)
                center_pos = coords[si_idx]
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                bond_len = np.random.normal(2.5, 0.2) # Approx Na-O distance
                new_pos = center_pos + direction * bond_len
            else:
                # Place randomly
                new_pos = np.random.random(3) * box

            new_pos = new_pos - box * np.floor(new_pos / box)

            # Check minimum distance from *all* placed atoms
            valid = True
            for j in range(si_placed + p_placed + o_placed + mod_placed):
                 dx = coords[j, 0] - new_pos[0]
                 dy = coords[j, 1] - new_pos[1]
                 dz = coords[j, 2] - new_pos[2]
                 if dx > box/2.0: dx -= box
                 elif dx < -box/2.0: dx += box
                 if dy > box/2.0: dy -= box
                 elif dy < -box/2.0: dy += box
                 if dz > box/2.0: dz -= box
                 elif dz < -box/2.0: dz += box
                 dist_sq = dx*dx + dy*dy + dz*dz
                 if dist_sq < (2.0)**2:
                     valid = False
                     break

            if valid:
                mod_coords[mod_placed] = new_pos
                mod_types[mod_placed] = NA_TID
                mod_placed += 1
                placed_successfully = True
            attempts += 1
        if not placed_successfully:
             print(f"Warning: Could not place Na atom {i+1}, max attempts reached.")
             # Fallback: Place randomly
             px, py, pz = np.random.random(3) * box
             mod_coords[mod_placed] = [px, py, pz]
             mod_types[mod_placed] = NA_TID
             mod_placed += 1
             print(f"  Placed Na {i+1} via fallback method.")
             placed_successfully = True # Mark as placed for loop logic


    # Copy mod_coords and mod_types back to main arrays
    coords[si_placed + p_placed + o_placed : si_placed + p_placed + o_placed + mod_placed] = mod_coords[:mod_placed]
    type_indices[si_placed + p_placed + o_placed : si_placed + p_placed + o_placed + mod_placed] = mod_types[:mod_placed]

    return mod_placed


def generate_structure(mg_percent, density, output_file, seed, scale_factor=1):
    counts = calculate_atom_counts(mg_percent, BASE_N_ATOMS * scale_factor) # Use scaled base
    N_SI = counts['Si']
    N_P = counts['P']
    N_NA = counts['Na']
    N_CA = counts['Ca']
    N_MG = counts['Mg']
    N_O = counts['O']

    # Calculate box size based on density and total mass
    total_mass_scaled = (
        N_SI * MASS['Si'] + N_P * MASS['P'] + N_NA * MASS['Na'] +
        N_CA * MASS['Ca'] + N_MG * MASS['Mg'] + N_O * MASS['O']
    )
    # Density in g/cm3, Mass in amu, convert to Angstroms
    box_angstroms_scaled = ((total_mass_scaled / NA) / density) * 1e24
    box_scaled = box_angstroms_scaled**(1/3.)
    print(f" Calculated box size: {box_scaled:.4f} Angstrom")

    total_atoms_scaled = N_SI + N_P + N_NA + N_CA + N_MG + N_O
    print(f" Scaled Total atoms: {total_atoms_scaled}")

    # Initialize arrays
    coords = np.empty((total_atoms_scaled, 3), dtype=np.float64)
    type_indices = np.empty(total_atoms_scaled, dtype=np.int32)
    placed = np.zeros(total_atoms_scaled, dtype=np.bool_)

    # Define type IDs
    SI_TID = TYPE_MAP['Si']
    CA_TID = TYPE_MAP['Ca']
    NA_TID = TYPE_MAP['Na']
    P_TID = TYPE_MAP['P']
    O_TID = TYPE_MAP['O']
    MG_TID = TYPE_MAP['Mg']

    print("[Stage 1/4] Placing nucleation framework (Si, P)...")
    si_placed, p_placed = place_atoms_nucleation_framework(coords, placed, N_SI, N_P, box_scaled, type_indices, SI_TID, P_TID)
    print(f"  Placed {si_placed} Si and {p_placed} P atoms.")

    print("[Stage 2/4] Placing oxygens near Si/P (network O)...")
    o_placed = place_oxygens_near_nucleation(coords, placed, N_O, si_placed, p_placed, box_scaled, type_indices, O_TID)
    print(f"  Placed {o_placed} O atoms.")

    print("[Stage 3/4] Placing modifiers (Affinity-based: Ca/Mg prefer P)...")
    mod_placed = place_modifiers_affinity_based(coords, placed, N_NA, N_CA, N_MG, si_placed, p_placed, o_placed, box_scaled, type_indices, NA_TID, CA_TID, MG_TID)
    print(f"  Placed {mod_placed} modifier atoms (Na, Ca, Mg).")

    # Verify placement
    placed_count = np.sum(placed)
    if placed_count != total_atoms_scaled:
        print(f"ERROR: Only placed {placed_count} atoms out of {total_atoms_scaled} expected!")
        return
    else:
        print(f" Successfully placed all {placed_count} atoms.")

    # Convert type indices back to symbols for final array
    all_coords = coords
    all_symbols = [ELEM_MAP[tid] for tid in type_indices]

    # ===== SHUFFLE (Optional but recommended for initial randomness) =====
    np.random.seed(seed + 5000)
    perm = np.random.permutation(total_atoms_scaled)
    all_symbols = [all_symbols[i] for i in perm]
    all_coords = all_coords[perm]

    # ===== WRITE XYZ =====
    out_path = Path(output_file)
    with open(out_path, 'w') as f:
        f.write(f"{total_atoms_scaled}\n")
        f.write(f"Mg-doped Glass based on Manuscript, x={mg_percent} mol% MgO, "
                f"N={total_atoms_scaled}, rho={density:.4f} g/cm3, "
                f"box={box_scaled:.4f} A, seed={seed}\n")
        for i in range(total_atoms_scaled):
            f.write(f"{all_symbols[i]:2s} "
                    f"{all_coords[i,0]:10.6f} "
                    f"{all_coords[i,1]:10.6f} "
                    f"{all_coords[i,2]:10.6f}\n")

    print(f"[OK] Initial structure saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mg-doped glass structure based on manuscript parameters.")
    parser.add_argument('--mg', type=float, default=0.0, help='Mg mol percent (default 0.0)')
    parser.add_argument('--density', type=float, required=True, help='Target density in g/cm3')
    parser.add_argument('--out', type=str, help='Output XYZ filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default 42)')
    parser.add_argument('--scale', type=int, default=4, choices=[1, 2, 4], help='Scale factor: 1=2835, 2=5670, 4=11340 atoms (default 4)')

    args = parser.parse_args()

    if args.out is None:
        args.out = f"initial_Mg{args.mg:.1f}.xyz"

    generate_structure(args.mg, args.density, args.out, args.seed, args.scale)
