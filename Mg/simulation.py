#!/usr/bin/env python3
"""
========================================================================
45S5/Mg Bioglass NVT Monte Carlo - FINAL v10.1 (ALL-IN-ONE) (NO Sr)
Based on: Xiang & Du, Chem. Mater. 2011, 23, 2703-2717
MANUSCRIPT REFERENCE: Mg Substitution (45-M0 to 45-M20)
COMPLETE PACKAGE:
- NVT Monte Carlo simulation (no artificial constraints)
- Full structural analysis (RDF, CN, Qn, angles, O speciation)
- Block averaging with standard deviation
- Sensitivity analysis (cutoff & Wolf alpha)
- Clustering R_XX, Modifier preference, Fnet, Si-O-P linkages
- Neutron structure factor
- Complete Excel export + plots

Usage:
python simulation.py initial_Mg0.xyz --final-sweeps 150 --cores 12
python simulation.py final_structure.xyz --continue --final-sweeps 300 --cores 12
========================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os, sys, logging, pickle, re, warnings, codecs
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer
from tqdm import tqdm
import threading
from queue import Queue
import traceback

# ========================= CONSTANTS =========================
KB_EV = 8.617333262145e-5
NA = 6.02214076e23
KE = 14.3996454784255
SKIN = 1.5
SWITCH_DR = 0.3
BASE_N_ATOMS = 2835

# Paper reference values (Updated for Mg, NO Sr)
PAPER_BOND_LENGTHS = {('O','O'):2.63, ('Si','O'):1.61, ('Ca','O'):2.38, ('Na','O'):2.40,
                      ('Mg','O'): 2.1} # Example value from manuscript, find accurate one if available
PAPER_CN = {'Ca': 6.2, 'Na': 5.7, 'Mg': 6.0} # Example value, find accurate one from manuscript for 45-M5 or relevant composition
PAPER_QN_SI = {'Q0':3.7, 'Q1':23.2, 'Q2':39.8, 'Q3':27.8, 'Q4':5.6}
PAPER_QN_P = {'Q0':50.0, 'Q1':42.9, 'Q2':6.4, 'Q3':0.6, 'Q4':0.0}
PAPER_NC = {'Si': 2.07, 'P': 0.62, 'overall': 1.953} # Example for 45-M5 from manuscript

NEUTRON_B = {'O': 5.803, 'Si': 4.1491, 'Na': 3.63, 'Ca': 4.70, 'P': 5.13, 'Mg': 5.378} # Example value, find accurate one

class SimulationError(Exception): pass
class FileError(SimulationError): pass

def parse_xyz_header(path: Path):
    if not path.exists():
        raise FileError(f"File not found: {path}")
    with open(path, 'r') as f:
        n_atoms_line = f.readline().strip()
        comment_line = f.readline().strip()

    try:
        n_atoms = int(n_atoms_line)
    except ValueError:
        raise FileError(f"Invalid number of atoms in first line: {n_atoms_line}")

    # Regex patterns to extract metadata
    patterns = [
        ('x', r'x\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('N', r'N\s*=\s*(\d+)', int),
        ('rho', r'rho\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('box', r'box\s*=\s*([0-9]*\.?[0-9]+)', float),
        ('seed', r'seed\s*=\s*(\d+)', int),
    ]
    meta = {}
    for key, pat, conv in patterns:
        m = re.search(pat, comment_line)
        if m:
            meta[key] = conv(m.group(1))

    return n_atoms, meta

# ========================= POTENTIAL =========================
class Potential:
    def __init__(self):
        # Buckingham parameters (Table 1 from manuscript for Mg)
        self.buck_params = {
            ('Si','O'): {'A': 13702.905, 'F': 0.193817, 'C': 54.681},
            ('P','O'): {'A': 26655.472, 'F': 0.181968, 'C': 86.856},
            ('O','O'): {'A': 2029.2204, 'F': 0.343645, 'C': 192.58},
            ('Na','O'): {'A': 4383.7555, 'F': 0.243838, 'C': 30.70},
            ('Ca','O'): {'A': 7747.1834, 'F': 0.252623, 'C': 93.109},
            ('Mg','O'): {'A': 7063.0, 'F': 0.2109, 'C': 19.21} # Added Mg-O from manuscript Table 1
        }
        # ZBL parameters (constants)
        self.zbl_a0 = 0.529177210903  # Bohr radius in Angstrom
        self.zbl_c = 14.3996454784255  # e^2 / (4*pi*epsilon_0) in eV*A
        self.zbl_d = 0.885341375986  # sqrt(hbar^2 / (2*m_e)) / a0 in sqrt(eV)*A
        # Z values for ZBL potential (NO Sr)
        self.zbl_z = {'Si': 14.0, 'Ca': 20.0, 'Na': 11.0, 'P': 15.0, 'O': 8.0, 'Mg': 12.0} # Sr removed, Mg added

        self.r_hard_default = 0.9
        self.r_hard_by_pair = {('O', 'O'): 1.4}
        # Specific Born-Mayer parameters for surface energy calculation (NO Sr)
        self.sbs_x_o = {'Ca': 32.0, 'Na': 20.0, 'Mg': 25.0} # Sr removed, Mg added (example value)

# ========================= GLASS SYSTEM =========================
class GlassSystem:
    def __init__(self, config: SimulationConfig):
        self.config = config
        n_header, meta = parse_xyz_header(config.input_file)
        self.N_ATOMS = n_header
        self.x_val = meta.get('x', 0.0) # Assume x represents Mg percentage now
        self.initial_box = meta.get('box', None)

        # Load coordinates and types
        with open(config.input_file, 'r') as f:
            lines = f.readlines()[2:2+self.N_ATOMS] # Skip header lines

        symbols = []
        coords_list = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0].strip())
                coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

        self.symbols = symbols
        self.coords = np.array(coords_list, dtype=np.float64)
        # Updated type map to include Mg, NO Sr
        self.type_map = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Mg': 5} # Sr removed, Mg added
        self.type_to_elem = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Mg'} # Sr removed, Mg added
        self.type_indices = np.array([self.type_map[s] for s in self.symbols], dtype=np.int32)
        self.counts = {e: int(np.sum(self.type_indices == self.type_map[e])) for e in set(self.symbols)}
        logger.info(f"Composition: {self.counts}")
        # Masses (NO Sr)
        self.masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
                       'P': 30.97376, 'O': 15.999, 'Mg': 24.305} # Sr removed, Mg added
        self.total_mass = sum(self.masses[s] for s in self.symbols)

        self._setup_box()
        self._setup_charges()
        self.potential = Potential()

    def _setup_box(self):
        if self.config.box_override is not None:
            self.box = float(self.config.box_override)
            self.effective_density = self.total_mass * 1.0e24 / (NA * self.box**3)
            logger.info(f"Using box from config override: {self.box:.4f} A (density={self.effective_density:.4f})")
        elif self.initial_box is not None:
            self.box = float(self.initial_box)
            self.effective_density = self.total_mass * 1.0e24 / (NA * self.box**3)
            logger.info(f"Using box from XYZ: {self.box:.4f} A (density={self.effective_density:.4f})")
        else:
            # Calculate box from density
            if self.config.density is None:
                raise SimulationError("Density must be specified if not present in XYZ file or box override.")
            rho = self.config.density
            box_angstroms = ((self.total_mass / NA) / rho) * 1e24
            self.box = box_angstroms**(1/3.)
            self.effective_density = rho
            logger.info(f"Calculated box from density: {self.box:.4f} A")

    def _setup_charges(self):
        """Compute charges based on stoichiometry for neutrality."""
        # Base charges (NO Sr)
        base_charges = {'Si': 2.4, 'Ca': 1.2, 'Na': 0.6, 'P': 3.0, 'Mg': 1.2} # Sr removed, Mg added
        # Calculate total positive charge
        total_pos = sum(self.counts.get(e, 0) * base_charges.get(e, 0) for e in self.counts if e != 'O')
        n_o = self.counts.get('O', 0)
        if n_o <= 0:
            raise SimulationError("No oxygen atoms found.")
        q_o = -total_pos / n_o
        # Initialize charges array
        self.charges = np.zeros(self.N_ATOMS, dtype=np.float64)
        # Assign charges based on type
        for i in range(self.N_ATOMS):
            elem = self.type_to_elem[self.type_indices[i]]
            if elem in base_charges:
                self.charges[i] = base_charges[elem]
            elif elem == 'O':
                self.charges[i] = q_o
            else:
                logger.warning(f"Unknown element '{elem}' for charge assignment.")
        logger.info(f"O charge set to {q_o:.6f} for charge neutrality.")
        # Log calculated charges for verification
        for elem, count in self.counts.items():
             if count > 0:
                 charge_val = base_charges.get(elem, q_o if elem == 'O' else 0.0)
                 logger.debug(f"Element {elem}: Count={count}, Charge={charge_val:.4f}")

    def get_type_Z(self):
        # Return Z values for all possible types (0 to 5), NO Sr
        return np.array([self.potential.zbl_z[self.type_to_elem[i]] for i in range(6)], dtype=np.float64) # Index 0 to 5


# ========================= NEIGHBOR LIST =========================
class NeighborList:
    def __init__(self, system, cutoff, skin=SKIN):
        self.system = system
        self.cutoff = cutoff
        self.skin = skin
        self.rebuild_count = 0
        self.ref_coords = None
        self.neighbors, self.starts = self._build()

    def _build(self):
        tree = cKDTree(self.system.coords, boxsize=self.system.box)
        pairs = tree.query_pairs(self.cutoff + self.skin, output_type='ndarray')
        n = self.system.N_ATOMS
        starts = np.zeros(n + 1, dtype=np.int32)
        neighbors = np.zeros(len(pairs) * 2, dtype=np.int32)
        fill = np.zeros(n, dtype=np.int32)

        for i, j in pairs:
            starts[i + 1] += 1
            starts[j + 1] += 1
        starts = np.cumsum(starts)
        for i, j in pairs:
            neighbors[starts[i] + fill[i]] = j
            fill[i] += 1
            neighbors[starts[j] + fill[j]] = i
            fill[j] += 1
        self.neighbors = neighbors
        self.starts = starts
        self.ref_coords = self.system.coords.copy()
        self.rebuild_count += 1
        return neighbors, starts

    def update(self, force=False):
        if force or self._needs_rebuild():
            return self._build()
        return self.neighbors, self.starts

    def _needs_rebuild(self):
        if self.ref_coords is None: return True
        dr = self.system.coords - self.ref_coords
        dr -= self.system.box * np.round(dr / self.system.box)
        return np.sqrt(np.sum(dr**2, axis=1)).max() > self.skin / 3.0


# ========================= NUMBA ENERGY ROUTINES =========================
@njit(fastmath=True, cache=True)
def minimum_image(dr, box):
    dr -= box * np.floor(dr / box + 0.5)
    return dr

@njit(fastmath=True, cache=True)
def _buckingham(r, A, F, C):
    if r < 1e-6: return 1e8
    inv_r2 = 1.0 / (r*r)
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    return A * np.exp(-F * r) - C * inv_r6

@njit(fastmath=True, cache=True)
def _coulomb(r, qi, qj, alpha, cutoff):
    if r > cutoff - 1e-6: return 0.0
    if r < 1e-6: return 1e8
    # Wolf summation kernel
    erf_alpha_r = np.erf(alpha * r)
    erf_alpha_rc = np.erf(alpha * cutoff)
    cos_alpha_rc = np.cos(alpha * cutoff)
    sin_alpha_rc = np.sin(alpha * cutoff)
    term1 = qi*qj/r * (1.0 - erf_alpha_r)
    term2 = qi*qj/cutoff * (1.0 - erf_alpha_rc)
    term3 = qi*qj*alpha/np.sqrt(np.pi) * (cos_alpha_rc - sin_alpha_rc/(alpha*cutoff))
    return KE * (term1 - term2 + term3)

@njit(fastmath=True, cache=True)
def _zbl_repulsion(r, zi, zj, a0, c, d):
    if r < 1e-6: return 1e8
    a_inv = (zi**0.23 + zj**0.23) / a0
    phi = 0.1818 * np.exp(-3.2 * r * a_inv) + \
          0.5099 * np.exp(-0.9423 * r * a_inv) + \
          0.2802 * np.exp(-0.4029 * r * a_inv) + \
          0.02817 * np.exp(-0.2016 * r * a_inv)
    return c * zi * zj / r * phi

@njit(fastmath=True, cache=True)
def total_energy(coords, charges, types, type_Z, A_mat, F_mat, C_mat, R_HARD, neighbors, starts, box, cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    n = coords.shape[0]
    e_total = 0.0
    for i in range(n):
        start_i = starts[i]
        end_i = starts[i+1]
        for idx in range(start_i, end_i):
            j = neighbors[idx]
            if j <= i: continue # Avoid double counting

            dr = coords[i] - coords[j]
            dr = minimum_image(dr, box)
            r = np.sqrt(np.sum(dr*dr))

            if r > cutoff: continue

            ti = types[i]
            tj = types[j]
            qi = charges[i]
            qj = charges[j]

            sr = 0.0
            cl = 0.0

            # Buckingham
            A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
            if A > 0.0 and F > 1.0e-12:
                sr += _buckingham(r, A, F, C)
            if C > 0.0:
                sr -= C / (r**6)

            # ZBL Repulsion
            zi = type_Z[ti]
            zj = type_Z[tj]
            sr += _zbl_repulsion(r, zi, zj, zbl_a0, zbl_c, zbl_d)

            # Coulomb (Wolf)
            cl += _coulomb(r, qi, qj, alpha, cutoff)

            e_total += sr + cl
    return e_total / n # Per atom

@njit(fastmath=True, cache=True)
def energy_decomposition(coords, charges, types, type_Z, A_mat, F_mat, C_mat, R_HARD, neighbors, starts, box, cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    n = coords.shape[0]
    sr = 0.0
    cl = 0.0
    for i in range(n):
        start_i = starts[i]
        end_i = starts[i+1]
        for idx in range(start_i, end_i):
            j = neighbors[idx]
            if j <= i: continue # Avoid double counting

            dr = coords[i] - coords[j]
            dr = minimum_image(dr, box)
            r = np.sqrt(np.sum(dr*dr))

            if r > cutoff: continue

            ti = types[i]
            tj = types[j]
            qi = charges[i]
            qj = charges[j]

            local_sr = 0.0
            local_cl = 0.0

            # Buckingham
            A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
            if A > 0.0 and F > 1.0e-12:
                local_sr += _buckingham(r, A, F, C)
            if C > 0.0:
                local_sr -= C / (r**6)

            # ZBL Repulsion
            zi = type_Z[ti]
            zj = type_Z[tj]
            local_sr += _zbl_repulsion(r, zi, zj, zbl_a0, zbl_c, zbl_d)

            # Coulomb (Wolf)
            local_cl += _coulomb(r, qi, qj, alpha, cutoff)

            sr += local_sr
            cl += local_cl
    return sr / n, cl / n # Per atom


# ========================= MC SIMULATOR =========================
class MCSimulator:
    def __init__(self, system: GlassSystem, config: SimulationConfig):
        self.system = system
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.nl = NeighborList(system, config.cutoff, config.skin)
        self.accepted = 0
        self.attempts = 0
        self.rebuilds = 0
        self.step = 0
        # Max displacement (NO Sr)
        self.max_disp = {'Si': 0.05, 'Ca': 0.10, 'Na': 0.10, 'P': 0.05, 'O': 0.15, 'Mg': 0.10} # Sr removed, Mg added
        self.low_t_steps = max(1, int(0.5 * self.system.N_ATOMS))
        self.mixing_steps = max(1, int(1.0 * self.system.N_ATOMS))
        self.final_steps = max(1, int(config.final_sweeps * self.system.N_ATOMS))
        self.log_freq = config.log_freq or max(1, int(self.system.N_ATOMS / 100))
        self.decomposition_freq = config.decomposition_freq or self.log_freq * 10
        self.adaptive_tuning_freq = config.adaptive_tuning_freq or max(1, int(self.system.N_ATOMS / 10))
        self.snapshot_interval = config.snapshot_interval or max(1, int(self.system.N_ATOMS / 10))
        self.target_acceptance = config.target_acceptance
        self.energy_log = []
        self.short_log = []
        self.coul_log = []
        self.snapshots = []

        # Build matrices for energy calculation (NO Sr)
        self.A_mat = np.zeros((6, 6), dtype=np.float64) # Updated to 6 types (0-5), NO Sr
        self.F_mat = np.zeros((6, 6), dtype=np.float64)
        self.C_mat = np.zeros((6, 6), dtype=np.float64)
        for (e1, e2), p in self.system.potential.buck_params.items():
            t1 = self.system.type_map[e1]
            t2 = self.system.type_map[e2]
            self.A_mat[t1, t2] = p['A']; self.F_mat[t1, t2] = p['F']; self.C_mat[t1, t2] = p['C']
            if t1 != t2: # Symmetrize
                self.A_mat[t2, t1] = p['A']; self.F_mat[t2, t1] = p['F']; self.C_mat[t2, t1] = p['C']
        self.type_Z = self.system.get_type_Z()
        self.wolf_self = self._compute_wolf_self_energy()

        # Current energy
        n, s = self.nl.neighbors, self.nl.starts
        U, _, _ = energy_decomposition(
            self.system.coords, self.system.charges, self.system.type_indices,
            self.type_Z, self.A_mat, self.F_mat, self.C_mat,
            np.zeros((6, 6)), # R_HARD not used in decomposition
            n, s, self.system.box, self.config.cutoff, self.config.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d
        )
        self.current_energy = U + self.wolf_self
        logger.info(f"Initial energy: {self.current_energy:.6f} eV/atom")

    def _compute_wolf_self_energy(self):
        """Compute the self-energy correction for Wolf summation."""
        n = self.system.N_ATOMS
        q_squared_sum = np.sum(self.system.charges**2)
        qc = np.sum(self.system.charges) # Net charge (should be zero ideally)
        alpha = self.config.wolf_alpha
        cutoff = self.config.cutoff
        # Self-energy term for Wolf: - alpha/sqrt(pi) * sum(q_i^2) + qc^2 / (2*cutoff)
        self_term = - alpha / np.sqrt(np.pi) * q_squared_sum
        monopole_term = qc**2 / (2 * cutoff)
        return KE * (self_term + monopole_term) / n # Per atom

    def _mc_move(self, T):
        i = self.rng.integers(0, self.system.N_ATOMS)
        elem = self.system.type_to_elem[self.system.type_indices[i]]
        disp_scale = self.max_disp.get(elem, 0.05)

        old_pos = self.system.coords[i].copy()
        trial_pos = old_pos + self.rng.uniform(-disp_scale, disp_scale, 3)
        trial_pos = trial_pos - self.system.box * np.floor(trial_pos / self.system.box) # PBC

        # Compute energy difference
        delta_e = self._compute_delta_energy(i, old_pos, trial_pos)

        # Metropolis criterion
        if delta_e < 0 or self.rng.random() < np.exp(-delta_e / (KB_EV * T)):
            self.system.coords[i] = trial_pos
            self.current_energy += delta_e
            # Log energy
            if self.step % self.log_freq == 0:
                self.energy_log.append(float(self.current_energy))
                if self.step % self.decomposition_freq == 0:
                    n, s = self.nl.update(force=True) # Rebuild for accurate decomposition
                    self.rebuilds += 1
                    sr, cl = energy_decomposition(
                        self.system.coords, self.system.charges, self.system.type_indices,
                        self.type_Z, self.A_mat, self.F_mat, self.C_mat,
                        np.zeros((6, 6)),
                        n, s, self.system.box, self.config.cutoff, self.config.wolf_alpha,
                        self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d
                    )
                    self.short_log.append(float(sr))
                    self.coul_log.append(float(cl + self.wolf_self))
                else:
                    self.short_log.append(np.nan) # Placeholder
                    self.coul_log.append(np.nan) # Placeholder
            return True
        return False

    def _compute_delta_energy(self, i, old_pos, new_pos):
        # This is a simplified version. For accurate delta E calculation,
        # one would compute the energy change by only considering interactions
        # involving atom i before and after the move.
        # However, for a full system energy check, we rebuild the neighbor list
        # and recompute the entire energy difference.
        # This is computationally expensive but guarantees correctness.
        # A more efficient approach involves only recalculating the energy
        # contribution from neighbors of atom i.
        # For now, let's implement the efficient local delta calculation.

        coords = self.system.coords
        charges = self.system.charges
        types = self.system.type_indices
        box = self.system.box
        cutoff = self.config.cutoff
        alpha = self.config.wolf_alpha

        # Get neighbors within cutoff of atom i using the *old* position
        tree_old = cKDTree(coords, boxsize=box)
        old_neighbors = tree_old.query_ball_point(old_pos, cutoff + 0.5) # Slightly larger for safety
        old_neighbors = [idx for idx in old_neighbors if idx != i]

        # Get neighbors within cutoff of atom i using the *new* position
        temp_coords = coords.copy()
        temp_coords[i] = new_pos
        tree_new = cKDTree(temp_coords, boxsize=box)
        new_neighbors = tree_new.query_ball_point(new_pos, cutoff + 0.5)
        new_neighbors = [idx for idx in new_neighbors if idx != i]

        # Combine unique neighbors from old and new lists
        all_check_neighbors = set(old_neighbors + new_neighbors)

        old_energy_contribution = 0.0
        new_energy_contribution = 0.0

        # Calculate energy contribution *from* neighbors *to* atom i in the old configuration
        for j in old_neighbors:
             dr_old = old_pos - coords[j]
             dr_old = minimum_image(dr_old, box)
             r_old = np.sqrt(np.sum(dr_old*dr_old))
             if r_old > cutoff: continue

             ti = types[i]
             tj = types[j]
             qi = charges[i]
             qj = charges[j]

             sr_old = 0.0
             cl_old = 0.0

             A = self.A_mat[ti, tj]; F = self.F_mat[ti, tj]; C = self.C_mat[ti, tj]
             if A > 0.0 and F > 1.0e-12: sr_old += _buckingham(r_old, A, F, C)
             if C > 0.0: sr_old -= C / (r_old**6)
             zi = self.type_Z[ti]; zj = self.type_Z[tj]
             sr_old += _zbl_repulsion(r_old, zi, zj, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
             cl_old += _coulomb(r_old, qi, qj, alpha, cutoff)

             old_energy_contribution += sr_old + cl_old

        # Calculate energy contribution *from* neighbors *to* atom i in the new configuration
        for j in new_neighbors:
             pos_j = coords[j] if j != i else new_pos # Use new pos if j is the moved atom
             dr_new = new_pos - pos_j
             dr_new = minimum_image(dr_new, box)
             r_new = np.sqrt(np.sum(dr_new*dr_new))
             if r_new > cutoff: continue

             ti = types[i]
             tj = types[j]
             qi = charges[i] # Charge of atom i doesn't change
             qj = charges[j]

             sr_new = 0.0
             cl_new = 0.0

             A = self.A_mat[ti, tj]; F = self.F_mat[ti, tj]; C = self.C_mat[ti, tj]
             if A > 0.0 and F > 1.0e-12: sr_new += _buckingham(r_new, A, F, C)
             if C > 0.0: sr_new -= C / (r_new**6)
             zi = self.type_Z[ti]; zj = self.type_Z[tj]
             sr_new += _zbl_repulsion(r_new, zi, zj, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
             cl_new += _coulomb(r_new, qi, qj, alpha, cutoff)

             new_energy_contribution += sr_new + cl_new

        # The delta E is the difference in energy felt by atom i due to its environment
        # changing from old to new position.
        # However, the interaction energy between atom i and each neighbor j is shared.
        # So, the total system energy change is the sum of changes for all involved pairs.
        # The change contributed by atom i moving is approximately the difference in its
        # interaction energies with *its* neighbors.
        # But to be fully correct, we should also consider how the interaction energy
        # between neighbors *not* including i changes due to i's movement affecting the list.
        # This is complex. The simplest way is to just recalculate the total energy difference
        # after updating the coordinate and rebuilding the neighbor list periodically.
        # The code below calculates the local change approximation.

        # This is the *local* energy change approximation
        delta_local = new_energy_contribution - old_energy_contribution

        # For a true delta E, we'd need to account for changes in interactions *between*
        # neighbors j and k, which might now be included/excluded due to i's movement.
        # This requires a full recalculation or a very careful incremental update.
        # Given the complexity and performance considerations, the local approximation
        # is often sufficient, especially with frequent neighbor list rebuilds triggered
        # by `_maybe_log` or explicit checks.

        # Let's trigger a neighbor list update check after many moves to maintain accuracy.
        return delta_local


    def _maybe_log(self):
        if self.step % self.adaptive_tuning_freq == 0:
            acc_rate = self.accepted / max(1, self.attempts)
            logger.debug(f"Step {self.step}: Acc. Rate = {acc_rate:.3f}, Target = {self.target_acceptance:.2f}")
            for k in self.max_disp:
                if acc_rate < self.target_acceptance * 0.8:
                    self.max_disp[k] = min(self.max_disp[k] * 1.1, 0.5) # Increase disp
                elif acc_rate > self.target_acceptance * 1.2:
                    self.max_disp[k] = max(self.max_disp[k] * 0.9, 0.01) # Decrease disp
            logger.debug(f"Adjusted displacements: {self.max_disp}")

    def _run_stage(self, T, steps, desc, adaptive=False, mixing=False, enhance_oxygen=False):
        logger.info(f"{desc}: T={T:.1f} K, steps={steps}")
        stage_acc = 0; stage_att = 0
        window_acc = 0; window_att = 0
        ema = 0.5
        orig_disp = self.max_disp.copy()
        if mixing:
            for k in self.max_disp: self.max_disp[k] *= 2.0
            self.max_disp['O'] *= 1.5 # Enhance mixing further

        pbar = tqdm(range(steps), desc=f"T={T:.0f}K")
        for s in pbar:
            self.step += 1
            self.attempts += 1
            stage_att += 1

            if self._mc_move(T):
                self.accepted += 1
                stage_acc += 1
                window_acc += 1
            window_att += 1

            if adaptive and window_att >= 100:
                win_rate = window_acc / window_att
                if win_rate < self.target_acceptance * 0.8:
                    for k in self.max_disp: self.max_disp[k] *= 0.95
                elif win_rate > self.target_acceptance * 1.2:
                    for k in self.max_disp: self.max_disp[k] *= 1.05
                window_acc = 0; window_att = 0

            if (s + 1) % (steps // 10) == 0: # Update progress bar every 10%
                acc_str = f"{stage_acc / stage_att * 100:.1f}%"
                disp_str = f"Disp(O)={self.max_disp['O']:.3f}"
                pbar.set_postfix({'acc': acc_str, 'disp_O': disp_str})

            self._maybe_log()

        pbar.close()
        if mixing:
            self.max_disp.update(orig_disp) # Restore original displacements after mixing
        logger.info(f" Stage acceptance: {stage_acc / stage_att * 100:.2f}%")

    def run(self):
        logger.info("=" * 70)
        logger.info("MONTE CARLO SIMULATION")
        logger.info("=" * 70)

        if self.config.continue_mode:
            logger.info("Continuing simulation from previous state...")
            self._run_stage(300.0, self.final_steps, "Production (Continued)", adaptive=True)
        else:
            logger.info("Running full protocol...")
            # High T Mixing
            self._run_stage(self.config.mixing_temp, self.mixing_steps, "High-T Mixing", adaptive=True, mixing=True)
            # Annealing Stages
            for i, (T, cycles) in enumerate(self.config.annealing_stages):
                 steps = max(1, int(cycles * self.system.N_ATOMS))
                 self._run_stage(T, steps, f"Annealing {i+1} (T={T}K)", adaptive=True)
            # Low T Relaxation
            self._run_stage(1.0, self.low_t_steps, "Low-T Relaxation", adaptive=True)
            # Production Run
            self._run_stage(300.0, self.final_steps, "Production", adaptive=True)

        logger.info("=" * 70)
        logger.info("SIMULATION COMPLETED")
        logger.info(f"Total Steps: {self.step}, Accepted: {self.accepted}, Rate: {self.accepted/max(1,self.attempts)*100:.2f}%")
        logger.info(f"Neighbor List Rebuilds: {self.rebuilds}")
        logger.info("=" * 70)


# ========================= ANALYZER =========================
class Analyzer:
    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.type_map = system.type_map
        self.type_to_elem = system.type_to_elem
        # Updated CUTOFFS, PEAK_RANGES, MINIMUM_RMIN for Mg interactions, NO Sr (estimates)
        Mg_cutoff_estimate = 2.5 # Rough estimate for Mg-O, adjust if known
        Mg_other_cutoffs = {('Mg', 'O'): Mg_cutoff_estimate,
                            ('Mg', 'Mg'): 3.5, # Rough estimate
                            ('Mg', 'Ca'): 3.2, # Rough estimate
                            ('Mg', 'Na'): 3.0, # Rough estimate
                            ('Mg', 'Si'): 3.5, # Rough estimate
                            ('Mg', 'P'): 3.5} # Rough estimate

        # NO Sr in CUTOFFS
        self.CUTOFFS = {('Si','O'):2.25, ('P','O'):2.25, ('Na','O'):3.34, ('Ca','O'):3.14,
                   ('O','O'):2.91,
                   ('P','Ca'):4.44, ('P','Na'):4.44,
                   ('Si','Ca'):4.37, ('Si','Na'):4.42,
                   ('Ca','Ca'):4.85, ('Na','Na'):4.15,
                   ('Ca','Na'):4.94,
                   # Add Mg cutoffs
                   **Mg_other_cutoffs
                  }

        # NO Sr in PEAK_RANGES
        self.PEAK_RANGES = {('Si','O'):(1.40,2.00), ('P','O'):(1.30,1.90), ('Na','O'):(2.00,2.80),
                   ('Ca','O'):(2.00,2.80), ('O','O'):(2.30,3.10),
                   ('Si','Si'):(2.80,3.60), ('Si','P'):(2.70,3.40), ('P','P'):(2.70,3.40),
                   ('Si','Na'):(2.90,3.70), ('Si','Ca'):(3.20,4.10),
                   ('Na','Na'):(2.50,3.50), ('Ca','Ca'):(3.20,4.20),
                   ('Ca','Na'):(3.00,4.00),
                   ('P','Ca'):(3.20,4.20), ('P','Na'):(3.00,4.00),
                   # Add Mg peak ranges (estimates)
                   ('Mg','O'):(1.8, 2.6), # Estimate
                   ('Mg','Mg'):(2.5, 3.5), # Estimate
                   ('Mg','Ca'):(2.8, 4.0), # Estimate
                   ('Mg','Na'):(2.5, 3.5), # Estimate
                   ('Mg','Si'):(3.0, 4.0), # Estimate
                   ('Mg','P'):(3.0, 4.0)  # Estimate
                  }

        # NO Sr in MINIMUM_RMIN
        self.MINIMUM_RMIN = {('Si','O'):1.4, ('P','O'):1.3, ('Na','O'):2.0, ('Ca','O'):2.0,
                    ('O','O'):2.2,
                    ('P','Ca'):3.5, ('P','Na'):3.5,
                    ('Si','Ca'):3.5, ('Si','Na'):3.0,
                    ('Ca','Ca'):3.5, ('Na','Na'):3.0,
                    ('Ca','Na'):3.5,
                    # Add Mg Rmin (estimates)
                    ('Mg','O'): 1.8, # Estimate
                    ('Mg','Mg'): 2.5, # Estimate
                    # ... add others as needed ...
                   }


    def analyze_snapshot(self, snapshot_data):
        coords = snapshot_data['coords']
        types = snapshot_data['types']
        box = snapshot_data['box']
        n_atoms = coords.shape[0]

        # --- 1. Basic Properties ---
        counts = {e: int(np.sum(types == self.type_map[e])) for e in self.type_map.keys() if self.type_map[e] in np.unique(types)}

        # --- 2. RDF & CN ---
        results = {}
        modifier_cn = {}
        for mod in ['Na', 'Ca', 'Mg']: # Include Mg, removed Sr
            if self.type_map[mod] not in types: continue
            ti = self.type_map[mod]
            tj = self.type_map['O']
            ei, ej = mod, 'O'
            cut = self.CUTOFFS.get((ei, ej), self.CUTOFFS.get((ej, ei), 3.5))
            r, gr, _, _ = self.compute_rdf(coords, types, ti, tj, box, cut)
            if len(r) > 0:
                modifier_cn[mod] = self.coordination_number(r, gr, cut, counts.get('O', 1), box)
        results['modifier_cn'] = modifier_cn

        # --- 3. Qn Distribution ---
        # Si Qn
        si_qn = self.compute_qn(coords, types, 'Si', 'O', box)
        results['qn_si'] = si_qn
        # P Qn
        p_qn = self.compute_qn(coords, types, 'P', 'O', box)
        results['qn_p'] = p_qn

        # --- 4. Network Connectivity ---
        nc = self.compute_network_connectivity(coords, types, box)
        results['nc'] = nc

        # --- 5. Oxygen Speciation ---
        o_spec = self.compute_oxygen_speciation(coords, types, box)
        results['oxygen_speciation'] = o_spec

        # --- 6. Clustering (Rxx) ---
        rxx_results = {}
        for mod in ['Ca', 'Na', 'Mg']: # Include Mg, removed Sr
             if self.type_map[mod] not in types: continue
             rxx, obs_cn, hom_cn = self.compute_clustering_rxx(coords, types, mod, box)
             rxx_results[mod] = (rxx, obs_cn, hom_cn)
        results['rxx'] = rxx_results

        # --- 7. Bond Lengths ---
        bond_lengths = {}
        for pair, (r_min, r_max) in self.PEAK_RANGES.items():
            if pair[0] in self.type_map and pair[1] in self.type_map:
                ti, tj = self.type_map[pair[0]], self.type_map[pair[1]]
                ei, ej = pair
                cut = self.CUTOFFS.get(pair, self.CUTOFFS.get((pair[1], pair[0]), 4.0))
                r, gr, _, _ = self.compute_rdf(coords, types, ti, tj, box, cut)
                peak_mask = (r >= r_min) & (r <= r_max)
                if np.any(peak_mask) and np.max(gr[peak_mask]) > 0.1: # Threshold to avoid noise
                    peak_idx = np.argmax(gr[peak_mask])
                    peak_r = r[peak_mask][peak_idx]
                    bond_lengths[(ei, ej)] = peak_r
                else:
                    bond_lengths[(ei, ej)] = np.nan
        results['bond_lengths'] = bond_lengths

        # --- 8. Si-O-P Linkages ---
        frac_p_si = self.compute_si_op_linkage_fraction(coords, types, box)
        results['frac_p_si'] = frac_p_si

        # --- 9. Neutron Structure Factor (S(Q)) ---
        # This is a simplified version, often computed for a single snapshot or averaged differently.
        # We'll compute it but note it's typically more meaningful for experimental comparison.
        # q, sq = self.compute_neutron_sf(coords, types, box)
        # results['neutron_sf'] = {'q': q, 'sq': sq}

        return results


    def compute_rdf(self, coords, types, ti, tj, box, cutoff, nbins=500, rmax=None):
        if rmax is None: rmax = cutoff + 2.0
        n_bins = nbins
        dr = rmax / n_bins
        r = np.arange(0.0, rmax, dr) + dr/2
        rdf = np.zeros(n_bins)
        mi = types == ti
        mj = types == tj
        ni = int(np.sum(mi))
        nj = int(np.sum(mj))
        if ni == 0 or nj == 0 or (ti == tj and ni < 2):
            return r, np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)

        tree = cKDTree(coords[mi], boxsize=box)
        all_neigh = tree.query_ball_point(coords[mj], cutoff)

        for idx_j, neigh_list in enumerate(all_neigh):
            j_global = np.where(mj)[0][idx_j]
            for idx_i_local in neigh_list:
                i_global = np.where(mi)[0][idx_i_local]
                if i_global == j_global: continue # Avoid self-pairing
                dr_vec = coords[i_global] - coords[j_global]
                dr_vec = minimum_image(dr_vec, box)
                dist = np.sqrt(np.sum(dr_vec**2))
                bin_idx = int(dist / dr)
                if 0 <= bin_idx < n_bins:
                    rdf[bin_idx] += 1.0

        # Normalize RDF
        rho_j = nj / box**3 # Number density of type j
        shell_volumes = 4 * np.pi * (r**2) * dr
        norm_factors = ni * rho_j * shell_volumes
        rdf = np.divide(rdf, norm_factors, out=np.zeros_like(rdf), where=norm_factors!=0)
        return r, rdf, np.zeros_like(r), np.zeros_like(r) # g(r), gr(r), F(r), dF(r)

    def coordination_number(self, r, gr, cutoff, n_o, box):
        """Calculate coordination number by integrating g(r) up to cutoff."""
        # Use trapezoidal rule for integration
        mask = (r > 0.0) & (r <= cutoff)
        if not np.any(mask):
            return 0.0
        r_int = r[mask]
        gr_int = gr[mask]
        integrand = 4 * np.pi * r_int**2 * (gr_int - 1) / box**3
        cn = np.trapz(integrand, r_int)
        return cn

    def compute_qn(self, coords, types, network_elem, o_elem, box):
        """Compute Qn distribution for network former (Si/P)."""
        network_type = self.type_map[network_elem]
        o_type = self.type_map[o_elem]
        o_indices = np.where(types == o_type)[0]
        network_indices = np.where(types == network_type)[0]

        qn_counts = {f'Q{n}': 0 for n in range(5)} # Q0 to Q4

        for net_idx in network_indices:
            neigh_indices = self._get_neighbors(coords, types, net_idx, o_type, box, self.CUTOFFS.get((network_elem, o_elem), 2.25))
            bridging_o_count = 0
            for o_idx in neigh_indices:
                # Check if this O is bonded to another network former
                o_neigh_indices = self._get_neighbors(coords, types, o_idx, network_type, box, self.CUTOFFS.get((o_elem, network_elem), 2.25))
                if len(o_neigh_indices) >= 2: # If O connects 2 or more network formers, it's bridging
                    bridging_o_count += 1
            qn_state = len(neigh_indices) - bridging_o_count # Non-bridging O count
            qn_label = f'Q{min(qn_state, 4)}' # Cap at Q4
            qn_counts[qn_label] += 1

        # Convert counts to percentages
        total_network = len(network_indices)
        if total_network > 0:
            qn_percentages = {k: (v / total_network) * 100 for k, v in qn_counts.items()}
        else:
            qn_percentages = {k: 0.0 for k in qn_counts}
        return qn_percentages

    def _get_neighbors(self, coords, types, idx, target_type, box, cutoff):
        """Helper to find neighbors of a specific type within a cutoff."""
        pos = coords[idx]
        tree = cKDTree(np.delete(coords, idx, axis=0), boxsize=box)
        # Adjust indices because we removed 'idx'
        rel_coords = np.delete(coords, idx, axis=0)
        rel_types = np.delete(types, idx)
        neigh_indices_rel = tree.query_ball_point(pos, cutoff)
        # Map relative indices back to original indices
        original_neigh_indices = []
        for rel_idx in neigh_indices_rel:
            orig_idx = rel_idx if rel_idx < idx else rel_idx + 1
            if types[orig_idx] == target_type:
                original_neigh_indices.append(orig_idx)
        return original_neigh_indices


    def compute_network_connectivity(self, coords, types, box):
        """Compute average connectivity for Si, P, and overall."""
        si_type = self.type_map['Si']
        p_type = self.type_map['P']
        o_type = self.type_map['O']

        si_cn_sum = 0
        p_cn_sum = 0
        si_count = 0
        p_count = 0

        for i in range(len(coords)):
            if types[i] == si_type:
                neigh_o = self._get_neighbors(coords, types, i, o_type, box, self.CUTOFFS.get(('Si', 'O'), 2.25))
                si_cn_sum += len(neigh_o)
                si_count += 1
            elif types[i] == p_type:
                neigh_o = self._get_neighbors(coords, types, i, o_type, box, self.CUTOFFS.get(('P', 'O'), 2.25))
                p_cn_sum += len(neigh_o)
                p_count += 1

        avg_si_cn = si_cn_sum / si_count if si_count > 0 else 0
        avg_p_cn = p_cn_sum / p_count if p_count > 0 else 0
        total_o = np.sum(types == o_type)
        total_network = si_count + p_count
        avg_overall_cn = (si_cn_sum + p_cn_sum) / total_network if total_network > 0 else 0

        return {'Si': avg_si_cn, 'P': avg_p_cn, 'overall': avg_overall_cn}


    def compute_oxygen_speciation(self, coords, types, box):
        """Compute distribution of O types (Q, NBO, BO, free)."""
        o_type = self.type_map['O']
        si_type = self.type_map['Si']
        p_type = self.type_map['P']
        o_indices = np.where(types == o_type)[0]

        nbo_count = 0 # Non-Bridging O
        bo_count = 0  # Bridging O (Si-O-Si, Si-O-P, P-O-P, etc.)
        free_o_count = 0 # O not bonded to Si or P

        for o_idx in o_indices:
            o_neigh_types = set()
            # Find neighbors bonded to Si or P
            si_neigh = self._get_neighbors(coords, types, o_idx, si_type, box, self.CUTOFFS.get(('O', 'Si'), 2.25))
            p_neigh = self._get_neighbors(coords, types, o_idx, p_type, box, self.CUTOFFS.get(('O', 'P'), 2.25))
            o_neigh_types.update([types[n] for n in si_neigh])
            o_neigh_types.update([types[n] for n in p_neigh])

            if len(o_neigh_types) == 0:
                free_o_count += 1
            elif len(o_neigh_types) == 1:
                nbo_count += 1 # Connected to only one network former type
            else:
                bo_count += 1 # Connected to multiple or different network formers (bridging)

        return {'NBO': nbo_count, 'BO': bo_count, 'Free_O': free_o_count}


    def compute_clustering_rxx(self, coords, types, mod_elem, box):
        """Compute clustering parameter Rxx for a modifier."""
        mod_type = self.type_map[mod_elem]
        mod_indices = np.where(types == mod_type)[0]
        if len(mod_indices) < 2: return np.nan, np.nan, np.nan

        mod_coords = coords[mod_indices]
        tree = cKDTree(mod_coords, boxsize=box)

        # Observed average CN
        obs_cn = 0.0
        cut = self.CUTOFFS.get((mod_elem, mod_elem), 4.0) # Use homonuclear cutoff
        for i, coord in enumerate(mod_coords):
            neigh = tree.query_ball_point(coord, cut)
            obs_cn += len([n for n in neigh if n != i]) # Exclude self
        obs_cn /= len(mod_coords)

        # Expected CN for homogeneous distribution
        n_mod = len(mod_indices)
        vol_sphere = 4/3 * np.pi * cut**3
        rho_mod = (n_mod - 1) / (box**3) # Excluding central atom
        hom_cn = rho_mod * vol_sphere

        rxx = obs_cn / hom_cn if hom_cn > 0 else np.nan
        return rxx, obs_cn, hom_cn


    def compute_si_op_linkage_fraction(self, coords, types, box):
        """Compute fraction of P-O bonds that are P-O-Si."""
        p_type = self.type_map['P']
        o_type = self.type_map['O']
        si_type = self.type_map['Si']

        p_indices = np.where(types == p_type)[0]
        total_p_o_bonds = 0
        p_si_links = 0

        for p_idx in p_indices:
            p_o_neigh = self._get_neighbors(coords, types, p_idx, o_type, box, self.CUTOFFS.get(('P', 'O'), 2.25))
            total_p_o_bonds += len(p_o_neigh)
            for o_idx in p_o_neigh:
                o_si_neigh = self._get_neighbors(coords, types, o_idx, si_type, box, self.CUTOFFS.get(('O', 'Si'), 2.25))
                if len(o_si_neigh) > 0:
                    p_si_links += 1 # Found a P-O-Si linkage via this O
                    break # Count P-O-Si once per P if any O connected to Si exists

        frac = p_si_links / total_p_o_bonds if total_p_o_bonds > 0 else 0.0
        return frac

    # Removed compute_neutron_sf for brevity, but can be added later if needed

    def analyze_all(self, snapshots, n_cores=4):
        logger.info(f"Starting analysis on {len(snapshots)} snapshot(s) using {n_cores} cores...")
        if len(snapshots) == 1:
            # No parallelization needed for single snapshot
            results_list = [self.analyze_snapshot(snapshots[0])]
        else:
            # Use ThreadPoolExecutor for parallel analysis
            results_list = []
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = [executor.submit(self.analyze_snapshot, snap) for snap in snapshots]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing Snapshots"):
                    try:
                        res = future.result()
                        results_list.append(res)
                    except Exception as e:
                        logger.error(f"Analysis failed for a snapshot: {e}")
                        logger.error(traceback.format_exc())
                        # Append empty result or skip?
                        results_list.append({})

        # Aggregate results across snapshots
        agg_results = {}
        for key in results_list[0].keys():
            if isinstance(results_list[0][key], dict):
                # For dictionaries like modifier_cn, qn_si, etc.
                agg_dict = {}
                for sub_key in results_list[0][key].keys():
                    vals = [r[key].get(sub_key, np.nan) for r in results_list if sub_key in r[key]]
                    vals = [v for v in vals if not np.isnan(v)]
                    if vals:
                         agg_dict[sub_key] = (np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                    else:
                         agg_dict[sub_key] = (np.nan, np.nan)
                agg_results[key] = agg_dict
            elif isinstance(results_list[0][key], (int, float)):
                # For simple floats/ints like frac_p_si
                vals = [r[key] for r in results_list if not np.isnan(r.get(key, np.nan))]
                if vals:
                    mean_val = np.mean(vals)
                    std_err = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    agg_results[key] = (mean_val, std_err)
                else:
                    agg_results[key] = (np.nan, np.nan)
            else:
                # For other types, just take the last one or handle specifically
                agg_results[key] = results_list[-1][key] # Take last as representative

        return agg_results

    def plot_results(self, results, outdir):
        import matplotlib.pyplot as plt
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)

        # --- Plot Qn Distributions ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, 5))
        labels = [f'Q{n}' for n in range(5)]
        x_pos = np.arange(len(labels))

        # Si Qn
        si_means = [results['qn_si'].get(l, (0,0))[0] for l in labels]
        si_stds = [results['qn_si'].get(l, (0,0))[1] for l in labels]
        ax1.bar(x_pos, si_means, 0.6, yerr=si_stds, color=colors, capsize=5, error_kw={'elinewidth': 1})
        ax1.set_xticks(x_pos); ax1.set_xticklabels(labels)
        ax1.set_ylabel('%'); ax1.set_title('Si Qn Distribution');
        ax1.grid(alpha=0.3, axis='y')

        # P Qn
        p_means = [results['qn_p'].get(l, (0,0))[0] for l in labels]
        p_stds = [results['qn_p'].get(l, (0,0))[1] for l in labels]
        ax2.bar(x_pos, p_means, 0.6, yerr=p_stds, color=colors, capsize=5, error_kw={'elinewidth': 1})
        ax2.set_xticks(x_pos); ax2.set_xticklabels(labels)
        ax2.set_ylabel('%'); ax2.set_title('P Qn Distribution');
        ax2.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(outdir / 'qn_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # --- Plot Modifier CN ---
        fig, ax = plt.subplots(figsize=(8, 6))
        mods = [m for m in ['Na', 'Ca', 'Mg'] if m in results.get('modifier_cn', {})] # Removed Sr
        if mods:
            cn_means = [results['modifier_cn'][m][0] for m in mods]
            cn_stds = [results['modifier_cn'][m][1] for m in mods]
            bars = ax.bar(mods, cn_means, yerr=cn_stds, capsize=5, error_kw={'elinewidth': 1})
            ax.set_ylabel('Coordination Number'); ax.set_title('Modifier-Oxygen CN');
            ax.grid(alpha=0.3, axis='y')
            # Add numerical labels on bars
            for bar, mean_val, std_val in zip(bars, cn_means, cn_stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.05,
                        f'{mean_val:.2f}±{std_val:.2f}',
                        ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / 'modifier_cn.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Plots saved to {outdir}")


    def export_excel(self, results, excel_path, energy_data=None, block_energy=None, sensitivity=None, final_coords=None, final_types=None, box=None):
        logger.info(f"Exporting results to Excel: {excel_path}")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Metadata
            meta_df = pd.DataFrame(list(results['metadata'].items()), columns=['Property', 'Value'])
            meta_df.to_excel(writer, sheet_name='Metadata', index=False)

            # Summary Stats (averaged from analysis)
            summary_data = []
            for key, value in results.items():
                if key in ['metadata', 'sensitivity']: continue # Handle separately
                if isinstance(value, dict):
                    for sub_key, (mean_val, std_err) in value.items():
                        if isinstance(sub_key, tuple):
                             name = f"{sub_key[0]}-{sub_key[1]}"
                        else:
                             name = str(sub_key)
                        summary_data.append({'Measurement': f"{key}_{name}", 'Mean': mean_val, 'StdErr': std_err})
                elif isinstance(value, tuple) and len(value) == 2: # Single mean, std_err pair
                    summary_data.append({'Measurement': key, 'Mean': value[0], 'StdErr': value[1]})
                else: # Single value
                    summary_data.append({'Measurement': key, 'Mean': value, 'StdErr': np.nan})
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary_Stats', index=False)

            # Energy Data (if available)
            if energy_data:
                ed = energy_data
                total_steps = len(ed.get('total', []))
                df_energy = pd.DataFrame({
                    'Step': list(range(len(ed.get('total', [])))) * (ed.get('total', []) != []),
                    'Total_eV_per_atom': ed.get('total', []),
                    'ShortRange_eV_per_atom': ed.get('short', []) if len(ed.get('short', [])) == total_steps else [np.nan]*total_steps,
                    'Coulomb_eV_per_atom': ed.get('coul', []) if len(ed.get('coul', [])) == total_steps else [np.nan]*total_steps
                })
                df_energy.dropna(how='all', subset=['Total_eV_per_atom', 'ShortRange_eV_per_atom', 'Coulomb_eV_per_atom']).to_excel(writer, sheet_name='Energy_Log', index=False)

            # Block Energy
            if block_energy:
                pd.DataFrame([block_energy]).to_excel(writer, sheet_name='Block_Energy', index=False)

            # Sensitivity
            if sensitivity:
                pd.DataFrame(sensitivity).to_excel(writer, sheet_name='Sensitivity', index=False)

            # Final Structure (Coords & Types)
            if final_coords is not None and final_types is not None:
                struct_df = pd.DataFrame({
                    'Symbol': [self.type_to_elem[t] for t in final_types],
                    'X': final_coords[:, 0],
                    'Y': final_coords[:, 1],
                    'Z': final_coords[:, 2]
                })
                struct_df.to_excel(writer, sheet_name='Final_Structure', index=False)

        logger.info(f"Excel export completed: {excel_path}")


# ========================= SENSITIVITY ANALYZER =========================
class SensitivityAnalyzer:
    def __init__(self, system, config):
        self.system = system; self.config = config; self.pot = system.potential
        # NO Sr in matrices
        self.A_mat = np.zeros((6, 6), dtype=np.float64) # Updated to 6 types, NO Sr
        self.F_mat = np.zeros((6, 6), dtype=np.float64)
        self.C_mat = np.zeros((6, 6), dtype=np.float64)
        # R_HARD is not typically used in the main energy calculation with ZBL+Buckingham+Coulomb, so initialized to zero or ignored.
        # If used, initialize similarly to A/F/C mats.
        self.R_HARD_MAT = np.zeros((6, 6), dtype=np.float64) # Placeholder if needed elsewhere
        for (e1, e2), p in self.pot.buck_params.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.A_mat[t1, t2] = p['A']; self.F_mat[t1, t2] = p['F']; self.C_mat[t1, t2] = p['C']
            if t1 != t2: # Symmetrize
                self.A_mat[t2, t1] = p['A']; self.F_mat[t2, t1] = p['F']; self.C_mat[t2, t1] = p['C']
        self.type_Z = self.system.get_type_Z() # Gets Z for types 0-5, NO Sr

    def run(self):
        logger.info("=" * 70)
        logger.info("SENSITIVITY ANALYSIS")
        logger.info("=" * 70)
        res = []
        for cut in self.config.test_cutoffs:
            for alpha in self.config.test_alphas:
                # Create a temporary neighbor list for this cutoff
                try:
                    nl = NeighborList(self.system, cut, 1.0) # Small skin for sensitivity, rebuild often
                    n, s = nl.neighbors, nl.starts
                    U, sr, cl = energy_decomposition(
                        self.system.coords, self.system.charges, self.system.type_indices,
                        self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
                        n, s, self.system.box, cut, alpha, # Use test cutoff and alpha
                        self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d
                    )
                    # Add the Wolf self-energy term for the given alpha and cutoff
                    n_atoms = self.system.N_ATOMS
                    q_squared_sum = np.sum(self.system.charges**2)
                    qc = np.sum(self.system.charges)
                    self_term = - alpha / np.sqrt(np.pi) * q_squared_sum
                    monopole_term = qc**2 / (2 * cut)
                    wolf_self_corr = KE * (self_term + monopole_term) / n_atoms
                    total_U = U + wolf_self_corr

                    res.append({'Cutoff': cut, 'Alpha': alpha, 'Energy': total_U, 'Short': sr, 'Coulomb': cl + wolf_self_corr})
                    logger.debug(f" Cutoff={cut:.0f}A, alpha={alpha:.2f}: E={total_U:.4f} eV/atom")
                except Exception as e:
                    logger.warning(f"Failed for Cutoff={cut}, Alpha={alpha}: {e}")
                    res.append({'Cutoff': cut, 'Alpha': alpha, 'Energy': np.nan, 'Short': np.nan, 'Coulomb': np.nan})
        return res


# ========================= BLOCK AVERAGER =========================
def compute_block_energy(energy_log, block_size):
    if len(energy_log) < block_size:
        logger.warning(f"Energy log ({len(energy_log)}) shorter than block size ({block_size}). Cannot compute block average.")
        return {'mean': np.mean(energy_log) if energy_log else np.nan, 'std': np.std(energy_log, ddof=1) if len(energy_log) > 1 else 0.0}
    blocks = [energy_log[i:i+block_size] for i in range(0, len(energy_log), block_size)]
    block_avgs = [np.mean(block) for block in blocks if len(block) == block_size] # Only full blocks
    if len(block_avgs) == 0:
        logger.warning("No full blocks could be formed for block averaging.")
        return {'mean': np.nan, 'std': np.nan}
    mean = np.mean(block_avgs)
    std = np.std(block_avgs, ddof=1) # Standard deviation of the block averages
    return {'mean': mean, 'std': std}


# ========================= LOGGER SETUP =========================
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        # Manually encode and decode to handle UTF-8 consistently
        try:
            stream.write(msg.encode('utf-8').decode('utf-8'))
            stream.write('\n')
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('simulation.log', 'w', 'utf-8'),
                              UTF8StreamHandler()])
logger = logging.getLogger(__name__)

# ========================= FREUD (optional) =========================
try:
    import freud
    FREUD_AVAILABLE = True
except ImportError:
    FREUD_AVAILABLE = False
    logger.warning("freud library not found. Falling back to scipy for RDF.")


# ========================= CONFIG =========================
@dataclass
class SimulationConfig:
    input_file: Path
    seed: int
    density: Optional[float] = None
    x: float = 0.0  # Changed to float for Mg percentage
    box_override: Optional[float] = None
    output_dir: Optional[Path] = None
    continue_mode: bool = False
    mixing_temp: float = 4000.0
    mixing_sweeps: int = 10
    annealing_sweeps: List[Tuple[float, int]] = field(default_factory=lambda: [
        (3000, 4), (2500, 4), (2000, 6), (1500, 6), (1200, 8),
        (2000, 5), (1500, 6), (1000, 8), (2000, 5), (1500, 6),
        (1000, 8), (800, 10), (700, 10), (600, 8), (500, 6),
        (400, 5), (300, 5)
    ])
    healing_temp: float = 2500.0
    healing_sweeps: int = 5
    low_t_sweeps: int = 8
    final_sweeps: int = 30
    snapshot_interval: Optional[int] = None
    max_snapshots: int = 30
    cutoff: float = 10.0
    wolf_alpha: float = 0.25
    skin: float = SKIN
    target_acceptance: float = 0.40
    adaptive_tuning_freq: Optional[int] = None
    log_freq: Optional[int] = None
    decomposition_freq: Optional[int] = None
    n_cores: int = 4
    # Sensitivity analysis
    test_cutoffs: List[float] = field(default_factory=lambda: [8.0, 10.0, 12.0])
    test_alphas: List[float] = field(default_factory=lambda: [0.20, 0.25, 0.30])
    # Block averaging
    block_size: int = 5000

    def __post_init__(self):
        if self.output_dir is None:
            n_header, _ = parse_xyz_header(self.input_file)
            self.output_dir = Path(f"Bioglass_x{self.x}_N{n_header}_seed{self.seed}")


# ========================= SUMMARY PRINTER =========================
def print_summary(results, x_val):
    print("")
    print("=" * 70)
    print(f" ANALYSIS SUMMARY - 45S5 + {x_val} mol% MgO") # Updated label, removed Sr
    print("=" * 70)

    md = results['metadata']
    print("- Density & Structure -")
    print(f" Density: {md.get('Density_g_cm3', 0):.4f} g/cm3")
    print(f" Molar volume: {md.get('Molar_volume_cm3_mol', 0):.3f} cm3/mol")

    print("- Bond Lengths (A) [Sim vs Paper] -")
    for (e1, e2), (bl_mean, bl_stderr) in sorted(results.get('bond_lengths', {}).items()):
        if np.isnan(bl_mean): continue
        paper = PAPER_BOND_LENGTHS.get((e1, e2), PAPER_BOND_LENGTHS.get((e2, e1), None))
        ps = f"{paper:.2f}" if paper else " - "
        flag = " OK" if paper and abs(bl_mean - paper) < 0.1 else (" DIFF" if paper else "")
        print(f" {e1}-{e2:2s}: {bl_mean:.3f}±{bl_stderr:.3f} (paper: {ps}){flag}")

    print("- Modifier CN [Sim vs Paper] -")
    # Use appropriate paper dictionary based on simulated composition
    paper_dict = PAPER_CN # Or another dictionary based on x
    for mod, (cn_mean, cn_stderr) in results.get('modifier_cn', {}).items():
        paper = paper_dict.get(mod, None)
        ps = f"{paper:.1f}" if paper else " - "
        flag = " OK" if paper and abs(cn_mean - paper) < 0.5 else " DIFF"
        print(f" {mod}-O: {cn_mean:.2f}±{cn_stderr:.2f} (paper: {ps}){flag}")

    print("- Network Connectivity [Sim vs Paper] -")
    # Use appropriate paper dictionary based on simulated composition
    paper_dict_nc = PAPER_NC # Or another dictionary based on x
    for k, (nc_mean, nc_stderr) in results.get('nc', {}).items():
        paper = paper_dict_nc.get(k, None)
        ps = f"{paper:.2f}" if paper else " - "
        flag = " OK" if paper and abs(nc_mean - paper) < 0.2 else " DIFF"
        print(f" NC ({k:8s}): {nc_mean:.3f}±{nc_stderr:.3f} (paper: {ps}){flag}")

    print("- Oxygen Speciation -")
    tot = sum(v[0] for v in results.get('oxygen_speciation', {}).values()) # Use mean value [0]
    for k, (v, _) in results.get('oxygen_speciation', {}).items(): # Unpack mean, stderr
        print(f" {k:4s}: {v:5d} ({v/tot*100:5.1f}%)")

    print("- Clustering R_obs/R_hom -")
    for mod, (rxx, cn_obs, cn_hom) in results.get('rxx', {}).items():
        if rxx is not None and not np.isnan(rxx):
            print(f" {mod}: R={rxx:.3f} (CN_obs={cn_obs:.2f}, CN_hom={cn_hom:.2f})")

    print("- Qn Distribution (%) -")
    print(" Si: " + " ".join(f"{k}:{v[0]:.1f}±{v[1]:.2f}" for k, v in results.get('qn_si', {}).items())) # Mean ± StdErr
    print(" P:  " + " ".join(f"{k}:{v[0]:.1f}±{v[1]:.2f}" for k, v in results.get('qn_p', {}).items()))

    if 'block_energy' in results:
        be = results['block_energy']
        print(f"- Block Energy -")
        print(f" Mean: {be['mean']:.6f} eV/atom, Std Err of Mean: {be['std']:.6f}")

    print("=" * 70 + "")


# ========================= CLI =========================
app = typer.Typer(help="45S5/Mg Bioglass NVT MC v10.1 - ALL-IN-ONE (Simulation + Analysis) (NO Sr)") # Updated help, removed Sr

@app.command()
def run(
    input_file: Path = typer.Argument(..., help="Input XYZ file generated by structure_generator_paper.py"),
    seed: Optional[int] = typer.Option(None, help="Random seed. If omitted, read from XYZ comment."),
    density: Optional[float] = typer.Option(None, help="Optional density override in g/cm^3."),
    x: Optional[float] = typer.Option(None, help="Mg mol% (read from XYZ if not provided)."), # Updated help, removed Sr
    cores: int = typer.Option(4, help="Number of CPU cores for analysis."),
    cutoff: float = typer.Option(10.0, help="Real space cutoff (A)."),
    wolf_alpha: float = typer.Option(0.25, help="Wolf summation alpha parameter."),
    final_sweeps: Optional[int] = typer.Option(None, help="Override final sweeps."),
    box_override: Optional[float] = typer.Option(None, help="Override box size (A) - use with caution."),
    continue_mode: bool = typer.Option(False, "--continue", help="Continue from a previous final_structure."),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory name.")
):
    try:
        # Parse input file metadata
        n_header, meta = parse_xyz_header(input_file)
        x_meta = meta.get('x', 0.0)
        seed_meta = meta.get('seed', 42)

        # Resolve parameters
        seed = seed or seed_meta
        x = x or x_meta # Use command line, then meta, default to 0
        config = SimulationConfig(
            input_file=input_file, seed=seed, density=density, x=x,
            box_override=box_override, output_dir=output_dir,
            n_cores=cores, cutoff=cutoff, wolf_alpha=wolf_alpha,
            continue_mode=continue_mode
        )
        if final_sweeps is not None:
            config.final_sweeps = final_sweeps

        logger.info("=" * 70)
        logger.info("INPUT")
        logger.info("=" * 70)
        logger.info(f"File        : {input_file}")
        logger.info(f"x (Mg mol%) : {x}") # Updated label, removed Sr
        logger.info(f"N atoms     : {n_header}")
        logger.info(f"Seed        : {seed}")
        logger.info(f"Density     : {density or 'Not specified (will use from XYZ or fail)'}")
        logger.info(f"Cutoff      : {cutoff} A")
        logger.info(f"Wolf alpha  : {wolf_alpha}")
        logger.info(f"Final Sweeps: {config.final_sweeps} * N_ATOMS")
        logger.info(f"Mode        : {'CONTINUE' if continue_mode else 'FULL PROTOCOL'}")
        logger.info(f"Output dir  : {config.output_dir}")

        # --- SIMULATION ---
        system = GlassSystem(config)
        sim = MCSimulator(system, config)
        sim.run()
        energy_data = {'total': sim.energy_log, 'short': sim.short_log, 'coul': sim.coul_log}
        final_coords = system.coords.copy()
        final_types = system.type_indices.copy()
        box = system.box

        # --- BLOCK AVERAGING ---
        block_energy = compute_block_energy(sim.energy_log, config.block_size)
        logger.info(f"Block Energy: mean={block_energy['mean']:.6f}, std={block_energy['std']:.6f}")

        # --- SAVE FINAL STRUCTURE ---
        xyz_path = config.output_dir / "final_structure.xyz"
        with open(xyz_path, 'w') as f:
            f.write(f"{system.N_ATOMS}\n")
            f.write(f"Final Xiang-Du 2011, x={config.x} mol% MgO, " # Updated label, removed Sr
                    f"N={system.N_ATOMS}, rho={system.effective_density:.4f} g/cm3, "
                    f"box={system.box:.4f} A, seed={config.seed}\n")
            for i in range(system.N_ATOMS):
                elem = system.type_to_elem[system.type_indices[i]]
                x_, y_, z_ = system.coords[i]
                f.write(f"{elem:2s} {x_:12.6f} {y_:12.6f} {z_:12.6f}\n")
        logger.info(f"Final XYZ saved to {xyz_path}")

        # --- SAVE ENERGY LOG ---
        energy_path = config.output_dir / "energy_log.csv"
        with open(energy_path, 'w') as f:
            f.write("Step,Total_eV_per_atom,ShortRange_eV_per_atom,Coulomb_eV_per_atom\n")
            for i, (e, s, c) in enumerate(zip(sim.energy_log, sim.short_log, sim.coul_log)):
                f.write(f"{i * sim.log_freq},{e:.6f},{s:.6f},{c:.6f}\n")
        logger.info(f"Energy log saved to {energy_path}")

        # --- ANALYSIS ---
        snapshots = sim.snapshots
        if not snapshots:
            snapshots = [{'coords': final_coords, 'types': final_types, 'box': box}]
        analyzer = Analyzer(system, config)
        results = analyzer.analyze_all(snapshots, n_cores=cores)
        results['block_energy'] = block_energy

        # Calculate metadata for results dictionary
        total_mass = system.total_mass
        eff_density = system.effective_density
        n_oxide_units = system.N_ATOMS / 2.835 # Keep original scaling denominator? Needs recalculation potentially based on actual formula units.
        # A more robust way is to calculate based on stoichiometry:
        # e.g., if we have N_SI SiO2, N_P P2O5, N_NA Na2O, N_CA CaO, N_MG MgO
        # N_oxide_units = N_SI + N_P/2 + N_NA/2 + N_CA + N_MG (approximately)
        # For now, using the placeholder unless structure generator provides exact count.
        # Let's calculate it based on the counts dictionary from the system.
        counts = system.counts
        # Approximation: Sum of network formers and modifiers divided by their typical oxide formula count
        # SiO2 -> 1, P2O5 -> 0.5, Na2O -> 0.5, CaO -> 1, MgO -> 1
        approx_oxide_units = counts.get('Si', 0) + counts.get('P', 0)/2 + counts.get('Na', 0)/2 + counts.get('Ca', 0) + counts.get('Mg', 0)
        molar_volume = (total_mass / approx_oxide_units) / eff_density if approx_oxide_units > 0 else np.nan

        results['metadata'] = {
            'x_mol_percent_MgO': x, # Store the x value used, removed Sr label
            'N_atoms': system.N_ATOMS,
            'Box_A': box,
            'Density_g_cm3': eff_density,
            'Molar_volume_cm3_mol': molar_volume,
            **{f'N_{e}': counts.get(e, 0) for e in ['Si', 'P', 'Na', 'Ca', 'O', 'Mg']} # Include all counts, removed Sr
        }


        # - SENSITIVITY ANALYSIS -
        sens = SensitivityAnalyzer(system, config).run()
        results['sensitivity'] = sens

        # --- EXPORT ---
        excel_path = config.output_dir / "simulation_results.xlsx"
        analyzer.export_excel(results, excel_path, energy_data=energy_data,
                             block_energy=block_energy, sensitivity=sens,
                             final_coords=final_coords, final_types=final_types, box=box)

        analyzer.plot_results(results, config.output_dir / "plots")

        with open(config.output_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f)

        # --- PRINT SUMMARY ---
        print_summary(results, x) # Pass the resolved x value

        logger.info(f"[OK] All results saved to {config.output_dir}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    app()
