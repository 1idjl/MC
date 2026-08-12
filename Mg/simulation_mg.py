#!/usr/bin/env python3
"""
========================================================================
45S5/Mg Bioglass NVT Monte Carlo - ULTIMATE VERSION v4.3
Combines the raw speed of v4.2 (Numba + Random Sampling) 
with the scientific accuracy and full I/O of v3.0.
========================================================================
"""
import numpy as np
import sys
import logging
import re
import warnings
import codecs
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm
from numba import njit
from math import erfc, exp, sqrt, pi

warnings.filterwarnings('ignore')

class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None: stream = sys.stdout
        if hasattr(stream, 'buffer'): stream = codecs.getwriter('utf-8')(stream.buffer)
        super().__init__(stream)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('simulation.log', 'w', 'utf-8'), UTF8StreamHandler()])
logger = logging.getLogger(__name__)

KB_EV = 8.617333262145e-5
NA = 6.02214076e23
KE = 14.3996454784255
SKIN = 1.5
SWITCH_DR = 0.3
BASE_N_ATOMS = 2835

class SimulationError(Exception): pass
class FileError(SimulationError): pass

def parse_xyz_header(path: Path):
    if not path.exists(): raise FileError(f"File not found: {path}")
    with open(path, 'r') as f:
        first = f.readline().strip()
        second = f.readline().strip()
    try: n_atoms = int(first)
    except Exception as exc: raise FileError(f"Cannot parse atom count: {first}") from exc
    meta = {'x': 0, 'N': n_atoms, 'density': None, 'box': None, 'seed': None}
    patterns = [('x', r"x\s*=\s*(\d+)", int), ('N', r"N\s*=\s*(\d+)", int),
                ('density', r"rho\s*=\s*([0-9]*\.?[0-9]+)", float), ('box', r"box\s*=\s*([0-9]*\.?[0-9]+)", float),
                ('seed', r"seed\s*=\s*(\d+)", int)]
    for key, pat, conv in patterns:
        m = re.search(pat, second)
        if m: meta[key] = conv(m.group(1))
    return n_atoms, second, meta

@dataclass
class SimulationConfig:
    input_file: Path
    seed: int
    density: Optional[float] = None
    x: int = 0
    box_override: Optional[float] = None
    output_dir: Optional[Path] = None
    continue_mode: bool = False
    mixing_temp: float = 5000.0
    mixing_sweeps: int = 40
    annealing_sweeps: List[Tuple[float, int]] = field(default_factory=lambda: [
        (4500, 8), (4000, 8), (3500, 10), (3000, 15), (2500, 25), (2000, 25), (1500, 20),
        (1200, 12), (1000, 12), (800, 10), (700, 10), (600, 8), (500, 6), (400, 6), (300, 6)])
    healing_temp: float = 2500.0
    healing_sweeps: int = 15
    low_t_sweeps: int = 8
    final_sweeps: int = 20000
    snapshot_interval: int = 5000
    cutoff: float = 12.0       # Kept at 12.0 for scientific accuracy
    wolf_alpha: float = 0.20   # Matches 12.0 cutoff
    skin: float = SKIN
    target_acceptance: float = 0.40
    swap_move_freq: int = 100
    bond_break_freq: int = 50
    block_size: int = 5000
    checkpoint_interval: int = 10000

    def __post_init__(self):
        if self.output_dir is None: self.output_dir = Path(f"Bioglass_Mg{self.x}_seed{self.seed}")
        else: self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class Potential:
    def __init__(self):
        self.zbl_a0 = 0.46850
        self.zbl_c = np.array([0.1818, 0.5099, 0.2802, 0.02817])
        self.zbl_d = np.array([3.2, 0.9423, 0.4029, 0.2016])
        self.zbl_z = {'Si': 14.0, 'Ca': 20.0, 'Na': 11.0, 'P': 15.0, 'O': 8.0, 'Mg': 12.0}
        self.buck_params = {
            ('Si', 'O'): {'A': 13702.905, 'F': 0.193817, 'C': 54.681}, ('Ca', 'O'): {'A': 7747.1834, 'F': 0.252623, 'C': 93.109},
            ('P',  'O'): {'A': 26655.472, 'F': 0.181968, 'C': 86.856}, ('Na', 'O'): {'A': 4383.7555, 'F': 0.243838, 'C': 30.70},
            ('Mg', 'O'): {'A': 7063.0,    'F': 0.2109,   'C': 19.21}, ('O',  'O'): {'A': 2029.2204, 'F': 0.343645, 'C': 192.58},
            ('Si', 'Si'): {'A': 5000.0, 'F': 0.25, 'C': 0.0}, ('Si', 'P'):  {'A': 5000.0, 'F': 0.25, 'C': 0.0}, ('P',  'P'):  {'A': 5000.0, 'F': 0.25, 'C': 0.0},
        }
        self.r_hard_default = 0.9
        self.r_hard_by_pair = {('O', 'O'): 1.4, ('Mg', 'O'): 1.0, ('Si', 'Si'): 1.8, ('Si', 'P'): 1.8, ('P', 'P'): 1.8}

class GlassSystem:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977, 'P': 30.97376, 'O': 15.999, 'Mg': 24.305}
        self.type_map = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Mg': 5}
        self.type_to_elem = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Mg'}
        self.potential = Potential()
        if not config.input_file.exists(): raise FileError(f"File not found: {config.input_file}")
        self._load_xyz()
        self._setup_box()
        self._compute_charges()
    
    def _load_xyz(self):
        with open(self.config.input_file, 'r') as f: lines = f.readlines()
        n_header = int(lines[0].strip())
        self.symbols, coords_list = [], []
        for line in lines[2:2 + n_header]:
            parts = line.strip().split()
            if len(parts) < 4: continue
            elem = parts[0]
            self.symbols.append(elem)
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        self.N_ATOMS = len(self.symbols)
        logger.info(f"Loaded {self.N_ATOMS} atoms.")
        self.coords = np.array(coords_list, dtype=np.float64)
        self.type_indices = np.array([self.type_map[s] for s in self.symbols], dtype=np.int32)
        self.total_mass = sum(self.masses[s] for s in self.symbols)
    
    def _setup_box(self):
        if self.config.box_override is not None:
            self.box = float(self.config.box_override)
            self.effective_density = self.total_mass * 1.0e24 / (NA * self.box ** 3)
        elif self.config.density is not None:
            self.effective_density = float(self.config.density)
            self.box = ((self.total_mass / NA) / self.effective_density * 1.0e24) ** (1.0 / 3.0)
        else: raise FileError("No box or density.")
        self.coords = self.coords % self.box
    
    def _compute_charges(self):
        base = {'Si': 2.4, 'Ca': 1.2, 'Na': 0.6, 'P': 3.0, 'O': -1.2, 'Mg': 1.2}
        self.charges = np.array([base.get(s, 0.0) for s in self.symbols], dtype=np.float64)
        # FIX: Ensure exact charge neutrality
        total_charge = float(np.sum(self.charges))
        o_mask = self.type_indices == self.type_map['O']
        n_o = int(np.sum(o_mask))
        if abs(total_charge) > 1.0e-8 and n_o > 0:
            self.charges[o_mask] -= total_charge / n_o
            logger.info("O charge adjusted for exact neutrality.")
    
    def get_type_Z(self):
        return np.array([self.potential.zbl_z[self.type_to_elem[i]] for i in range(6)], dtype=np.float64)

class NeighborList:
    def __init__(self, system, cutoff, skin=SKIN):
        self.system = system
        self.cutoff = cutoff
        self.skin = skin
        self.sc = cutoff + skin
        self.neighbors = None
        self.starts = None
        self.ref_coords = None
        self.rebuild_count = 0
        self._build()
    
    def _build(self):
        tree = cKDTree(self.system.coords, boxsize=self.system.box)
        pairs = tree.query_pairs(self.sc, output_type='ndarray')
        n = len(self.system.coords)
        counts = np.zeros(n, dtype=np.int32)
        for i, j in pairs: counts[i] += 1; counts[j] += 1
        starts = np.zeros(n + 1, dtype=np.int32)
        starts[1:] = np.cumsum(counts)
        neighbors = np.empty(starts[-1], dtype=np.int32)
        fill = starts[:-1].copy()
        for i, j in pairs:
            neighbors[fill[i]] = j; fill[i] += 1
            neighbors[fill[j]] = i; fill[j] += 1
        self.neighbors = neighbors
        self.starts = starts
        self.ref_coords = self.system.coords.copy()
        self.rebuild_count += 1
        return neighbors, starts
    
    def update(self, force=False):
        if force or self._needs_rebuild(): return self._build()
        return self.neighbors, self.starts
    
    def _needs_rebuild(self):
        if self.ref_coords is None: return True
        dr = self.system.coords - self.ref_coords
        dr -= self.system.box * np.round(dr / self.system.box)
        return np.sqrt(np.sum(dr ** 2, axis=1)).max() > self.skin / 3.0

@njit(fastmath=True, cache=True)
def zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d):
    if r < 1.0e-6: return 1.0e8
    a = zbl_a0 / (Zi ** 0.23 + Zj ** 0.23)
    x = r / a
    phi = (zbl_c[0] * exp(-zbl_d[0] * x) + zbl_c[1] * exp(-zbl_d[1] * x) + zbl_c[2] * exp(-zbl_d[2] * x) + zbl_c[3] * exp(-zbl_d[3] * x))
    return KE * Zi * Zj * phi / r

@njit(fastmath=True, cache=True)
def wolf_coulomb(qi, qj, r, alpha, cutoff):
    if r >= cutoff or r < 1.0e-12: return 0.0
    ar = alpha * r; ac = alpha * cutoff
    er = erfc(ar); ec = erfc(ac)
    t1 = er / r; t2 = ec / cutoff
    t3 = ((ec / (cutoff * cutoff)) + (2.0 * alpha / sqrt(pi)) * exp(-ac * ac) / cutoff) * (r - cutoff)
    return KE * qi * qj * (t1 - t2 + t3)

@njit(fastmath=True, cache=True)
def pair_energy(r, qi, qj, ti, tj, Zi, Zj, A_mat, F_mat, C_mat, R_HARD_MAT, cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    if r >= cutoff or r < 1.0e-12: return 0.0
    r_in = R_HARD_MAT[ti, tj]; r_out = r_in + SWITCH_DR
    if r < r_out:
        e_zbl = zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d)
        if r < r_in: return e_zbl
        A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12: e_buck += A * exp(-r / F)
        if C > 0.0: e_buck -= C / (r ** 6)
        e_coul = wolf_coulomb(qi, qj, r, alpha, cutoff)
        e_full = e_buck + e_coul
        x_switch = (r - r_in) / SWITCH_DR
        s = x_switch ** 3 * (10.0 - 15.0 * x_switch + 6.0 * x_switch ** 2)
        return s * e_full + (1.0 - s) * e_zbl
    else:
        A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12: e_buck += A * exp(-r / F)
        if C > 0.0: e_buck -= C / (r ** 6)
        return e_buck + wolf_coulomb(qi, qj, r, alpha, cutoff)

@njit(fastmath=True, cache=True)
def local_energy(idx, coords, charges, types, type_Z, A_mat, F_mat, C_mat, R_HARD_MAT, box, cutoff, alpha, neighbors, starts, zbl_a0, zbl_c, zbl_d):
    e = 0.0
    qi = charges[idx]; ti = types[idx]; Zi = type_Z[ti]
    xi, yi, zi = coords[idx]
    for p in range(starts[idx], starts[idx + 1]):
        j = neighbors[p]
        dx = xi - coords[j, 0]; dy = yi - coords[j, 1]; dz = zi - coords[j, 2]
        dx -= box * round(dx / box); dy -= box * round(dy / box); dz -= box * round(dz / box)
        r2 = dx * dx + dy * dy + dz * dz
        if r2 >= cutoff * cutoff: continue
        r = sqrt(r2)
        e += pair_energy(r, qi, charges[j], ti, types[j], Zi, type_Z[types[j]], A_mat, F_mat, C_mat, R_HARD_MAT, cutoff, alpha, zbl_a0, zbl_c, zbl_d)
    e += -KE * (alpha / sqrt(pi)) * qi * qi
    return e

@njit(fastmath=True, cache=True)
def total_energy(coords, charges, types, type_Z, A_mat, F_mat, C_mat, R_HARD_MAT, neighbors, starts, box, cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    e = 0.0; n = coords.shape[0]
    for i in range(n):
        xi, yi, zi = coords[i]; qi = charges[i]; ti = types[i]; Zi = type_Z[ti]
        for p in range(starts[i], starts[i + 1]):
            j = neighbors[p]
            if j <= i: continue
            dx = xi - coords[j, 0]; dy = yi - coords[j, 1]; dz = zi - coords[j, 2]
            dx -= box * round(dx / box); dy -= box * round(dy / box); dz -= box * round(dz / box)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= cutoff * cutoff: continue
            r = sqrt(r2)
            e += pair_energy(r, qi, charges[j], ti, types[j], Zi, type_Z[types[j]], A_mat, F_mat, C_mat, R_HARD_MAT, cutoff, alpha, zbl_a0, zbl_c, zbl_d)
    return e

@njit(fastmath=True, cache=True)
def energy_decomposition(coords, charges, types, type_Z, A_mat, F_mat, C_mat, R_HARD_MAT, neighbors, starts, box, cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    sr = 0.0; cl = 0.0; n = coords.shape[0]
    for i in range(n):
        xi, yi, zi = coords[i]; qi = charges[i]; ti = types[i]; Zi = type_Z[ti]
        for p in range(starts[i], starts[i + 1]):
            j = neighbors[p]
            if j <= i: continue
            dx = xi - coords[j, 0]; dy = yi - coords[j, 1]; dz = zi - coords[j, 2]
            dx -= box * round(dx / box); dy -= box * round(dy / box); dz -= box * round(dz / box)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= cutoff * cutoff: continue
            r = sqrt(r2)
            tj = types[j]; rh = R_HARD_MAT[ti, tj]
            if r < rh: sr += zbl_repulsion(r, Zi, type_Z[tj], zbl_a0, zbl_c, zbl_d)
            elif r < rh + SWITCH_DR:
                e_zbl = zbl_repulsion(r, Zi, type_Z[tj], zbl_a0, zbl_c, zbl_d)
                A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
                e_buck = 0.0
                if A > 0.0 and F > 1.0e-12: e_buck += A * exp(-r / F)
                if C > 0.0: e_buck -= C / (r ** 6)
                e_coul = wolf_coulomb(qi, charges[j], r, alpha, cutoff)
                x_switch = (r - rh) / SWITCH_DR
                s = x_switch ** 3 * (10.0 - 15.0 * x_switch + 6.0 * x_switch ** 2)
                sr += s * e_buck + (1.0 - s) * e_zbl; cl += s * e_coul
            else:
                A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
                if A > 0.0 and F > 1.0e-12: sr += A * exp(-r / F)
                if C > 0.0: sr -= C / (r ** 6)
                cl += wolf_coulomb(qi, charges[j], r, alpha, cutoff)
    return sr, cl

# ✅ SPEED OPTIMIZATION 1: Numba-compiled invalid bond check
@njit(fastmath=True, cache=True)
def check_invalid_bonds_numba(idx, new_pos, types, coords, neigh, st, box):
    ti = types[idx]
    if ti != 0 and ti != 3 and ti != 4: return False
    for p in range(st[idx], st[idx + 1]):
        j = neigh[p]
        if j == idx: continue
        tj = types[j]
        if (ti == 0 or ti == 3) and (tj == 0 or tj == 3):
            dx = new_pos[0] - coords[j, 0]; dy = new_pos[1] - coords[j, 1]; dz = new_pos[2] - coords[j, 2]
            dx -= box * round(dx / box); dy -= box * round(dy / box); dz -= box * round(dz / box)
            if dx*dx + dy*dy + dz*dz < 6.25: return True
        elif ti == 4 and tj == 4:
            dx = new_pos[0] - coords[j, 0]; dy = new_pos[1] - coords[j, 1]; dz = new_pos[2] - coords[j, 2]
            dx -= box * round(dx / box); dy -= box * round(dy / box); dz -= box * round(dz / box)
            if dx*dx + dy*dy + dz*dz < 2.56: return True
    return False

class MCSimulator:
    def __init__(self, system: GlassSystem, config: SimulationConfig):
        self.system = system
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.nl = NeighborList(system, config.cutoff, config.skin)
        self.accepted = 0; self.attempts = 0; self.rebuilds = 0
        self.tracker = {'need_rebuild': False}
        self.skin_limit = config.skin / 3.0
        self.max_disp = {'Si': 0.04, 'Ca': 0.08, 'Na': 0.08, 'P': 0.05, 'O': 0.08, 'Mg': 0.08}
        self.energy_log = []; self.short_log = []; self.coul_log = []
        self.step = 0; self.current_energy = 0.0; self.current_short = 0.0; self.current_coul = 0.0
        self.coords_ref = None
        self.swap_attempts = 0; self.swap_accepted = 0
        self.bond_break_attempts = 0; self.bond_break_accepted = 0
        
        N = system.N_ATOMS
        self.mixing_steps = max(1, int(config.mixing_sweeps * N))
        self.annealing_stages = [(T, max(1, int(s * N))) for T, s in config.annealing_sweeps]
        self.healing_steps = max(1, int(config.healing_sweeps * N))
        self.low_t_steps = max(1, int(config.low_t_sweeps * N))
        self.final_steps = max(1, int(config.final_sweeps * N))
        
        self.tune_freq = max(500, N // 10)
        self.log_freq = max(100, N // 100)
        self.decomposition_freq = max(5000, N)
        self.snapshot_interval_steps = config.snapshot_interval * N
        self.checkpoint_interval_steps = config.checkpoint_interval
        
        self._prepare_matrices()
        self._init_energy()
        self._build_cation_lists()
        
        self.start_step = 0
        if config.continue_mode: self._load_checkpoint()
        logger.info(f"MC schedule: mixing={self.mixing_steps}, annealing={sum(s for _, s in self.annealing_stages)}, production={self.final_steps}")
    
    def _build_cation_lists(self):
        self.na_indices = np.where(self.system.type_indices == 2)[0]
        self.ca_indices = np.where(self.system.type_indices == 1)[0]
        self.mg_indices = np.where(self.system.type_indices == 5)[0]
    
    def _prepare_matrices(self):
        nt = 6
        self.A_mat = np.zeros((nt, nt)); self.F_mat = np.zeros((nt, nt)); self.C_mat = np.zeros((nt, nt))
        self.R_HARD_MAT = np.full((nt, nt), self.system.potential.r_hard_default)
        for (e1, e2), val in self.system.potential.r_hard_by_pair.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.R_HARD_MAT[t1, t2] = val; self.R_HARD_MAT[t2, t1] = val
        for (e1, e2), p in self.system.potential.buck_params.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.A_mat[t1, t2] = p['A']; self.F_mat[t1, t2] = p['F']; self.C_mat[t1, t2] = p['C']
            if t1 != t2:
                self.A_mat[t2, t1] = p['A']; self.F_mat[t2, t1] = p['F']; self.C_mat[t2, t1] = p['C']
        self.type_Z = self.system.get_type_Z()
        self.wolf_self = -KE * (self.config.wolf_alpha / sqrt(pi)) * np.sum(self.system.charges ** 2)
    
    def _init_energy(self):
        n, s = self.nl.neighbors, self.nl.starts
        U = total_energy(self.system.coords, self.system.charges, self.system.type_indices, self.type_Z,
                         self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, n, s, self.system.box, 
                         self.config.cutoff, self.config.wolf_alpha, self.system.potential.zbl_a0, 
                         self.system.potential.zbl_c, self.system.potential.zbl_d)
        self.current_energy = U + self.wolf_self
        self.coords_ref = self.system.coords.copy()
        logger.info(f"Initial energy: {self.current_energy / self.system.N_ATOMS:.6f} eV/atom")
    
    def _maybe_log(self):
        if self.step % self.log_freq == 0:
            if self.step % self.decomposition_freq == 0:
                n, s = self.nl.neighbors, self.nl.starts
                sr, cl = energy_decomposition(self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, n, s, self.system.box, self.config.cutoff, self.config.wolf_alpha, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
                self.current_short = sr; self.current_coul = cl + self.wolf_self
            self.energy_log.append(self.current_energy / self.system.N_ATOMS)
            self.short_log.append(self.current_short / self.system.N_ATOMS)
            self.coul_log.append(self.current_coul / self.system.N_ATOMS)
    
    def _save_checkpoint(self):
        checkpoint_data = {'step': self.step, 'accepted': self.accepted, 'attempts': self.attempts,
                           'swap_attempts': self.swap_attempts, 'swap_accepted': self.swap_accepted,
                           'bond_break_attempts': self.bond_break_attempts, 'bond_break_accepted': self.bond_break_accepted,
                           'current_energy': self.current_energy, 'max_disp': self.max_disp, 'rebuilds': self.rebuilds}
        with open(self.config.output_dir / "checkpoint.json", 'w') as f: json.dump(checkpoint_data, f, indent=2)
        np.save(self.config.output_dir / "checkpoint_coords.npy", self.system.coords)
        np.save(self.config.output_dir / "checkpoint_types.npy", self.system.type_indices)
        np.save(self.config.output_dir / "checkpoint_charges.npy", self.system.charges)
    
    def _load_checkpoint(self):
        cp_path = self.config.output_dir / "checkpoint.json"
        if not cp_path.exists(): return
        try:
            with open(cp_path, 'r') as f: cp_data = json.load(f)
            self.start_step = cp_data['step']; self.accepted = cp_data['accepted']; self.attempts = cp_data['attempts']
            self.max_disp = cp_data['max_disp']; self.rebuilds = cp_data['rebuilds']; self.step = self.start_step
            self.system.coords = np.load(self.config.output_dir / "checkpoint_coords.npy")
            self.system.type_indices = np.load(self.config.output_dir / "checkpoint_types.npy")
            self.system.charges = np.load(self.config.output_dir / "checkpoint_charges.npy")
            self.nl = NeighborList(self.system, self.config.cutoff, self.config.skin)
            self.coords_ref = self.system.coords.copy()
            self.tracker['need_rebuild'] = False
            self.current_energy = cp_data['current_energy']
            logger.info(f"Resumed from step {self.start_step}")
        except Exception as e: logger.error(f"Checkpoint load failed: {e}"); self.start_step = 0

    # ✅ SPEED OPTIMIZATION 2: Random sampling for bond breaking O(1) instead of O(N)
    def _bond_breaking_move(self, T: float) -> bool:
        self.bond_break_attempts += 1
        neigh, st = self.nl.neighbors, self.nl.starts
        coords = self.system.coords; types = self.system.type_indices; box = self.system.box
        n = self.system.N_ATOMS
        
        for _ in range(10):
            i = self.rng.integers(0, n)
            ti = types[i]
            if ti != 0 and ti != 3 and ti != 4: continue
            for p in range(st[i], st[i + 1]):
                j = neigh[p]
                tj = types[j]
                is_invalid = False
                if (ti == 0 or ti == 3) and (tj == 0 or tj == 3):
                    d = coords[i] - coords[j]; d -= box * np.round(d / box)
                    if np.sum(d*d) < 6.25: is_invalid = True
                elif ti == 4 and tj == 4:
                    d = coords[i] - coords[j]; d -= box * np.round(d / box)
                    if np.sum(d*d) < 2.56: is_invalid = True
                    
                if is_invalid:
                    old_e_i = local_energy(i, coords, self.system.charges, types, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
                    old_e_j = local_energy(j, coords, self.system.charges, types, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
                    old_total = old_e_i + old_e_j
                    old_pos_i = coords[i].copy(); old_pos_j = coords[j].copy()
                    d = coords[i] - coords[j]; d -= box * np.round(d / box)
                    r_vec = np.linalg.norm(d)
                    d = d / r_vec if r_vec > 1e-6 else (self.rng.normal(size=3) / np.linalg.norm(self.rng.normal(size=3)))
                    push_dist = 0.3
                    coords[i] = (coords[i] + d * push_dist) % box; coords[j] = (coords[j] - d * push_dist) % box
                    new_e_i = local_energy(i, coords, self.system.charges, types, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
                    new_e_j = local_energy(j, coords, self.system.charges, types, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
                    de = (new_e_i + new_e_j) - old_total
                    if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
                        self.current_energy += de; self.bond_break_accepted += 1
                        if push_dist > self.skin_limit: self.tracker['need_rebuild'] = True
                        return True
                    else:
                        coords[i] = old_pos_i; coords[j] = old_pos_j
                        return False
        return False
    
    def _mc_move(self, T: float) -> bool:
        n = self.system.N_ATOMS
        i = int(self.rng.integers(0, n))
        old_pos = self.system.coords[i].copy()
        elem = self.system.type_to_elem[self.system.type_indices[i]]
        maxd = self.max_disp.get(elem, 0.06)
        neigh, st = self.nl.neighbors, self.nl.starts
        
        old_e = local_energy(i, self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, self.system.box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
        delta = self.rng.uniform(-maxd, maxd, size=3)
        new_pos = (old_pos + delta) % self.system.box
        
        # ✅ Use Numba-compiled function for max speed
        if check_invalid_bonds_numba(i, new_pos, self.system.type_indices, self.system.coords, neigh, st, self.system.box):
            return False
        
        self.system.coords[i] = new_pos
        new_e = local_energy(i, self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, self.system.box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
        de = new_e - old_e
        
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            self.current_energy += de
            dr = self.system.coords[i] - self.coords_ref[i]
            dr -= self.system.box * np.round(dr / self.system.box)
            if np.sqrt(np.sum(dr ** 2)) > self.skin_limit:
                self.tracker['need_rebuild'] = True; self._update_neighbors()
            return True
        else:
            self.system.coords[i] = old_pos
            return False
            
    def _swap_move(self, T: float) -> bool:
        available = []
        if self.na_indices.size > 0: available.append(self.na_indices)
        if self.ca_indices.size > 0: available.append(self.ca_indices)
        if self.mg_indices.size > 0: available.append(self.mg_indices)
        if len(available) < 2: return False
        idx1, idx2 = self.rng.choice(len(available), size=2, replace=False)
        i = available[idx1][self.rng.integers(0, available[idx1].size)]
        j = available[idx2][self.rng.integers(0, available[idx2].size)]
        self.swap_attempts += 1
        neigh, st = self.nl.neighbors, self.nl.starts
        e_i_old = local_energy(i, self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, self.system.box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
        e_j_old = local_energy(j, self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, self.system.box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
        old_total = e_i_old + e_j_old
        ti_old, tj_old = self.system.type_indices[i], self.system.type_indices[j]
        qi_old, qj_old = self.system.charges[i], self.system.charges[j]
        self.system.type_indices[i], self.system.type_indices[j] = tj_old, ti_old
        self.system.charges[i], self.system.charges[j] = qj_old, qi_old
        e_i_new = local_energy(i, self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, self.system.box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
        e_j_new = local_energy(j, self.system.coords, self.system.charges, self.system.type_indices, self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT, self.system.box, self.config.cutoff, self.config.wolf_alpha, neigh, st, self.system.potential.zbl_a0, self.system.potential.zbl_c, self.system.potential.zbl_d)
        de = (e_i_new + e_j_new) - old_total
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            self.current_energy += de; self.swap_accepted += 1; return True
        else:
            self.system.type_indices[i], self.system.type_indices[j] = ti_old, tj_old
            self.system.charges[i], self.system.charges[j] = qi_old, qj_old
            return False
    
    def _update_neighbors(self):
        if self.tracker['need_rebuild']:
            self.nl.update(force=True); self.coords_ref = self.system.coords.copy()
            self.tracker['need_rebuild'] = False; self.rebuilds += 1
            
    def _run_stage(self, T, steps, desc, adaptive=False, mixing=False, enhance_oxygen=False, enable_swap=False, enable_bond_break=False):
        logger.info(f"{desc}: T={T:.1f} K, steps={steps}")
        stage_acc = 0; stage_att = 0; window_acc = 0; window_att = 0; ema = 0.5
        orig_disp = self.max_disp.copy()
        if mixing:
            for k in self.max_disp: self.max_disp[k] *= 2.0
            self.max_disp['O'] *= 1.5
        if enhance_oxygen and T >= 1500.0: self.max_disp['O'] = min(self.max_disp['O'] * 1.3, 0.20)
            
        pbar = tqdm(range(steps), desc=f"T={T:.0f}K")
        for step_in_stage in pbar:
            if self.step < self.start_step: self.step += 1; pbar.update(1); continue
            self.step += 1; self.attempts += 1; stage_att += 1; window_att += 1
            if enable_swap and (step_in_stage % self.config.swap_move_freq == 0): self._swap_move(T)
            if enable_bond_break and (step_in_stage % self.config.bond_break_freq == 0): self._bond_breaking_move(T)
            if self._mc_move(T): self.accepted += 1; stage_acc += 1; window_acc += 1
            self._maybe_log()
            if self.step % self.checkpoint_interval_steps == 0: self._save_checkpoint()
            if adaptive and (self.step % self.tune_freq == 0):
                ca = window_acc / max(1, window_att); ema = 0.8 * ema + 0.2 * ca
                if ema < self.config.target_acceptance - 0.05:
                    for e in self.max_disp: self.max_disp[e] = max(self.max_disp[e] * 0.92, 0.02)
                elif ema > self.config.target_acceptance + 0.05:
                    for e in self.max_disp: self.max_disp[e] = min(self.max_disp[e] * 1.08, 0.25)
                window_acc = 0; window_att = 0
            pbar.set_postfix({'acc': f'{self.accepted / max(1, self.attempts) * 100:.1f}%'})
        pbar.close()
        self.max_disp = orig_disp

    def run(self):
        logger.info("=" * 70)
        logger.info(f"STARTING NVT MONTE CARLO - Mg label 45-M{self.config.x}")
        logger.info("=" * 70)
        
        if self.config.continue_mode and self.start_step > 0:
            self.final_steps = self.start_step + max(1, int(self.config.final_sweeps * self.system.N_ATOMS))
        else:
            self._run_stage(self.config.mixing_temp, self.mixing_steps, "Mixing", adaptive=True, mixing=True, enhance_oxygen=True, enable_bond_break=True)
            for i, (T, steps) in enumerate(self.annealing_stages):
                self._run_stage(T, steps, f"Annealing {i + 1}", adaptive=True, enhance_oxygen=True, enable_swap=True, enable_bond_break=(T >= 1500.0))
            for k in self.max_disp: self.max_disp[k] *= 0.7
            self._run_stage(self.config.healing_temp, self.healing_steps, "Defect Healing", adaptive=True, enhance_oxygen=True, enable_swap=True, enable_bond_break=True)
            self._run_stage(1000.0, max(1, int(3 * self.system.N_ATOMS)), "Cool", adaptive=True, enable_swap=True)
            self._run_stage(600.0, max(1, int(2 * self.system.N_ATOMS)), "Cool", adaptive=True)
            for k in self.max_disp: self.max_disp[k] *= 0.5
            self._run_stage(1.0, self.low_t_steps, "Low-T relaxation")
        
        for k in self.max_disp: self.max_disp[k] = min(max(self.max_disp[k] * 0.8, 0.02), 0.15)
        logger.info(f"Production: T=300 K, total steps={self.final_steps}")
        
        start_time = time.time()
        pbar = tqdm(range(self.final_steps), desc="T=300K")
        for s in pbar:
            if self.step < self.start_step: self.step += 1; pbar.update(1); continue
            self.step += 1; self.attempts += 1
            if s % self.config.swap_move_freq == 0: self._swap_move(300.0)
            if s % self.config.bond_break_freq == 0: self._bond_breaking_move(300.0)
            if self._mc_move(300.0): self.accepted += 1
            self._maybe_log()
            if self.step % self.checkpoint_interval_steps == 0: self._save_checkpoint()
            if s > self.start_step:
                rate = (s - self.start_step) / max(1e-6, time.time() - start_time)
                eta = (self.final_steps - s) / max(1e-6, rate)
                pbar.set_postfix({'acc': f'{self.accepted / max(1, self.attempts) * 100:.1f}%', 'eta': f"{eta/3600:.1f}h"})
        pbar.close()
        self._save_checkpoint()

def compute_block_energy(energy_log, block_size=5000):
    if len(energy_log) < block_size * 3: return {'mean': float(np.mean(energy_log)) if energy_log else 0.0, 'std': 0.0}
    prod_energy = np.array(energy_log); n_blocks = len(prod_energy) // block_size
    block_avgs = [np.mean(prod_energy[i * block_size:(i + 1) * block_size]) for i in range(n_blocks)]
    return {'mean': float(np.mean(block_avgs)), 'std': float(np.std(block_avgs, ddof=1) / np.sqrt(n_blocks))}

def main():
    parser = argparse.ArgumentParser(description="45S5/Mg Bioglass NVT Monte Carlo - v4.3 Ultimate")
    parser.add_argument('input_file', type=Path)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--density', type=float, default=None)
    parser.add_argument('--final-sweeps', type=int, default=None)
    parser.add_argument('--cutoff', type=float, default=12.0)
    parser.add_argument('--wolf-alpha', type=float, default=0.20)
    parser.add_argument('--continue', dest='continue_mode', action='store_true')
    args = parser.parse_args()
    
    try:
        n_header, comment, meta = parse_xyz_header(args.input_file)
        x = meta.get('x', 0)
        seed = args.seed if args.seed is not None else meta.get('seed', 42)
        density = args.density if args.density is not None else meta.get('density', None)
        box_override = meta.get('box', None)
        output_dir = Path(f"Bioglass_Mg{x}_N{n_header}_seed{seed}")
        
        config = SimulationConfig(input_file=args.input_file, seed=seed, density=density, x=x, box_override=box_override, output_dir=output_dir, cutoff=args.cutoff, wolf_alpha=args.wolf_alpha, continue_mode=args.continue_mode)
        if args.final_sweeps is not None: config.final_sweeps = args.final_sweeps
        
        system = GlassSystem(config)
        sim = MCSimulator(system, config)
        sim.run()
        
        # ✅ Save final structure
        xyz_path = config.output_dir / "final_structure.xyz"
        with open(xyz_path, 'w') as f:
            f.write(f"{system.N_ATOMS}\n")
            f.write(f"Final Mg-doped 45S5, label=45-M{x}, N={system.N_ATOMS}, rho={system.effective_density:.6f}, box={system.box:.6f}, seed={seed}\n")
            for i in range(system.N_ATOMS):
                elem = system.type_to_elem[system.type_indices[i]]
                f.write(f"{elem:2s} {system.coords[i,0]:12.6f} {system.coords[i,1]:12.6f} {system.coords[i,2]:12.6f}\n")
        logger.info(f"Final XYZ saved to {xyz_path}")
        
        # ✅ Save Energy Log
        energy_path = config.output_dir / "energy_log.csv"
        with open(energy_path, 'w') as f:
            f.write("Step,Total_eV_per_atom,ShortRange_eV_per_atom,Coulomb_eV_per_atom\n")
            for i, (e, s, c) in enumerate(zip(sim.energy_log, sim.short_log, sim.coul_log)):
                step = (i + 1) * sim.log_freq
                f.write(f"{step},{e:.6f},{s:.6f},{c:.6f}\n")
        logger.info(f"Energy log saved to {energy_path}")
        
        # ✅ Save Block Averaging
        block_energy = compute_block_energy(sim.energy_log, config.block_size)
        block_path = config.output_dir / "block_averaging.txt"
        with open(block_path, 'w') as f:
            f.write(f"Block size: {config.block_size}\n")
            f.write(f"Number of blocks: {len(sim.energy_log) // config.block_size}\n")
            f.write(f"Mean energy: {block_energy['mean']:.6f} eV/atom\n")
            f.write(f"Standard error: {block_energy['std']:.6f} eV/atom\n")
        logger.info(f"Block averaging saved to {block_path}")
        
        logger.info(f"[OK] All results saved to {config.output_dir}")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True); sys.exit(1)

if __name__ == "__main__":
    main()
    
    
    