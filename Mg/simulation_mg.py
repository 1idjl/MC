#!/usr/bin/env python3
"""
========================================================================
45S5/Mg Bioglass NVT Monte Carlo - OPTIMIZED VERSION v3.0
Based on: Moghanian et al., Mg-doped 45S5 bioglass manuscript

v3.0 IMPROVEMENTS:
1. Optimized annealing schedule with longer dwell at 2500-1500 K
   (critical for breaking artificial BO bonds)
2. Increased mixing sweeps (20 -> 30) for better initial homogenization
3. Increased healing sweeps (5 -> 10) for final defect removal
4. Identity Swap enabled in Production (cation equilibration)
5. Automatic checkpoint saving for resume capability
6. Snapshot saving for trajectory analysis

Usage:
  python simulation_mg.py initial_Mg0.xyz --final-sweeps 20000
  python simulation_mg.py initial_Mg0.xyz --continue  # Resume from checkpoint
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

# ========================= UTF-8 LOGGING =========================
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        if hasattr(stream, 'buffer'):
            stream = codecs.getwriter('utf-8')(stream.buffer)
        super().__init__(stream)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log', 'w', 'utf-8'),
        UTF8StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================= CONSTANTS =========================
KB_EV = 8.617333262145e-5
NA = 6.02214076e23
KE = 14.3996454784255
SKIN = 1.5
SWITCH_DR = 0.3
BASE_N_ATOMS = 2835

class SimulationError(Exception):
    pass

class FileError(SimulationError):
    pass

def parse_xyz_header(path: Path):
    if not path.exists():
        raise FileError(f"File not found: {path}")
    with open(path, 'r') as f:
        first = f.readline().strip()
        second = f.readline().strip()
    try:
        n_atoms = int(first)
    except Exception as exc:
        raise FileError(f"Cannot parse atom count: {first}") from exc
    meta = {'x': 0, 'N': n_atoms, 'density': None, 'box': None, 'seed': None}
    patterns = [
        ('x', r"x\s*=\s*(\d+)", int),
        ('N', r"N\s*=\s*(\d+)", int),
        ('density', r"rho\s*=\s*([0-9]*\.?[0-9]+)", float),
        ('box', r"box\s*=\s*([0-9]*\.?[0-9]+)", float),
        ('seed', r"seed\s*=\s*(\d+)", int),
    ]
    for key, pat, conv in patterns:
        m = re.search(pat, second)
        if m:
            meta[key] = conv(m.group(1))
    return n_atoms, second, meta

# ========================= CONFIG =========================
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
    mixing_sweeps: int = 30  # ✅ Increased from 20 for better initial homogenization
    
    # ✅ OPTIMIZED ANNEALING SCHEDULE:
    # Longer dwell at 2500-1500 K to break artificial BO bonds
    annealing_sweeps: List[Tuple[float, int]] = field(default_factory=lambda: [
        (4500, 6),
        (4000, 6),
        (3500, 8),
        (3000, 12),   # Longer at high T
        (2500, 20),   # ⭐ Critical temperature for topology rearrangement
        (2000, 20),   # ⭐ Continue rearrangement
        (1500, 15),   # ⭐ Structure stabilization
        (1200, 10),
        (1000, 10),
        (800, 8),
        (700, 8),
        (600, 6),
        (500, 5),
        (400, 5),
        (300, 5),
    ])
    
    healing_temp: float = 2500.0
    healing_sweeps: int = 10  # ✅ Increased from 5 for final defect removal
    low_t_sweeps: int = 8
    
    final_sweeps: int = 20000
    snapshot_interval: int = 5000  # Save snapshot every 5000 sweeps
    max_snapshots: int = 50
    
    cutoff: float = 12.0
    wolf_alpha: float = 0.20
    
    skin: float = SKIN
    target_acceptance: float = 0.40
    adaptive_tuning_freq: Optional[int] = None
    log_freq: Optional[int] = None
    decomposition_freq: Optional[int] = None
    
    swap_move_freq: int = 100
    
    block_size: int = 5000
    checkpoint_interval: int = 10000  # Save checkpoint every 10000 steps
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path(f"Bioglass_Mg{self.x}_seed{self.seed}")
        else:
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

# ========================= POTENTIAL =========================
class Potential:
    def __init__(self):
        # ZBL repulsion parameters
        self.zbl_a0 = 0.46850
        self.zbl_c = np.array([0.1818, 0.5099, 0.2802, 0.02817])
        self.zbl_d = np.array([3.2, 0.9423, 0.4029, 0.2016])
        
        # Atomic numbers for ZBL
        self.zbl_z = {
            'Si': 14.0, 'Ca': 20.0, 'Na': 11.0,
            'P': 15.0, 'O': 8.0, 'Mg': 12.0,
        }
        
        # Buckingham parameters from Moghanian et al. Table 1
        self.buck_params = {
            ('Si', 'O'): {'A': 13702.905, 'F': 0.193817, 'C': 54.681},
            ('Ca', 'O'): {'A': 7747.1834, 'F': 0.252623, 'C': 93.109},
            ('P',  'O'): {'A': 26655.472, 'F': 0.181968, 'C': 86.856},
            ('Na', 'O'): {'A': 4383.7555, 'F': 0.243838, 'C': 30.70},
            ('Mg', 'O'): {'A': 7063.0,    'F': 0.2109,   'C': 19.21},
            ('O',  'O'): {'A': 2029.2204, 'F': 0.343645, 'C': 192.58},
        }
        
        # Hard-core switching distances
        self.r_hard_default = 0.9
        self.r_hard_by_pair = {
            ('O', 'O'): 1.4,
            ('Mg', 'O'): 1.0,
        }

# ========================= GLASS SYSTEM =========================
class GlassSystem:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.masses = {
            'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
            'P': 30.97376, 'O': 15.999, 'Mg': 24.305,
        }
        self.type_map = {
            'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Mg': 5,
        }
        self.type_to_elem = {
            0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Mg',
        }
        self.potential = Potential()
        
        if not config.input_file.exists():
            raise FileError(f"File not found: {config.input_file}")
        
        self._load_xyz()
        self._setup_box()
        self._compute_charges()
    
    def _load_xyz(self):
        with open(self.config.input_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            raise FileError("XYZ file too short.")
        
        n_header = int(lines[0].strip())
        self.symbols = []
        coords_list = []
        
        for line in lines[2:2 + n_header]:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            elem = parts[0]
            if elem not in self.type_map:
                raise FileError(f"Unknown element in XYZ file: {elem}")
            self.symbols.append(elem)
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        self.N_ATOMS = len(self.symbols)
        if self.N_ATOMS != n_header:
            logger.warning(f"XYZ header says {n_header}, but {self.N_ATOMS} lines read.")
        
        if self.N_ATOMS % BASE_N_ATOMS != 0:
            logger.warning(f"Total atoms {self.N_ATOMS} is not a multiple of {BASE_N_ATOMS}.")
        
        logger.info(f"Loaded {self.N_ATOMS} atoms from {self.config.input_file}")
        
        self.coords = np.array(coords_list, dtype=np.float64)
        self.type_indices = np.array(
            [self.type_map[s] for s in self.symbols],
            dtype=np.int32
        )
        self.counts = {
            e: int(np.sum(self.type_indices == self.type_map[e]))
            for e in set(self.symbols)
        }
        logger.info(f"Composition: {self.counts}")
        self.total_mass = sum(self.masses[s] for s in self.symbols)
    
    def _setup_box(self):
        if self.config.box_override is not None:
            self.box = float(self.config.box_override)
            self.effective_density = self.total_mass * 1.0e24 / (NA * self.box ** 3)
            logger.info(
                f"Using box from XYZ: {self.box:.6f} A "
                f"(density={self.effective_density:.6f} g/cm3)"
            )
        elif self.config.density is not None:
            self.effective_density = float(self.config.density)
            volume_A3 = (self.total_mass / NA) / self.effective_density * 1.0e24
            self.box = volume_A3 ** (1.0 / 3.0)
            logger.info(
                f"Using density {self.effective_density:.6f} -> box={self.box:.6f} A"
            )
        else:
            raise FileError("Neither box nor density could be determined.")
        
        self.coords = self.coords % self.box
    
    def _compute_charges(self):
        base = {
            'Si': 2.4, 'Ca': 1.2, 'Na': 0.6,
            'P': 3.0, 'O': -1.2, 'Mg': 1.2,
        }
        self.charges = np.array(
            [base.get(s, 0.0) for s in self.symbols],
            dtype=np.float64
        )
        
        total_charge = float(np.sum(self.charges))
        o_mask = self.type_indices == self.type_map['O']
        n_o = int(np.sum(o_mask))
        
        if abs(total_charge) > 1.0e-8 and n_o > 0:
            self.charges[o_mask] -= total_charge / n_o
            logger.warning(
                f"Initial charges were not neutral (total={total_charge:.6e}). "
                f"O charge adjusted for neutrality."
            )
        else:
            logger.info("Paper fixed charges are charge neutral.")
        
        if n_o > 0:
            logger.info(f"O charge set to {self.charges[o_mask][0]:.6f}")
    
    def get_type_Z(self):
        return np.array(
            [self.potential.zbl_z[self.type_to_elem[i]] for i in range(6)],
            dtype=np.float64
        )

# ========================= NEIGHBOR LIST =========================
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
        
        for i, j in pairs:
            counts[i] += 1
            counts[j] += 1
        
        starts = np.zeros(n + 1, dtype=np.int32)
        starts[1:] = np.cumsum(counts)
        neighbors = np.empty(starts[-1], dtype=np.int32)
        fill = starts[:-1].copy()
        
        for i, j in pairs:
            neighbors[fill[i]] = j
            fill[i] += 1
            neighbors[fill[j]] = i
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
        if self.ref_coords is None:
            return True
        dr = self.system.coords - self.ref_coords
        dr -= self.system.box * np.round(dr / self.system.box)
        return np.sqrt(np.sum(dr ** 2, axis=1)).max() > self.skin / 3.0

# ========================= NUMBA ENERGY ROUTINES =========================
@njit(fastmath=True, cache=True)
def zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d):
    if r < 1.0e-6:
        return 1.0e8
    a = zbl_a0 / (Zi ** 0.23 + Zj ** 0.23)
    x = r / a
    phi = (
        zbl_c[0] * exp(-zbl_d[0] * x) +
        zbl_c[1] * exp(-zbl_d[1] * x) +
        zbl_c[2] * exp(-zbl_d[2] * x) +
        zbl_c[3] * exp(-zbl_d[3] * x)
    )
    return KE * Zi * Zj * phi / r

@njit(fastmath=True, cache=True)
def wolf_coulomb(qi, qj, r, alpha, cutoff):
    if r >= cutoff or r < 1.0e-12:
        return 0.0
    ar = alpha * r
    ac = alpha * cutoff
    er = erfc(ar)
    ec = erfc(ac)
    t1 = er / r
    t2 = ec / cutoff
    t3 = (
        (ec / (cutoff * cutoff)) +
        (2.0 * alpha / sqrt(pi)) * exp(-ac * ac) / cutoff
    ) * (r - cutoff)
    return KE * qi * qj * (t1 - t2 + t3)

@njit(fastmath=True, cache=True)
def pair_energy(
    r, qi, qj, ti, tj, Zi, Zj,
    A_mat, F_mat, C_mat, R_HARD_MAT,
    cutoff, alpha, zbl_a0, zbl_c, zbl_d
):
    if r >= cutoff or r < 1.0e-12:
        return 0.0
    
    r_in = R_HARD_MAT[ti, tj]
    r_out = r_in + SWITCH_DR
    
    if r < r_out:
        e_zbl = zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d)
        if r < r_in:
            return e_zbl
        
        A = A_mat[ti, tj]
        F = F_mat[ti, tj]
        C = C_mat[ti, tj]
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12:
            e_buck += A * exp(-r / F)
        if C > 0.0:
            e_buck -= C / (r ** 6)
        e_coul = wolf_coulomb(qi, qj, r, alpha, cutoff)
        e_full = e_buck + e_coul
        
        x_switch = (r - r_in) / SWITCH_DR
        s = x_switch ** 3 * (10.0 - 15.0 * x_switch + 6.0 * x_switch ** 2)
        return s * e_full + (1.0 - s) * e_zbl
    else:
        A = A_mat[ti, tj]
        F = F_mat[ti, tj]
        C = C_mat[ti, tj]
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12:
            e_buck += A * exp(-r / F)
        if C > 0.0:
            e_buck -= C / (r ** 6)
        return e_buck + wolf_coulomb(qi, qj, r, alpha, cutoff)

@njit(fastmath=True, cache=True)
def local_energy(
    idx, coords, charges, types, type_Z,
    A_mat, F_mat, C_mat, R_HARD_MAT,
    box, cutoff, alpha, neighbors, starts,
    zbl_a0, zbl_c, zbl_d
):
    e = 0.0
    qi = charges[idx]
    ti = types[idx]
    Zi = type_Z[ti]
    xi, yi, zi = coords[idx]
    
    for p in range(starts[idx], starts[idx + 1]):
        j = neighbors[p]
        dx = xi - coords[j, 0]
        dy = yi - coords[j, 1]
        dz = zi - coords[j, 2]
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        r2 = dx * dx + dy * dy + dz * dz
        
        if r2 >= cutoff * cutoff:
            continue
        
        r = sqrt(r2)
        e += pair_energy(
            r, qi, charges[j], ti, types[j], Zi, type_Z[types[j]],
            A_mat, F_mat, C_mat, R_HARD_MAT,
            cutoff, alpha, zbl_a0, zbl_c, zbl_d
        )
    
    e += -KE * (alpha / sqrt(pi)) * qi * qi
    return e

@njit(fastmath=True, cache=True)
def total_energy(
    coords, charges, types, type_Z,
    A_mat, F_mat, C_mat, R_HARD_MAT,
    neighbors, starts, box, cutoff, alpha,
    zbl_a0, zbl_c, zbl_d
):
    e = 0.0
    n = coords.shape[0]
    
    for i in range(n):
        xi, yi, zi = coords[i]
        qi = charges[i]
        ti = types[i]
        Zi = type_Z[ti]
        
        for p in range(starts[i], starts[i + 1]):
            j = neighbors[p]
            if j <= i:
                continue
            
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            dx -= box * round(dx / box)
            dy -= box * round(dy / box)
            dz -= box * round(dz / box)
            r2 = dx * dx + dy * dy + dz * dz
            
            if r2 >= cutoff * cutoff:
                continue
            
            r = sqrt(r2)
            e += pair_energy(
                r, qi, charges[j], ti, types[j], Zi, type_Z[types[j]],
                A_mat, F_mat, C_mat, R_HARD_MAT,
                cutoff, alpha, zbl_a0, zbl_c, zbl_d
            )
    
    return e

@njit(fastmath=True, cache=True)
def energy_decomposition(
    coords, charges, types, type_Z,
    A_mat, F_mat, C_mat, R_HARD_MAT,
    neighbors, starts, box, cutoff, alpha,
    zbl_a0, zbl_c, zbl_d
):
    sr = 0.0
    cl = 0.0
    n = coords.shape[0]
    
    for i in range(n):
        xi, yi, zi = coords[i]
        qi = charges[i]
        ti = types[i]
        Zi = type_Z[ti]
        
        for p in range(starts[i], starts[i + 1]):
            j = neighbors[p]
            if j <= i:
                continue
            
            dx = xi - coords[j, 0]
            dy = yi - coords[j, 1]
            dz = zi - coords[j, 2]
            dx -= box * round(dx / box)
            dy -= box * round(dy / box)
            dz -= box * round(dz / box)
            r2 = dx * dx + dy * dy + dz * dz
            
            if r2 >= cutoff * cutoff:
                continue
            
            r = sqrt(r2)
            tj = types[j]
            rh = R_HARD_MAT[ti, tj]
            
            if r < rh:
                sr += zbl_repulsion(r, Zi, type_Z[tj], zbl_a0, zbl_c, zbl_d)
            elif r < rh + SWITCH_DR:
                e_zbl = zbl_repulsion(r, Zi, type_Z[tj], zbl_a0, zbl_c, zbl_d)
                A = A_mat[ti, tj]
                F = F_mat[ti, tj]
                C = C_mat[ti, tj]
                e_buck = 0.0
                if A > 0.0 and F > 1.0e-12:
                    e_buck += A * exp(-r / F)
                if C > 0.0:
                    e_buck -= C / (r ** 6)
                e_coul = wolf_coulomb(qi, charges[j], r, alpha, cutoff)
                x_switch = (r - rh) / SWITCH_DR
                s = x_switch ** 3 * (10.0 - 15.0 * x_switch + 6.0 * x_switch ** 2)
                sr += s * e_buck + (1.0 - s) * e_zbl
                cl += s * e_coul
            else:
                A = A_mat[ti, tj]
                F = F_mat[ti, tj]
                C = C_mat[ti, tj]
                if A > 0.0 and F > 1.0e-12:
                    sr += A * exp(-r / F)
                if C > 0.0:
                    sr -= C / (r ** 6)
                cl += wolf_coulomb(qi, charges[j], r, alpha, cutoff)
    
    return sr, cl

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
        self.tracker = {'need_rebuild': False}
        
        self.max_disp = {
            'Si': 0.04, 'Ca': 0.08, 'Na': 0.08,
            'P': 0.05, 'O': 0.08, 'Mg': 0.08,
        }
        
        self.energy_log = []
        self.short_log = []
        self.coul_log = []
        self.step = 0
        
        self.current_energy = 0.0
        self.current_short = 0.0
        self.current_coul = 0.0
        self.coords_ref = None
        
        self.swap_attempts = 0
        self.swap_accepted = 0
        
        N = system.N_ATOMS
        self.mixing_steps = max(1, int(config.mixing_sweeps * N))
        self.annealing_stages = [
            (T, max(1, int(s * N))) for T, s in config.annealing_sweeps
        ]
        self.healing_steps = max(1, int(config.healing_sweeps * N))
        self.low_t_steps = max(1, int(config.low_t_sweeps * N))
        self.final_steps = max(1, int(config.final_sweeps * N))
        
        self.tune_freq = (
            config.adaptive_tuning_freq if config.adaptive_tuning_freq is not None
            else max(500, N // 10)
        )
        self.log_freq = (
            config.log_freq if config.log_freq is not None
            else max(100, N // 100)
        )
        self.decomposition_freq = (
            config.decomposition_freq if config.decomposition_freq is not None
            else max(5000, N)
        )
        
        self.snapshot_interval_steps = config.snapshot_interval * N
        self.checkpoint_interval_steps = config.checkpoint_interval
        
        self._prepare_matrices()
        self._init_energy()
        self._build_cation_lists()
        
        # Load checkpoint if continuing
        self.start_step = 0
        if config.continue_mode:
            self._load_checkpoint()
        
        total_annealing = sum(s for _, s in self.annealing_stages)
        logger.info(
            f"MC schedule: mixing={self.mixing_steps}, "
            f"annealing={total_annealing}, "
            f"healing={self.healing_steps}, low-T={self.low_t_steps}, "
            f"production={self.final_steps}"
        )
        logger.info(f"Identity Swap frequency: every {config.swap_move_freq} steps")
        logger.info(f"Snapshot interval: every {config.snapshot_interval} sweeps")
        logger.info(f"Checkpoint interval: every {config.checkpoint_interval} steps")
    
    def _build_cation_lists(self):
        self.na_indices = np.where(
            self.system.type_indices == self.system.type_map['Na']
        )[0]
        self.ca_indices = np.where(
            self.system.type_indices == self.system.type_map['Ca']
        )[0]
        self.mg_indices = np.where(
            self.system.type_indices == self.system.type_map['Mg']
        )[0]
    
    def _prepare_matrices(self):
        nt = 6
        self.A_mat = np.zeros((nt, nt), dtype=np.float64)
        self.F_mat = np.zeros((nt, nt), dtype=np.float64)
        self.C_mat = np.zeros((nt, nt), dtype=np.float64)
        self.R_HARD_MAT = np.full(
            (nt, nt),
            self.system.potential.r_hard_default,
            dtype=np.float64
        )
        
        for (e1, e2), val in self.system.potential.r_hard_by_pair.items():
            t1 = self.system.type_map[e1]
            t2 = self.system.type_map[e2]
            self.R_HARD_MAT[t1, t2] = val
            self.R_HARD_MAT[t2, t1] = val
        
        pot = self.system.potential
        for (e1, e2), p in pot.buck_params.items():
            t1 = self.system.type_map[e1]
            t2 = self.system.type_map[e2]
            self.A_mat[t1, t2] = p['A']
            self.F_mat[t1, t2] = p['F']
            self.C_mat[t1, t2] = p['C']
            if t1 != t2:
                self.A_mat[t2, t1] = p['A']
                self.F_mat[t2, t1] = p['F']
                self.C_mat[t2, t1] = p['C']
        
        self.type_Z = self.system.get_type_Z()
        self.wolf_self = (
            -KE * (self.config.wolf_alpha / sqrt(pi)) *
            np.sum(self.system.charges ** 2)
        )
    
    def _init_energy(self):
        n, s = self.nl.neighbors, self.nl.starts
        U = total_energy(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, self.system.box, self.config.cutoff, self.config.wolf_alpha,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        self.current_energy = U + self.wolf_self
        
        sr, cl = energy_decomposition(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, self.system.box, self.config.cutoff,
            self.config.wolf_alpha,
            self.system.potential.zbl_a0,
            self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        self.current_short = sr
        self.current_coul = cl + self.wolf_self
        self.coords_ref = self.system.coords.copy()
        
        logger.info(
            f"Initial energy: {self.current_energy / self.system.N_ATOMS:.6f} eV/atom"
        )
    
    def _maybe_log(self):
        if self.step % self.log_freq == 0:
            if self.step % self.decomposition_freq == 0:
                self._update_energy_decomp()
            self.energy_log.append(self.current_energy / self.system.N_ATOMS)
            self.short_log.append(self.current_short / self.system.N_ATOMS)
            self.coul_log.append(self.current_coul / self.system.N_ATOMS)
    
    def _save_snapshot(self, sweep_num):
        """Save structure snapshot."""
        snapshot_dir = self.config.output_dir / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        snapshot_path = snapshot_dir / f"snapshot_{sweep_num:06d}.xyz"
        with open(snapshot_path, 'w') as f:
            f.write(f"{self.system.N_ATOMS}\n")
            f.write(
                f"Snapshot at sweep {sweep_num}, "
                f"step={self.step}, energy={self.current_energy/self.system.N_ATOMS:.6f} eV/atom\n"
            )
            for i in range(self.system.N_ATOMS):
                elem = self.system.type_to_elem[self.system.type_indices[i]]
                x_, y_, z_ = self.system.coords[i]
                f.write(f"{elem:2s} {x_:12.6f} {y_:12.6f} {z_:12.6f}\n")
        
        logger.info(f"Snapshot saved: {snapshot_path.name}")
    
    def _save_checkpoint(self):
        """Save checkpoint for restart capability."""
        checkpoint_path = self.config.output_dir / "checkpoint.json"
        
        checkpoint_data = {
            'step': self.step,
            'accepted': self.accepted,
            'attempts': self.attempts,
            'swap_attempts': self.swap_attempts,
            'swap_accepted': self.swap_accepted,
            'current_energy': self.current_energy,
            'current_short': self.current_short,
            'current_coul': self.current_coul,
            'max_disp': self.max_disp,
            'rebuilds': self.rebuilds,
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        coords_path = self.config.output_dir / "checkpoint_coords.npy"
        np.save(coords_path, self.system.coords)
        
        types_path = self.config.output_dir / "checkpoint_types.npy"
        np.save(types_path, self.system.type_indices)
        
        charges_path = self.config.output_dir / "checkpoint_charges.npy"
        np.save(charges_path, self.system.charges)
    
    def _load_checkpoint(self):
        """Load checkpoint to resume simulation."""
        checkpoint_path = self.config.output_dir / "checkpoint.json"
        coords_path = self.config.output_dir / "checkpoint_coords.npy"
        types_path = self.config.output_dir / "checkpoint_types.npy"
        charges_path = self.config.output_dir / "checkpoint_charges.npy"
        
        if not checkpoint_path.exists():
            logger.warning("No checkpoint found, starting from beginning")
            return
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.start_step = checkpoint_data['step']
            self.accepted = checkpoint_data['accepted']
            self.attempts = checkpoint_data['attempts']
            self.swap_attempts = checkpoint_data['swap_attempts']
            self.swap_accepted = checkpoint_data['swap_accepted']
            self.current_energy = checkpoint_data['current_energy']
            self.current_short = checkpoint_data['current_short']
            self.current_coul = checkpoint_data['current_coul']
            self.max_disp = checkpoint_data['max_disp']
            self.rebuilds = checkpoint_data['rebuilds']
            self.step = self.start_step
            
            self.system.coords = np.load(coords_path)
            self.system.type_indices = np.load(types_path)
            self.system.charges = np.load(charges_path)
            
            logger.info(f"Resumed from checkpoint at step {self.start_step}")
            logger.info(f"Current energy: {self.current_energy / self.system.N_ATOMS:.6f} eV/atom")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Starting from beginning")
            self.start_step = 0
    
    def _mc_move(self, T: float) -> bool:
        n = self.system.N_ATOMS
        i = int(self.rng.integers(0, n))
        old_pos = self.system.coords[i].copy()
        elem = self.system.type_to_elem[self.system.type_indices[i]]
        maxd = self.max_disp.get(elem, 0.06)
        neigh, st = self.nl.neighbors, self.nl.starts
        
        old_e = local_energy(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.config.cutoff, self.config.wolf_alpha,
            neigh, st,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        
        delta = self.rng.uniform(-maxd, maxd, size=3)
        self.system.coords[i] = (old_pos + delta) % self.system.box
        
        new_e = local_energy(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.config.cutoff, self.config.wolf_alpha,
            neigh, st,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        
        de = new_e - old_e
        accepted = False
        
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            accepted = True
            self.current_energy += de
        else:
            self.system.coords[i] = old_pos
        
        if accepted:
            dr = self.system.coords[i] - self.coords_ref[i]
            dr -= self.system.box * np.round(dr / self.system.box)
            if np.sqrt(np.sum(dr ** 2)) > self.config.skin / 3.0:
                self.tracker['need_rebuild'] = True
                self._update_neighbors()
        
        return accepted
    
    def _swap_move(self, T: float) -> bool:
        """Chemical Identity Swap for cation equilibration."""
        available = []
        if self.na_indices.size > 0:
            available.append(('Na', self.na_indices))
        if self.ca_indices.size > 0:
            available.append(('Ca', self.ca_indices))
        if self.mg_indices.size > 0:
            available.append(('Mg', self.mg_indices))
        
        if len(available) < 2:
            return False
        
        idx1, idx2 = self.rng.choice(len(available), size=2, replace=False)
        name1, indices1 = available[idx1]
        name2, indices2 = available[idx2]
        
        i = indices1[self.rng.integers(0, indices1.size)]
        j = indices2[self.rng.integers(0, indices2.size)]
        
        self.swap_attempts += 1
        neigh, st = self.nl.neighbors, self.nl.starts
        
        e_i_old = local_energy(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.config.cutoff, self.config.wolf_alpha,
            neigh, st,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        
        e_j_old = local_energy(
            j, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.config.cutoff, self.config.wolf_alpha,
            neigh, st,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        
        old_total = e_i_old + e_j_old
        
        ti_old = self.system.type_indices[i]
        tj_old = self.system.type_indices[j]
        qi_old = self.system.charges[i]
        qj_old = self.system.charges[j]
        
        self.system.type_indices[i] = tj_old
        self.system.type_indices[j] = ti_old
        self.system.charges[i] = qj_old
        self.system.charges[j] = qi_old
        
        e_i_new = local_energy(
            i, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.config.cutoff, self.config.wolf_alpha,
            neigh, st,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        
        e_j_new = local_energy(
            j, self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            self.system.box, self.config.cutoff, self.config.wolf_alpha,
            neigh, st,
            self.system.potential.zbl_a0, self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        
        new_total = e_i_new + e_j_new
        de = new_total - old_total
        
        accepted = False
        if de <= 0.0 or self.rng.random() < exp(-de / (KB_EV * T)):
            accepted = True
            self.current_energy += de
            self.swap_accepted += 1
        else:
            self.system.type_indices[i] = ti_old
            self.system.type_indices[j] = tj_old
            self.system.charges[i] = qi_old
            self.system.charges[j] = qj_old
        
        return accepted
    
    def _update_neighbors(self):
        if self.tracker['need_rebuild']:
            self.nl.update(force=True)
            self.coords_ref = self.system.coords.copy()
            self.tracker['need_rebuild'] = False
            self.rebuilds += 1
    
    def _update_energy_decomp(self):
        n, s = self.nl.neighbors, self.nl.starts
        sr, cl = energy_decomposition(
            self.system.coords, self.system.charges,
            self.system.type_indices, self.type_Z,
            self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
            n, s, self.system.box, self.config.cutoff,
            self.config.wolf_alpha,
            self.system.potential.zbl_a0,
            self.system.potential.zbl_c,
            self.system.potential.zbl_d
        )
        self.current_short = sr
        self.current_coul = cl + self.wolf_self
    
    def _run_stage(
        self, T, steps, desc,
        adaptive=False, mixing=False,
        enhance_oxygen=False, enable_swap=False
    ):
        logger.info(f"{desc}: T={T:.1f} K, steps={steps}")
        stage_acc = 0
        stage_att = 0
        window_acc = 0
        window_att = 0
        ema = 0.5
        orig_disp = self.max_disp.copy()
        
        if mixing:
            for k in self.max_disp:
                self.max_disp[k] *= 2.0
            self.max_disp['O'] *= 1.5
        
        if enhance_oxygen and T >= 1500.0:
            self.max_disp['O'] = min(self.max_disp['O'] * 1.3, 0.20)
        
        pbar = tqdm(range(steps), desc=f"T={T:.0f}K")
        
        for step_in_stage in pbar:
            if self.step < self.start_step:
                self.step += 1
                pbar.update(1)
                continue
            
            self.step += 1
            self.attempts += 1
            stage_att += 1
            window_att += 1
            
            if enable_swap and (step_in_stage % self.config.swap_move_freq == 0):
                self._swap_move(T)
            
            acc = self._mc_move(T)
            if acc:
                self.accepted += 1
                stage_acc += 1
                window_acc += 1
            
            self._maybe_log()
            
            if self.step % self.checkpoint_interval_steps == 0:
                self._save_checkpoint()
            
            if self.step % self.snapshot_interval_steps == 0:
                sweep_num = self.step // self.system.N_ATOMS
                self._save_snapshot(sweep_num)
            
            if adaptive and (self.step % self.tune_freq == 0):
                ca = window_acc / max(1, window_att)
                ema = 0.8 * ema + 0.2 * ca
                target = self.config.target_acceptance
                factor = 1.0
                
                if ema < target - 0.05:
                    factor = 0.92
                elif ema > target + 0.05:
                    factor = 1.08
                
                if factor != 1.0:
                    for e in self.max_disp:
                        self.max_disp[e] = min(
                            max(self.max_disp[e] * factor, 0.02),
                            0.25
                        )
                
                window_acc = 0
                window_att = 0
            
            pbar.set_postfix(
                {'acc': f'{self.accepted / max(1, self.attempts) * 100:.1f}%'}
            )
        
        pbar.close()
        self.max_disp = orig_disp
        logger.info(
            f"  Stage acceptance: {stage_acc / max(1, stage_att) * 100:.2f}%"
        )
    
    def run(self):
        logger.info("=" * 70)
        logger.info(f"STARTING NVT MONTE CARLO - Mg label 45-M{self.config.x}")
        if self.config.continue_mode:
            logger.info("MODE: CONTINUE (Resuming from checkpoint)")
        logger.info("=" * 70)
        
        if self.config.continue_mode and self.start_step > 0:
            logger.info(f"Resuming from step {self.start_step}, skipping to production")
        else:
            # Full protocol
            self._run_stage(
                self.config.mixing_temp,
                self.mixing_steps,
                "Mixing",
                adaptive=True,
                mixing=True,
                enhance_oxygen=True
            )
            
            for i, (T, steps) in enumerate(self.annealing_stages):
                self._run_stage(
                    T, steps, f"Annealing {i + 1}",
                    adaptive=True,
                    enhance_oxygen=True,
                    enable_swap=True
                )
            
            for k in self.max_disp:
                self.max_disp[k] *= 0.7
            
            self._run_stage(
                self.config.healing_temp,
                self.healing_steps,
                "Defect Healing",
                adaptive=True,
                enhance_oxygen=True,
                enable_swap=True
            )
            
            self._run_stage(
                1000.0,
                max(1, int(3 * self.system.N_ATOMS)),
                "Cool after healing",
                adaptive=True,
                enable_swap=True
            )
            
            self._run_stage(
                600.0,
                max(1, int(2 * self.system.N_ATOMS)),
                "Cool after healing",
                adaptive=True
            )
            
            for k in self.max_disp:
                self.max_disp[k] *= 0.5
            
            self._run_stage(1.0, self.low_t_steps, "Low-T relaxation")
        
        # PRODUCTION
        for k in self.max_disp:
            self.max_disp[k] = min(max(self.max_disp[k] * 0.8, 0.02), 0.15)
        
        logger.info(f"Production: T=300 K, steps={self.final_steps}")
        logger.info("Identity Swap ENABLED in production for cation equilibration")
        
        start_time = time.time()
        pbar = tqdm(range(self.final_steps), desc="T=300K")
        
        for s in pbar:
            if self.step < self.start_step:
                self.step += 1
                pbar.update(1)
                continue
            
            self.step += 1
            self.attempts += 1
            
            if s % self.config.swap_move_freq == 0:
                self._swap_move(300.0)
            
            if self._mc_move(300.0):
                self.accepted += 1
            
            self._maybe_log()
            
            if self.step % self.checkpoint_interval_steps == 0:
                self._save_checkpoint()
            
            if self.step % self.snapshot_interval_steps == 0:
                sweep_num = self.step // self.system.N_ATOMS
                self._save_snapshot(sweep_num)
            
            elapsed = time.time() - start_time
            if s > 0:
                rate = s / elapsed
                eta = (self.final_steps - s) / rate
                eta_str = f"ETA: {eta/3600:.1f}h"
            else:
                eta_str = ""
            
            pbar.set_postfix({
                'acc': f'{self.accepted / max(1, self.attempts) * 100:.1f}%',
                'swap': f'{self.swap_accepted}/{self.swap_attempts}',
                'eta': eta_str
            })
        
        pbar.close()
        
        self._save_checkpoint()
        
        logger.info("=" * 70)
        logger.info("SIMULATION COMPLETED")
        logger.info("=" * 70)
        logger.info(
            f"Total attempts: {self.attempts}, "
            f"acceptance: {self.accepted / max(1, self.attempts) * 100:.2f}%, "
            f"rebuilds: {self.rebuilds}"
        )
        logger.info(
            f"Swap moves: attempts={self.swap_attempts}, "
            f"accepted={self.swap_accepted} "
            f"({self.swap_accepted / max(1, self.swap_attempts) * 100:.2f}%)"
        )

# ========================= BLOCK AVERAGING =========================
def compute_block_energy(energy_log, block_size=5000):
    if len(energy_log) < block_size * 3:
        return {
            'mean': float(np.mean(energy_log)) if energy_log else 0.0,
            'std': 0.0,
        }
    
    prod_energy = np.array(energy_log)
    n_blocks = len(prod_energy) // block_size
    
    if n_blocks < 3:
        return {
            'mean': float(np.mean(prod_energy)),
            'std': 0.0,
        }
    
    block_avgs = [
        np.mean(prod_energy[i * block_size:(i + 1) * block_size])
        for i in range(n_blocks)
    ]
    
    mean_e = float(np.mean(block_avgs))
    std_e = float(np.std(block_avgs, ddof=1) / np.sqrt(n_blocks))
    
    return {
        'mean': mean_e,
        'std': std_e,
    }

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser(
        description="45S5/Mg Bioglass NVT Monte Carlo - OPTIMIZED VERSION v3.0"
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input XYZ file generated by structure_generator.py'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed. If omitted, read from XYZ comment.'
    )
    parser.add_argument(
        '--density',
        type=float,
        default=None,
        help='Optional density override in g/cm3.'
    )
    parser.add_argument(
        '--final-sweeps',
        type=int,
        default=None,
        help='Override production MC sweeps.'
    )
    parser.add_argument(
        '--cutoff',
        type=float,
        default=12.0,
        help='Effective cutoff in Angstrom.'
    )
    parser.add_argument(
        '--wolf-alpha',
        type=float,
        default=0.20,
        help='Wolf damping parameter alpha.'
    )
    parser.add_argument(
        '--continue',
        dest='continue_mode',
        action='store_true',
        help='Resume from checkpoint if available.'
    )
    parser.add_argument(
        '--snapshot-interval',
        type=int,
        default=5000,
        help='Save snapshot every N sweeps (default: 5000)'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=10000,
        help='Save checkpoint every N steps (default: 10000)'
    )
    
    args = parser.parse_args()
    
    try:
        n_header, comment, meta = parse_xyz_header(args.input_file)
        x = meta.get('x', 0)
        seed = args.seed if args.seed is not None else meta.get('seed', 42)
        if seed is None:
            seed = 42
        density = args.density if args.density is not None else meta.get('density', None)
        box_override = meta.get('box', None)
        output_dir = Path(f"Bioglass_Mg{x}_N{n_header}_seed{seed}")
        
        config = SimulationConfig(
            input_file=args.input_file,
            seed=seed,
            density=density,
            x=x,
            box_override=box_override,
            output_dir=output_dir,
            cutoff=args.cutoff,
            wolf_alpha=args.wolf_alpha,
            continue_mode=args.continue_mode,
            snapshot_interval=args.snapshot_interval,
            checkpoint_interval=args.checkpoint_interval,
        )
        
        if args.final_sweeps is not None:
            config.final_sweeps = args.final_sweeps
        
        logger.info("=" * 70)
        logger.info("INPUT")
        logger.info("=" * 70)
        logger.info(f"File       : {args.input_file}")
        logger.info(f"Mg label   : 45-M{x}")
        logger.info(f"N atoms    : {n_header}")
        logger.info(f"Seed       : {seed}")
        logger.info(f"Mode       : {'CONTINUE' if args.continue_mode else 'FULL PROTOCOL'}")
        logger.info(f"Output dir : {config.output_dir}")
        
        system = GlassSystem(config)
        sim = MCSimulator(system, config)
        sim.run()
        
        energy_data = {
            'total': sim.energy_log,
            'short': sim.short_log,
            'coul': sim.coul_log,
        }
        
        block_energy = compute_block_energy(sim.energy_log, config.block_size)
        logger.info(
            f"Block Energy: mean={block_energy['mean']:.6f}, "
            f"std={block_energy['std']:.6f}"
        )
        
        # Save final structure
        xyz_path = config.output_dir / "final_structure.xyz"
        with open(xyz_path, 'w') as f:
            f.write(f"{system.N_ATOMS}\n")
            f.write(
                f"Final Mg-doped 45S5 paper, label=45-M{x}, "
                f"x={x}, N={system.N_ATOMS}, "
                f"rho={system.effective_density:.6f} g/cm3, "
                f"box={system.box:.6f} A, seed={config.seed}\n"
            )
            for i in range(system.N_ATOMS):
                elem = system.type_to_elem[system.type_indices[i]]
                x_, y_, z_ = system.coords[i]
                f.write(f"{elem:2s} {x_:12.6f} {y_:12.6f} {z_:12.6f}\n")
        logger.info(f"Final XYZ saved to {xyz_path}")
        
        # Save energy log
        energy_path = config.output_dir / "energy_log.csv"
        with open(energy_path, 'w') as f:
            f.write("Step,Total_eV_per_atom,ShortRange_eV_per_atom,Coulomb_eV_per_atom\n")
            for i, (e, s, c) in enumerate(
                zip(sim.energy_log, sim.short_log, sim.coul_log)
            ):
                step = (i + 1) * sim.log_freq
                f.write(f"{step},{e:.6f},{s:.6f},{c:.6f}\n")
        logger.info(f"Energy log saved to {energy_path}")
        
        # Save block averaging
        block_path = config.output_dir / "block_averaging.txt"
        with open(block_path, 'w') as f:
            f.write(f"Block size: {config.block_size}\n")
            f.write(f"Number of blocks: {len(sim.energy_log) // config.block_size}\n")
            f.write(f"Mean energy: {block_energy['mean']:.6f} eV/atom\n")
            f.write(f"Standard error: {block_energy['std']:.6f} eV/atom\n")
        logger.info(f"Block averaging saved to {block_path}")
        
        logger.info(f"[OK] All results saved to {config.output_dir}")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()