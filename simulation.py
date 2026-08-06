#!/usr/bin/env python3
"""
========================================================================
45S5/Sr Bioglass NVT Monte Carlo - FINAL v10.0 (ALL-IN-ONE)
Based on: Xiang & Du, Chem. Mater. 2011, 23, 2703-2717

COMPLETE PACKAGE:
  - NVT Monte Carlo simulation (no artificial constraints)
  - Full structural analysis (RDF, CN, Qn, angles, O speciation)
  - Block averaging with standard deviation
  - Sensitivity analysis (cutoff & Wolf alpha)
  - Clustering R_XX, Modifier preference, Fnet, Si-O-P linkages
  - Neutron structure factor
  - Complete Excel export + plots

Usage:
  python Sr.py initial_Sr0.xyz --final-sweeps 150 --cores 12
  python Sr.py final_structure.xyz --continue --final-sweeps 300 --cores 12
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
from tqdm import tqdm
from numba import njit
from math import erfc, exp, sqrt, pi
from collections import defaultdict
from multiprocessing import Pool
import typer

warnings.filterwarnings('ignore')

# ========================= UTF-8 LOGGING =========================
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None: stream = sys.stdout
        if hasattr(stream, 'buffer'):
            stream = codecs.getwriter('utf-8')(stream.buffer)
        super().__init__(stream)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('simulation.log', 'w', 'utf-8'), UTF8StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========================= FREUD (optional) =========================
try:
    import freud
    FREUD_AVAILABLE = True
except Exception:
    FREUD_AVAILABLE = False

# ========================= CONSTANTS =========================
KB_EV = 8.617333262145e-5
NA = 6.02214076e23
KE = 14.3996454784255
SKIN = 1.5
SWITCH_DR = 0.3
BASE_N_ATOMS = 2835

# Paper reference values
PAPER_BOND_LENGTHS = {
    ('O','O'):2.63, ('Si','O'):1.61, ('Ca','O'):2.38, ('Na','O'):2.40,
    ('Sr','O'):2.59, ('Si','Si'):3.16, ('Si','Na'):3.29, ('Si','Ca'):3.64
}
PAPER_CN = {'Ca': 6.2, 'Na': 5.7, 'Sr': 7.0}
PAPER_QN_SI = {'Q0':3.7, 'Q1':23.2, 'Q2':39.8, 'Q3':27.8, 'Q4':5.6}
PAPER_QN_P = {'Q0':50.0, 'Q1':42.9, 'Q2':6.4, 'Q3':0.6, 'Q4':0.0}
PAPER_NC = {'Si': 2.07, 'P': 0.62, 'overall': 1.92}

NEUTRON_B = {'O': 5.803, 'Si': 4.1491, 'Na': 3.63, 'Ca': 4.70, 'Sr': 7.02, 'P': 5.13}


class SimulationError(Exception): pass
class FileError(SimulationError): pass


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
    for key, pat, conv in [
        ('x', r"x\s*=\s*(\d+)", int),
        ('N', r"N\s*=\s*(\d+)", int),
        ('density', r"rho\s*=\s*([0-9]*\.?[0-9]+)", float),
        ('box', r"box\s*=\s*([0-9]*\.?[0-9]+)", float),
        ('seed', r"seed\s*=\s*(\d+)", int),
    ]:
        m = re.search(pat, second)
        if m: meta[key] = conv(m.group(1))
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
            self.output_dir = Path(f"Bioglass_x{self.x}_N_seed{self.seed}")
        else:
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ========================= POTENTIAL =========================
class Potential:
    def __init__(self):
        self.zbl_a0 = 0.46850
        self.zbl_c = np.array([0.1818, 0.5099, 0.2802, 0.02817])
        self.zbl_d = np.array([3.2, 0.9423, 0.4029, 0.2016])
        self.zbl_z = {'Si': 14.0, 'Ca': 20.0, 'Na': 11.0, 'P': 15.0, 'O': 8.0, 'Sr': 38.0}

        # Xiang & Du 2011, Table 2 - ORIGINAL values only
        self.buck_params = {
            ('Si', 'O'): {'A': 13702.905,  'F': 0.193817, 'C': 54.681},
            ('P',  'O'): {'A': 26655.472,  'F': 0.181968, 'C': 86.856},
            ('O',  'O'): {'A': 2029.2204, 'F': 0.343645, 'C': 192.58},
            ('Na', 'O'): {'A': 4383.7555,  'F': 0.243838, 'C': 30.70},
            ('Ca', 'O'): {'A': 7747.1834,  'F': 0.252623, 'C': 93.109},
            ('Sr', 'O'): {'A': 14566.637,  'F': 0.245015, 'C': 81.773},
        }
        self.r_hard_default = 0.9
        self.r_hard_by_pair = {('O', 'O'): 1.4}
        self.sbs_x_o = {'Ca': 32.0, 'Na': 20.0, 'Sr': 32.0}


# ========================= GLASS SYSTEM =========================
class GlassSystem:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.masses = {'Si': 28.0855, 'Ca': 40.078, 'Na': 22.98977,
                       'P': 30.97376, 'O': 15.999, 'Sr': 87.62}
        self.type_map = {'Si': 0, 'Ca': 1, 'Na': 2, 'P': 3, 'O': 4, 'Sr': 5}
        self.type_to_elem = {0: 'Si', 1: 'Ca', 2: 'Na', 3: 'P', 4: 'O', 5: 'Sr'}
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
            if len(parts) < 4: continue
            self.symbols.append(parts[0])
            coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
        self.N_ATOMS = len(self.symbols)
        if self.N_ATOMS != n_header:
            logger.warning(f"XYZ header says {n_header}, but {self.N_ATOMS} lines read.")
        if self.N_ATOMS % BASE_N_ATOMS != 0:
            logger.warning(f"Total atoms {self.N_ATOMS} not a multiple of {BASE_N_ATOMS}.")
        logger.info(f"Loaded {self.N_ATOMS} atoms from {self.config.input_file}")
        self.coords = np.array(coords_list, dtype=np.float64)
        self.type_indices = np.array([self.type_map[s] for s in self.symbols], dtype=np.int32)
        self.counts = {e: int(np.sum(self.type_indices == self.type_map[e])) for e in set(self.symbols)}
        logger.info(f"Composition: {self.counts}")
        self.total_mass = sum(self.masses[s] for s in self.symbols)

    def _setup_box(self):
        if self.config.box_override is not None:
            self.box = float(self.config.box_override)
            self.effective_density = self.total_mass * 1.0e24 / (NA * self.box**3)
            logger.info(f"Using box from XYZ: {self.box:.4f} A (density={self.effective_density:.4f})")
        elif self.config.density is not None:
            self.effective_density = float(self.config.density)
            volume_A3 = (self.total_mass / NA) / self.effective_density * 1.0e24
            self.box = volume_A3 ** (1.0 / 3.0)
            logger.info(f"Using density {self.effective_density:.4f} -> box={self.box:.4f} A")
        else:
            raise FileError("Neither box nor density could be determined.")
        self.coords = self.coords % self.box

    def _compute_charges(self):
        base = {'Si': 2.4, 'Ca': 1.2, 'Na': 0.6, 'P': 3.0, 'O': -1.2, 'Sr': 1.2}
        self.charges = np.array([base.get(s, 0.0) for s in self.symbols], dtype=np.float64)
        total_pos = sum(self.counts.get(e, 0) * base.get(e, 0) for e in self.counts if e != 'O')
        n_o = self.counts.get('O', 0)
        if n_o <= 0:
            raise SimulationError("No oxygen atoms found.")
        q_o = -total_pos / n_o
        self.charges[self.type_indices == self.type_map['O']] = q_o
        logger.info(f"O charge set to {q_o:.6f} for charge neutrality.")

    def get_type_Z(self):
        return np.array([self.potential.zbl_z[self.type_to_elem[i]] for i in range(6)], dtype=np.float64)


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
            counts[i] += 1; counts[j] += 1
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
def zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d):
    if r < 1.0e-6: return 1.0e8
    a = zbl_a0 / (Zi**0.23 + Zj**0.23)
    x = r / a
    phi = (zbl_c[0]*exp(-zbl_d[0]*x) + zbl_c[1]*exp(-zbl_d[1]*x) +
           zbl_c[2]*exp(-zbl_d[2]*x) + zbl_c[3]*exp(-zbl_d[3]*x))
    return KE * Zi * Zj * phi / r

@njit(fastmath=True, cache=True)
def wolf_coulomb(qi, qj, r, alpha, cutoff):
    if r >= cutoff or r < 1.0e-12: return 0.0
    ar = alpha * r
    ac = alpha * cutoff
    er = erfc(ar); ec = erfc(ac)
    t1 = er / r
    t2 = ec / cutoff
    t3 = ((ec/(cutoff*cutoff)) + (2.0*alpha/sqrt(pi))*exp(-ac*ac)/cutoff) * (r - cutoff)
    return KE * qi * qj * (t1 - t2 + t3)

@njit(fastmath=True, cache=True)
def pair_energy(r, qi, qj, ti, tj, Zi, Zj, A_mat, F_mat, C_mat, R_HARD_MAT,
                cutoff, alpha, zbl_a0, zbl_c, zbl_d):
    if r >= cutoff or r < 1.0e-12: return 0.0
    r_in = R_HARD_MAT[ti, tj]
    r_out = r_in + SWITCH_DR
    if r < r_out:
        e_zbl = zbl_repulsion(r, Zi, Zj, zbl_a0, zbl_c, zbl_d)
        if r < r_in: return e_zbl
        A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12: e_buck += A * exp(-r / F)
        if C > 0.0: e_buck -= C / (r**6)
        e_coul = wolf_coulomb(qi, qj, r, alpha, cutoff)
        e_full = e_buck + e_coul
        x_switch = (r - r_in) / SWITCH_DR
        s = x_switch**3 * (10.0 - 15.0*x_switch + 6.0*x_switch**2)
        return s * e_full + (1.0 - s) * e_zbl
    else:
        A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
        e_buck = 0.0
        if A > 0.0 and F > 1.0e-12: e_buck += A * exp(-r / F)
        if C > 0.0: e_buck -= C / (r**6)
        return e_buck + wolf_coulomb(qi, qj, r, alpha, cutoff)

@njit(fastmath=True, cache=True)
def local_energy(idx, coords, charges, types, type_Z, A_mat, F_mat, C_mat,
                 R_HARD_MAT, box, cutoff, alpha, neighbors, starts,
                 zbl_a0, zbl_c, zbl_d):
    e = 0.0
    qi = charges[idx]; ti = types[idx]; Zi = type_Z[ti]
    xi, yi, zi = coords[idx]
    for p in range(starts[idx], starts[idx+1]):
        j = neighbors[p]
        dx = xi - coords[j,0]; dy = yi - coords[j,1]; dz = zi - coords[j,2]
        dx -= box * round(dx / box)
        dy -= box * round(dy / box)
        dz -= box * round(dz / box)
        r2 = dx*dx + dy*dy + dz*dz
        if r2 >= cutoff*cutoff: continue
        r = sqrt(r2)
        e += pair_energy(r, qi, charges[j], ti, types[j], Zi, type_Z[types[j]],
                         A_mat, F_mat, C_mat, R_HARD_MAT, cutoff, alpha,
                         zbl_a0, zbl_c, zbl_d)
    e += -KE * (alpha / sqrt(pi)) * qi * qi
    return e

@njit(fastmath=True, cache=True)
def total_energy(coords, charges, types, type_Z, A_mat, F_mat, C_mat,
                 R_HARD_MAT, neighbors, starts, box, cutoff, alpha,
                 zbl_a0, zbl_c, zbl_d):
    e = 0.0
    n = coords.shape[0]
    for i in range(n):
        xi, yi, zi = coords[i]
        qi = charges[i]; ti = types[i]; Zi = type_Z[ti]
        for p in range(starts[i], starts[i+1]):
            j = neighbors[p]
            if j <= i: continue
            dx = xi - coords[j,0]; dy = yi - coords[j,1]; dz = zi - coords[j,2]
            dx -= box * round(dx / box)
            dy -= box * round(dy / box)
            dz -= box * round(dz / box)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= cutoff*cutoff: continue
            r = sqrt(r2)
            e += pair_energy(r, qi, charges[j], ti, types[j], Zi, type_Z[types[j]],
                             A_mat, F_mat, C_mat, R_HARD_MAT, cutoff, alpha,
                             zbl_a0, zbl_c, zbl_d)
    return e

@njit(fastmath=True, cache=True)
def energy_decomposition(coords, charges, types, type_Z, A_mat, F_mat, C_mat,
                         R_HARD_MAT, neighbors, starts, box, cutoff, alpha,
                         zbl_a0, zbl_c, zbl_d):
    sr = 0.0; cl = 0.0
    n = coords.shape[0]
    for i in range(n):
        xi, yi, zi = coords[i]
        qi = charges[i]; ti = types[i]; Zi = type_Z[ti]
        for p in range(starts[i], starts[i+1]):
            j = neighbors[p]
            if j <= i: continue
            dx = xi - coords[j,0]; dy = yi - coords[j,1]; dz = zi - coords[j,2]
            dx -= box * round(dx / box)
            dy -= box * round(dy / box)
            dz -= box * round(dz / box)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= cutoff*cutoff: continue
            r = sqrt(r2)
            tj = types[j]
            rh = R_HARD_MAT[ti, tj]
            if r < rh:
                sr += zbl_repulsion(r, Zi, type_Z[tj], zbl_a0, zbl_c, zbl_d)
            elif r < rh + SWITCH_DR:
                e_zbl = zbl_repulsion(r, Zi, type_Z[tj], zbl_a0, zbl_c, zbl_d)
                A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
                e_buck = 0.0
                if A > 0.0 and F > 1.0e-12: e_buck += A * exp(-r / F)
                if C > 0.0: e_buck -= C / (r**6)
                e_coul = wolf_coulomb(qi, charges[j], r, alpha, cutoff)
                x_switch = (r - rh) / SWITCH_DR
                s = x_switch**3 * (10.0 - 15.0*x_switch + 6.0*x_switch**2)
                sr += s * e_buck + (1.0 - s) * e_zbl
                cl += s * e_coul
            else:
                A = A_mat[ti, tj]; F = F_mat[ti, tj]; C = C_mat[ti, tj]
                if A > 0.0 and F > 1.0e-12: sr += A * exp(-r / F)
                if C > 0.0: sr -= C / (r**6)
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
        self.max_disp = {'Si': 0.04, 'Ca': 0.08, 'Na': 0.08,
                         'P': 0.05, 'O': 0.08, 'Sr': 0.08}
        self.energy_log = []
        self.short_log = []
        self.coul_log = []
        self.snapshots = []
        self.step = 0
        self.current_energy = 0.0
        self.current_short = 0.0
        self.current_coul = 0.0
        self.coords_ref = None

        N = system.N_ATOMS
        self.mixing_steps = max(1, int(config.mixing_sweeps * N))
        self.annealing_stages = [(T, max(1, int(s * N))) for T, s in config.annealing_sweeps]
        self.healing_steps = max(1, int(config.healing_sweeps * N))
        self.low_t_steps = max(1, int(config.low_t_sweeps * N))
        self.final_steps = max(1, int(config.final_sweeps * N))
        self.snapshot_interval = (
            config.snapshot_interval if config.snapshot_interval is not None
            else max(N, self.final_steps // max(1, config.max_snapshots))
        )
        self.tune_freq = (config.adaptive_tuning_freq if config.adaptive_tuning_freq is not None
                          else max(500, N // 10))
        self.log_freq = (config.log_freq if config.log_freq is not None
                         else max(100, N // 100))
        self.decomposition_freq = (config.decomposition_freq if config.decomposition_freq is not None
                                   else max(5000, N))

        self._prepare_matrices()
        self._init_energy()
        logger.info(f"MC schedule: mixing={self.mixing_steps}, "
                    f"annealing={sum(s for _, s in self.annealing_stages)}, "
                    f"healing={self.healing_steps}, low-T={self.low_t_steps}, "
                    f"production={self.final_steps}")

    def _prepare_matrices(self):
        nt = 6
        self.A_mat = np.zeros((nt, nt), dtype=np.float64)
        self.F_mat = np.zeros((nt, nt), dtype=np.float64)
        self.C_mat = np.zeros((nt, nt), dtype=np.float64)
        self.R_HARD_MAT = np.full((nt, nt), self.system.potential.r_hard_default, dtype=np.float64)
        for (e1, e2), val in self.system.potential.r_hard_by_pair.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.R_HARD_MAT[t1, t2] = val
            self.R_HARD_MAT[t2, t1] = val
        pot = self.system.potential
        for (e1, e2), p in pot.buck_params.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.A_mat[t1, t2] = p['A']
            self.F_mat[t1, t2] = p['F']
            self.C_mat[t1, t2] = p['C']
            if t1 != t2:
                self.A_mat[t2, t1] = p['A']
                self.F_mat[t2, t1] = p['F']
                self.C_mat[t2, t1] = p['C']
        self.type_Z = self.system.get_type_Z()
        self.wolf_self = -KE * (self.config.wolf_alpha / sqrt(pi)) * np.sum(self.system.charges**2)

    def _init_energy(self):
        n, s = self.nl.neighbors, self.nl.starts
        U = total_energy(self.system.coords, self.system.charges, self.system.type_indices,
                         self.type_Z, self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
                         n, s, self.system.box, self.config.cutoff, self.config.wolf_alpha,
                         self.system.potential.zbl_a0, self.system.potential.zbl_c,
                         self.system.potential.zbl_d)
        self.current_energy = U + self.wolf_self
        sr, cl = energy_decomposition(self.system.coords, self.system.charges,
                                      self.system.type_indices, self.type_Z,
                                      self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
                                      n, s, self.system.box, self.config.cutoff,
                                      self.config.wolf_alpha,
                                      self.system.potential.zbl_a0,
                                      self.system.potential.zbl_c,
                                      self.system.potential.zbl_d)
        self.current_short = sr
        self.current_coul = cl + self.wolf_self
        self.coords_ref = self.system.coords.copy()
        logger.info(f"Initial energy: {self.current_energy / self.system.N_ATOMS:.6f} eV/atom")

    def _maybe_log(self):
        if self.step % self.log_freq == 0:
            if self.step % self.decomposition_freq == 0:
                self._update_energy_decomp()
            self.energy_log.append(self.current_energy / self.system.N_ATOMS)
            self.short_log.append(self.current_short / self.system.N_ATOMS)
            self.coul_log.append(self.current_coul / self.system.N_ATOMS)

    def _mc_move(self, T: float) -> bool:
        n = self.system.N_ATOMS
        i = int(self.rng.integers(0, n))
        old_pos = self.system.coords[i].copy()
        elem = self.system.type_to_elem[self.system.type_indices[i]]
        maxd = self.max_disp.get(elem, 0.06)
        neigh, st = self.nl.neighbors, self.nl.starts
        old_e = local_energy(i, self.system.coords, self.system.charges,
                             self.system.type_indices, self.type_Z,
                             self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
                             self.system.box, self.config.cutoff, self.config.wolf_alpha,
                             neigh, st,
                             self.system.potential.zbl_a0, self.system.potential.zbl_c,
                             self.system.potential.zbl_d)
        delta = self.rng.uniform(-maxd, maxd, size=3)
        self.system.coords[i] = (old_pos + delta) % self.system.box
        new_e = local_energy(i, self.system.coords, self.system.charges,
                             self.system.type_indices, self.type_Z,
                             self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
                             self.system.box, self.config.cutoff, self.config.wolf_alpha,
                             neigh, st,
                             self.system.potential.zbl_a0, self.system.potential.zbl_c,
                             self.system.potential.zbl_d)
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
            if np.sqrt(np.sum(dr**2)) > self.config.skin / 3.0:
                self.tracker['need_rebuild'] = True
                self._update_neighbors()
        return accepted

    def _update_neighbors(self):
        if self.tracker['need_rebuild']:
            self.nl.update(force=True)
            self.coords_ref = self.system.coords.copy()
            self.tracker['need_rebuild'] = False
            self.rebuilds += 1

    def _update_energy_decomp(self):
        n, s = self.nl.neighbors, self.nl.starts
        sr, cl = energy_decomposition(self.system.coords, self.system.charges,
                                      self.system.type_indices, self.type_Z,
                                      self.A_mat, self.F_mat, self.C_mat, self.R_HARD_MAT,
                                      n, s, self.system.box, self.config.cutoff,
                                      self.config.wolf_alpha,
                                      self.system.potential.zbl_a0,
                                      self.system.potential.zbl_c,
                                      self.system.potential.zbl_d)
        self.current_short = sr
        self.current_coul = cl + self.wolf_self

    def _run_stage(self, T, steps, desc, adaptive=False, mixing=False, enhance_oxygen=False):
        logger.info(f"{desc}: T={T:.1f} K, steps={steps}")
        stage_acc = 0; stage_att = 0
        window_acc = 0; window_att = 0
        ema = 0.5
        orig_disp = self.max_disp.copy()
        if mixing:
            for k in self.max_disp: self.max_disp[k] *= 2.0
            self.max_disp['O'] *= 1.5
        if enhance_oxygen and T >= 1500.0:
            self.max_disp['O'] = min(self.max_disp['O'] * 1.3, 0.20)
        pbar = tqdm(range(steps), desc=f"T={T:.0f}K")
        for _ in pbar:
            self.step += 1
            self.attempts += 1
            stage_att += 1
            window_att += 1
            acc = self._mc_move(T)
            if acc:
                self.accepted += 1
                stage_acc += 1
                window_acc += 1
            self._maybe_log()
            if adaptive and (self.step % self.tune_freq == 0):
                ca = window_acc / max(1, window_att)
                ema = 0.8 * ema + 0.2 * ca
                target = self.config.target_acceptance
                factor = 1.0
                if ema < target - 0.05: factor = 0.92
                elif ema > target + 0.05: factor = 1.08
                if factor != 1.0:
                    for e in self.max_disp:
                        self.max_disp[e] = min(max(self.max_disp[e] * factor, 0.02), 0.25)
                window_acc = 0; window_att = 0
            pbar.set_postfix({'acc': f'{self.accepted / max(1, self.attempts) * 100:.1f}%'})
        pbar.close()
        self.max_disp = orig_disp
        logger.info(f"  Stage acceptance: {stage_acc / max(1, stage_att) * 100:.2f}%")

    def run(self):
        logger.info("=" * 70)
        logger.info(f"STARTING NVT MONTE CARLO - x={self.config.x} mol% SrO")
        if self.config.continue_mode:
            logger.info("MODE: CONTINUE (Skipping mixing/annealing)")
        logger.info("=" * 70)
        if self.config.continue_mode:
            self._run_stage(2500.0, max(1, int(3 * self.system.N_ATOMS)),
                            "Short Healing", adaptive=True, enhance_oxygen=True)
            self._run_stage(1500.0, max(1, int(2 * self.system.N_ATOMS)),
                            "Cool down 1", adaptive=True)
            self._run_stage(800.0, max(1, int(2 * self.system.N_ATOMS)),
                            "Cool down 2", adaptive=True)
            for k in self.max_disp: self.max_disp[k] *= 0.5
            self._run_stage(1.0, self.low_t_steps, "Low-T relaxation")
        else:
            self._run_stage(self.config.mixing_temp, self.mixing_steps,
                            "Mixing", adaptive=True, mixing=True, enhance_oxygen=True)
            for i, (T, steps) in enumerate(self.annealing_stages):
                self._run_stage(T, steps, f"Annealing {i+1}", adaptive=True, enhance_oxygen=True)
            for k in self.max_disp: self.max_disp[k] *= 0.7
            self._run_stage(self.config.healing_temp, self.healing_steps,
                            "Defect Healing", adaptive=True, enhance_oxygen=True)
            self._run_stage(1000.0, max(1, int(3 * self.system.N_ATOMS)),
                            "Cool after healing", adaptive=True)
            self._run_stage(600.0, max(1, int(2 * self.system.N_ATOMS)),
                            "Cool after healing", adaptive=True)
            for k in self.max_disp: self.max_disp[k] *= 0.5
            self._run_stage(1.0, self.low_t_steps, "Low-T relaxation")
        # PRODUCTION
        for k in self.max_disp:
            self.max_disp[k] = min(max(self.max_disp[k] * 0.8, 0.02), 0.15)
        logger.info(f"Production: T=300 K, steps={self.final_steps}")
        pbar = tqdm(range(self.final_steps), desc="T=300K")
        for s in pbar:
            self.step += 1
            self.attempts += 1
            if self._mc_move(300.0):
                self.accepted += 1
            self._maybe_log()
            if (s + 1) % self.snapshot_interval == 0:
                self.snapshots.append({
                    'coords': self.system.coords.copy(),
                    'types': self.system.type_indices.copy(),
                    'box': self.system.box
                })
            pbar.set_postfix({'acc': f'{self.accepted / max(1, self.attempts) * 100:.1f}%'})
        pbar.close()
        logger.info("=" * 70)
        logger.info("SIMULATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total attempts: {self.attempts}, "
                    f"acceptance: {self.accepted / max(1, self.attempts) * 100:.2f}%, "
                    f"rebuilds: {self.rebuilds}")


# ========================= ANALYZER =========================
class Analyzer:
    def __init__(self, system, config):
        self.system = system
        self.config = config
        self.use_freud = FREUD_AVAILABLE
        self.FIXED_CUTOFFS = {
            ('Si','O'):2.25, ('P','O'):2.25, ('Na','O'):3.34, ('Ca','O'):3.14,
            ('Sr','O'):3.35, ('O','O'):2.91,
            ('P','Ca'):4.44, ('P','Na'):4.44, ('P','Sr'):4.58,
            ('Si','Ca'):4.37, ('Si','Na'):4.42, ('Si','Sr'):4.51,
            ('Ca','Ca'):4.85, ('Na','Na'):4.15, ('Sr','Sr'):5.14,
            ('Ca','Na'):4.94, ('Ca','Sr'):5.00, ('Na','Sr'):4.40
        }
        self.RMIN_VALUES = {
            ('Si','O'):1.4, ('P','O'):1.3, ('Na','O'):2.0, ('Ca','O'):2.0,
            ('Sr','O'):2.2, ('O','O'):2.2,
            ('P','Ca'):3.5, ('P','Na'):3.5, ('P','Sr'):3.5,
            ('Si','Ca'):3.5, ('Si','Na'):3.0, ('Si','Sr'):3.5,
            ('Ca','Ca'):3.5, ('Na','Na'):3.0, ('Sr','Sr'):4.0,
            ('Ca','Na'):3.5, ('Ca','Sr'):4.0, ('Na','Sr'):3.5
        }
        self.PEAK_RANGES = {
            ('Si','O'):(1.2,2.2), ('P','O'):(1.2,1.9), ('Na','O'):(1.8,2.8),
            ('Ca','O'):(1.8,2.8), ('Sr','O'):(2.0,3.0), ('O','O'):(2.0,3.2),
            ('P','Ca'):(3.0,5.0), ('P','Na'):(3.0,5.0), ('P','Sr'):(3.0,5.2),
            ('Si','Ca'):(3.0,5.0), ('Si','Na'):(2.8,5.0), ('Si','Sr'):(3.0,5.2),
            ('Ca','Ca'):(3.0,5.5), ('Na','Na'):(2.5,5.0), ('Sr','Sr'):(3.5,6.0),
            ('Ca','Na'):(3.0,5.5), ('Ca','Sr'):(3.5,5.8), ('Na','Sr'):(3.0,5.2)
        }
        self.SBS_X_O = {'Ca': 32.0, 'Na': 20.0, 'Sr': 32.0}
        self.ALL_PAIRS = [p for p in self.FIXED_CUTOFFS
                          if all(self.system.counts.get(e, 0) > 0 for e in p)
                          and (p[0] != p[1] or self.system.counts.get(p[0], 0) > 1)]
        self.type_map = system.type_map
        self.type_to_elem = system.type_to_elem
        self.si_o_cut_extra = [2.15, 2.20, 2.25, 2.30, 2.35]

    def _find_first_minimum(self, r, gr, pair, rmin=None, rmax=4.0):
        if rmin is None:
            rmin = self.RMIN_VALUES.get(pair, 1.5)
        mask = (r >= rmin) & (r <= rmax)
        if not np.any(mask): return None
        r_masked = r[mask]; gr_masked = gr[mask]
        if len(r_masked) < 3: return None
        gr_smooth = gaussian_filter1d(gr_masked, sigma=2.0)
        diff = np.diff(gr_smooth)
        for i in range(1, len(diff)):
            if diff[i-1] < 0 and diff[i] >= 0:
                return r_masked[i]
        return r_masked[np.argmin(gr_smooth)]

    def compute_rdf(self, coords, types, ei, ej, box, cut, rmax=6.0, nbins=400):
        ti = self.type_map[ei]; tj = self.type_map[ej]
        mi = types == ti; mj = types == tj
        ni = int(np.sum(mi)); nj = int(np.sum(mj))
        if ni == 0 or nj == 0 or (ti == tj and ni < 2):
            return 0.0, 0.0, np.zeros(nbins), np.zeros(nbins)
        if self.use_freud:
            try:
                rdf = freud.density.RDF(bins=nbins, r_max=rmax)
                rdf.compute(system=(coords[mi], box), query_points=coords[mj])
                r = rdf.bin_centers; gr = rdf.rdf
                pr = self.PEAK_RANGES.get((ei, ej), (1.0, 3.0))
                mask_peak = (r >= pr[0]) & (r <= pr[1])
                peak = 0.0
                if np.any(mask_peak) and np.max(gr[mask_peak]) > 0:
                    peak = r[mask_peak][np.argmax(gr[mask_peak])]
                mask_cn = (r > 0.3) & (r <= cut)
                if np.any(mask_cn):
                    rho = (nj - 1 if ti == tj else nj) / box**3
                    cn = trapezoid(gr[mask_cn]*4*np.pi*r[mask_cn]**2*rho, r[mask_cn])
                else: cn = 0.0
                return cn, peak, r, gr
            except Exception:
                pass
        if ni <= nj:
            small_idx = np.where(mi)[0]; large_coords = coords[mj]
        else:
            small_idx = np.where(mj)[0]; large_coords = coords[mi]
        hist = np.zeros(nbins); bw = rmax / nbins
        for idx in small_idx:
            xi, yi, zi = coords[idx]
            dx = xi - large_coords[:,0]; dy = yi - large_coords[:,1]; dz = zi - large_coords[:,2]
            dx -= box * np.round(dx / box)
            dy -= box * np.round(dy / box)
            dz -= box * np.round(dz / box)
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            mask = (r > 0.3) & (r < rmax)
            r_masked = r[mask]
            if len(r_masked) == 0: continue
            bins = np.floor(r_masked / bw).astype(np.int64)
            bins = bins[bins < nbins]
            hist[bins] += 1
        r_edges = np.linspace(0, rmax, nbins + 1)
        rc = 0.5 * (r_edges[:-1] + r_edges[1:])
        dr = r_edges[1] - r_edges[0]
        vol = 4 * np.pi * rc**2 * dr
        norm_factor = box**3 / (ni*(ni-1)) if ti == tj else box**3 / (ni*nj)
        gr = (hist * norm_factor) / vol
        pr = self.PEAK_RANGES.get((ei, ej), (1.0, 3.0))
        mask_peak = (rc >= pr[0]) & (rc <= pr[1])
        peak = 0.0
        if np.any(mask_peak) and np.max(gr[mask_peak]) > 0:
            peak = rc[mask_peak][np.argmax(gr[mask_peak])]
        mask_cn = (rc > 0.3) & (rc <= cut)
        if np.any(mask_cn):
            rho = (nj - 1 if ti == tj else nj) / box**3
            cn = trapezoid(gr[mask_cn]*4*np.pi*rc[mask_cn]**2*rho, rc[mask_cn])
        else: cn = 0.0
        return cn, peak, rc, gr

    def _compute_angles(self, coords, types, box):
        si_o_cut = self.FIXED_CUTOFFS[('Si','O')]
        si_idx = np.where(types == self.type_map['Si'])[0]
        o_idx = np.where(types == self.type_map['O'])[0]
        n = len(coords)
        tree = cKDTree(coords, boxsize=box)
        pairs = tree.query_pairs(si_o_cut + 0.1, output_type='ndarray')
        starts = np.zeros(n+1, dtype=np.int32)
        counts = np.zeros(n, dtype=np.int32)
        for i, j in pairs:
            counts[i] += 1; counts[j] += 1
        starts[1:] = np.cumsum(counts)
        neighbors = np.empty(starts[-1], dtype=np.int32)
        fill = starts[:-1].copy()
        for i, j in pairs:
            neighbors[fill[i]] = j; fill[i] += 1
            neighbors[fill[j]] = i; fill[j] += 1
        osi_angles = []
        for si in si_idx:
            o_neighbors = []
            for k in range(starts[si], starts[si+1]):
                j = neighbors[k]
                if types[j] == self.type_map['O']:
                    dx = coords[si,0]-coords[j,0]; dy = coords[si,1]-coords[j,1]; dz = coords[si,2]-coords[j,2]
                    dx -= box*np.round(dx/box); dy -= box*np.round(dy/box); dz -= box*np.round(dz/box)
                    r = sqrt(dx*dx+dy*dy+dz*dz)
                    if r < si_o_cut: o_neighbors.append(j)
            for a in range(len(o_neighbors)):
                for b in range(a+1, len(o_neighbors)):
                    v1 = coords[o_neighbors[a]] - coords[si]
                    v2 = coords[o_neighbors[b]] - coords[si]
                    v1 -= box*np.round(v1/box); v2 -= box*np.round(v2/box)
                    cos_ang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                    osi_angles.append(np.degrees(np.arccos(np.clip(cos_ang,-1,1))))
        siosi_angles = []
        o_si = [[] for _ in range(n)]
        for o in o_idx:
            for k in range(starts[o], starts[o+1]):
                j = neighbors[k]
                if types[j] == self.type_map['Si']:
                    dx = coords[o,0]-coords[j,0]; dy = coords[o,1]-coords[j,1]; dz = coords[o,2]-coords[j,2]
                    dx -= box*np.round(dx/box); dy -= box*np.round(dy/box); dz -= box*np.round(dz/box)
                    r = sqrt(dx*dx+dy*dy+dz*dz)
                    if r < si_o_cut: o_si[o].append(j)
        for o, sil in enumerate(o_si):
            if len(sil) == 2:
                v1 = coords[sil[0]] - coords[o]; v2 = coords[sil[1]] - coords[o]
                v1 -= box*np.round(v1/box); v2 -= box*np.round(v2/box)
                cos_ang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                siosi_angles.append(np.degrees(np.arccos(np.clip(cos_ang,-1,1))))
        return osi_angles, siosi_angles

    def analyze_snapshot(self, snap):
        cs = snap['coords']; ts = snap['types']; bx = snap['box']
        tree = cKDTree(cs, boxsize=bx)
        max_cut = max(self.FIXED_CUTOFFS.values())
        pairs = tree.query_pairs(max_cut + 0.1, output_type='ndarray')
        n = len(cs)
        counts = np.zeros(n, dtype=np.int32)
        for i, j in pairs: counts[i] += 1; counts[j] += 1
        starts = np.zeros(n+1, dtype=np.int32)
        starts[1:] = np.cumsum(counts)
        neighbors = np.empty(starts[-1], dtype=np.int32)
        fill = starts[:-1].copy()
        for i, j in pairs:
            neighbors[fill[i]] = j; fill[i] += 1
            neighbors[fill[j]] = i; fill[j] += 1
        cn_fixed = {}; cn_auto = {}; auto_cutoff_dict = {}; cn_extra = {}
        for pair in self.ALL_PAIRS:
            ei, ej = pair
            cut_fixed = self.FIXED_CUTOFFS[pair]
            cn, peak, r, gr = self.compute_rdf(cs, ts, ei, ej, bx, cut_fixed, rmax=6.0, nbins=400)
            cn_fixed[pair] = cn
            auto_cut = self._find_first_minimum(r, gr, pair)
            if auto_cut is not None and auto_cut > 0.5:
                auto_cutoff_dict[pair] = auto_cut
                mask_cn = (r > 0.3) & (r <= auto_cut)
                if np.any(mask_cn):
                    r_cn = r[mask_cn]; gr_cn = gr[mask_cn]
                    ti, tj = self.type_map[ei], self.type_map[ej]
                    nj = int(np.sum(ts == tj))
                    rho = (nj - 1 if ti == tj else nj) / bx**3
                    cn_auto_val = trapezoid(4*np.pi*r_cn**2*rho*gr_cn, r_cn)
                else: cn_auto_val = 0.0
                cn_auto[pair] = cn_auto_val
            else:
                auto_cutoff_dict[pair] = cut_fixed
                cn_auto[pair] = cn_fixed[pair]
            if pair == ('Si','O'):
                cn_extra[pair] = {}
                for rcut in self.si_o_cut_extra:
                    mask_cn = (r > 0.3) & (r <= rcut)
                    if np.any(mask_cn):
                        r_cn = r[mask_cn]; gr_cn = gr[mask_cn]
                        ti, tj = self.type_map[ei], self.type_map[ej]
                        nj = int(np.sum(ts == tj))
                        rho = (nj - 1 if ti == tj else nj) / bx**3
                        cn_val = trapezoid(4*np.pi*r_cn**2*rho*gr_cn, r_cn)
                    else: cn_val = 0.0
                    cn_extra[pair][rcut] = cn_val
        cn_res = cn_fixed
        si_o_cut = self.FIXED_CUTOFFS[('Si','O')]
        p_o_cut = self.FIXED_CUTOFFS[('P','O')]
        si_idx = np.where(ts == self.type_map['Si'])[0]
        o_idx = np.where(ts == self.type_map['O'])[0]
        p_idx = np.where(ts == self.type_map['P'])[0]
        o_nf = []
        for o in o_idx:
            nsi = 0; np_ = 0
            for k in range(starts[o], starts[o+1]):
                j = neighbors[k]
                dx = cs[o,0]-cs[j,0]; dy = cs[o,1]-cs[j,1]; dz = cs[o,2]-cs[j,2]
                dx -= bx*np.round(dx/bx); dy -= bx*np.round(dy/bx); dz -= bx*np.round(dz/bx)
                r = sqrt(dx*dx+dy*dy+dz*dz)
                if ts[j] == self.type_map['Si'] and r < si_o_cut: nsi += 1
                elif ts[j] == self.type_map['P'] and r < p_o_cut: np_ += 1
            o_nf.append(nsi + np_)
        fo = sum(1 for c in o_nf if c == 0)
        nbo = sum(1 for c in o_nf if c == 1)
        bo = sum(1 for c in o_nf if c == 2)
        to = sum(1 for c in o_nf if c >= 3)
        bridging_set = {o_idx[i] for i, c in enumerate(o_nf) if c >= 2}
        si_bo = []
        for si in si_idx:
            bcnt = 0
            for k in range(starts[si], starts[si+1]):
                j = neighbors[k]
                if ts[j] == self.type_map['O']:
                    dx = cs[si,0]-cs[j,0]; dy = cs[si,1]-cs[j,1]; dz = cs[si,2]-cs[j,2]
                    dx -= bx*np.round(dx/bx); dy -= bx*np.round(dy/bx); dz -= bx*np.round(dz/bx)
                    r = sqrt(dx*dx+dy*dy+dz*dz)
                    if r < si_o_cut and j in bridging_set: bcnt += 1
            si_bo.append(bcnt)
        qn_si = {n: int(np.sum(np.array(si_bo) == n)) for n in range(5)}
        if len(p_idx) > 0:
            p_bo = []
            for p in p_idx:
                bcnt = 0
                for k in range(starts[p], starts[p+1]):
                    j = neighbors[k]
                    if ts[j] == self.type_map['O']:
                        dx = cs[p,0]-cs[j,0]; dy = cs[p,1]-cs[j,1]; dz = cs[p,2]-cs[j,2]
                        dx -= bx*np.round(dx/bx); dy -= bx*np.round(dy/bx); dz -= bx*np.round(dz/bx)
                        r = sqrt(dx*dx+dy*dy+dz*dz)
                        if r < p_o_cut and j in bridging_set: bcnt += 1
                p_bo.append(bcnt)
            qn_p = {n: int(np.sum(np.array(p_bo) == n)) for n in range(5)}
        else:
            qn_p = {n: 0 for n in range(5)}
        N_SI = self.system.counts.get('Si', 0)
        N_P = self.system.counts.get('P', 0)
        N_net = N_SI + N_P
        qn_c = {n: (qn_si[n]+qn_p[n])/N_net*100 if N_net > 0 else 0.0 for n in range(5)}
        nc = sum(n * qn_c[n] for n in range(1, 5)) / 100.0
        V = bx**3
        rxx = {}
        for elem in ['Ca', 'Na', 'Sr']:
            tid = self.type_map[elem]
            Nx = int(np.sum(ts == tid))
            if Nx == 0: rxx[elem] = 0.0; continue
            pair = (elem, elem)
            if pair not in cn_res: rxx[elem] = 0.0; continue
            CN = cn_res.get(pair, 0.0)
            rc = self.FIXED_CUTOFFS.get(pair, 1.0)
            number_density = (Nx - 1) / V if Nx > 1 else 0.0
            denom = (4.0/3.0) * pi * rc**3 * number_density
            rxx[elem] = CN / denom if denom > 0 else 0.0
        modifier_preference = {}
        for A in ['Si', 'P']:
            for B, C in [('Sr','Ca'), ('Ca','Na'), ('Sr','Na')]:
                key = f"{A}_{B}_vs_{C}"
                NB = int(np.sum(ts == self.type_map[B]))
                NC = int(np.sum(ts == self.type_map[C]))
                if NB == 0 or NC == 0: modifier_preference[key] = 0.0; continue
                CN_AB = cn_res.get((A,B), cn_res.get((B,A), 0.0))
                CN_AC = cn_res.get((A,C), cn_res.get((C,A), 0.0))
                if CN_AC > 0: modifier_preference[key] = (CN_AB/CN_AC)*(NC/NB)
                else: modifier_preference[key] = 0.0
        fnet = 0.0
        if N_net > 0:
            s = 0.0
            for elem in ['Ca', 'Na', 'Sr']:
                tid = self.type_map[elem]
                Cx = int(np.sum(ts == tid))
                if Cx == 0: continue
                nv = {'Ca':2, 'Na':1, 'Sr':2}[elem]
                sbs = self.SBS_X_O[elem]
                CN_O = cn_res.get((elem, 'O'), 0.0)
                s += Cx * nv * sbs * CN_O * nc
            fnet = s / N_net
        osi_angles, siosi_angles = self._compute_angles(cs, ts, bx)
        osi_mean = np.mean(osi_angles) if osi_angles else np.nan
        siosi_mean = np.mean(siosi_angles) if siosi_angles else np.nan
        p_si_links = 0; total_p_o = 0
        if len(p_idx) > 0:
            for p in p_idx:
                o_n = []
                for k in range(starts[p], starts[p+1]):
                    j = neighbors[k]
                    if ts[j] == self.type_map['O']:
                        dx = cs[p,0]-cs[j,0]; dy = cs[p,1]-cs[j,1]; dz = cs[p,2]-cs[j,2]
                        dx -= bx*np.round(dx/bx); dy -= bx*np.round(dy/bx); dz -= bx*np.round(dz/bx)
                        r = sqrt(dx*dx+dy*dy+dz*dz)
                        if r < p_o_cut: o_n.append(j)
                total_p_o += len(o_n)
                for o in o_n:
                    for k in range(starts[o], starts[o+1]):
                        j2 = neighbors[k]
                        if ts[j2] == self.type_map['Si']:
                            dx = cs[o,0]-cs[j2,0]; dy = cs[o,1]-cs[j2,1]; dz = cs[o,2]-cs[j2,2]
                            dx -= bx*np.round(dx/bx); dy -= bx*np.round(dy/bx); dz -= bx*np.round(dz/bx)
                            r = sqrt(dx*dx+dy*dy+dz*dz)
                            if r < si_o_cut:
                                p_si_links += 1
                                break
        return {
            'cn_fixed': cn_fixed, 'cn_auto': cn_auto, 'auto_cutoff': auto_cutoff_dict,
            'cn_extra': cn_extra, 'cn': cn_fixed,
            'qn_si': qn_si, 'qn_p': qn_p, 'qn_combined': qn_c,
            'bo': bo, 'nbo': nbo, 'fo': fo, 'to': to,
            'nc': nc, 'r_xx': rxx, 'modifier_preference': modifier_preference,
            'fnet': fnet, 'osi_mean': osi_mean, 'siosi_mean': siosi_mean,
            'p_si_links': p_si_links, 'total_p_o_bonds': total_p_o
        }

    def analyze_all(self, snapshots, n_cores=4):
        logger.info(f"Analyzing {len(snapshots)} snapshots using {n_cores} cores")
        if len(snapshots) <= 1 or n_cores <= 1:
            results = [self.analyze_snapshot(s) for s in snapshots]
        else:
            with Pool(processes=min(n_cores, len(snapshots))) as pool:
                results = list(tqdm(pool.imap(self.analyze_snapshot, snapshots),
                                    total=len(snapshots)))

        def mean_std(vals):
            vals = [v for v in vals if not np.isnan(v)]
            if not vals: return 0.0, 0.0
            if len(vals) == 1: return float(vals[0]), 0.0
            return float(np.mean(vals)), float(np.std(vals, ddof=1)/np.sqrt(len(vals)))

        agg = {}
        cn_fixed_dict = defaultdict(list); cn_auto_dict = defaultdict(list)
        auto_cutoff_dict = defaultdict(list)
        cn_extra_dict = defaultdict(lambda: defaultdict(list))
        for res in results:
            for pair, val in res['cn_fixed'].items(): cn_fixed_dict[pair].append(val)
            for pair, val in res['cn_auto'].items(): cn_auto_dict[pair].append(val)
            for pair, val in res['auto_cutoff'].items(): auto_cutoff_dict[pair].append(val)
            if 'cn_extra' in res:
                for pair, extra_vals in res['cn_extra'].items():
                    for rcut, val in extra_vals.items(): cn_extra_dict[pair][rcut].append(val)
        agg['cn_fixed'] = {p: mean_std(v) for p, v in cn_fixed_dict.items()}
        agg['cn_auto'] = {p: mean_std(v) for p, v in cn_auto_dict.items()}
        agg['auto_cutoff'] = {p: mean_std(v) for p, v in auto_cutoff_dict.items()}
        agg['cn_extra'] = {}
        for pair, rdict in cn_extra_dict.items():
            agg['cn_extra'][pair] = {r: mean_std(v) for r, v in rdict.items()}
        for key in ['cn', 'qn_si', 'qn_p', 'qn_combined']:
            d = defaultdict(list)
            for res in results:
                for k, v in res[key].items(): d[k].append(v)
            agg[key] = {k: mean_std(v) for k, v in d.items()}
        for key in ['bo', 'nbo', 'fo', 'to', 'nc', 'fnet', 'p_si_links', 'total_p_o_bonds']:
            agg[key] = mean_std([r[key] for r in results])
        for key in ['r_xx', 'modifier_preference']:
            d = defaultdict(list)
            for res in results:
                for k, v in res[key].items(): d[k].append(v)
            agg[key] = {k: mean_std(v) for k, v in d.items()}
        osi = [r['osi_mean'] for r in results if not np.isnan(r['osi_mean'])]
        siosi = [r['siosi_mean'] for r in results if not np.isnan(r['siosi_mean'])]
        agg['osi_mean'] = float(np.mean(osi)) if osi else np.nan
        agg['siosi_mean'] = float(np.mean(siosi)) if siosi else np.nan
        if agg['total_p_o_bonds'][0] > 0:
            agg['frac_p_si'] = agg['p_si_links'][0] / agg['total_p_o_bonds'][0]
        else:
            agg['frac_p_si'] = 0.0
        return agg

    def compute_cn_curves(self, excel_path, rmax=6.0, output_dir=None):
        xl = pd.ExcelFile(excel_path)
        rdf_sheets = [s for s in xl.sheet_names if s.startswith('RDF_')]
        if not rdf_sheets: return
        box = self.system.box; volume = box**3; counts = self.system.counts
        n_atom2 = {}
        for pair in self.ALL_PAIRS:
            ei, ej = pair
            n_atom2[f"{ei}-{ej}"] = counts.get(ei, 0) if ei == ej else counts.get(ej, 0)
        cn_results = {}
        for sheet in rdf_sheets:
            pair_name = sheet.replace('RDF_', '').replace('_', '-')
            df = pd.read_excel(excel_path, sheet_name=sheet)
            if 'r_A' not in df.columns or 'g_r' not in df.columns: continue
            r = df['r_A'].values; gr = df['g_r'].values
            n2 = n_atom2.get(pair_name, 0)
            if n2 == 0: continue
            rho = n2 / volume
            mask = r <= rmax
            r_masked = r[mask]; gr_masked = gr[mask]
            if len(r_masked) < 2: continue
            integrand = 4*np.pi*r_masked**2*rho*gr_masked
            cn = np.zeros_like(r_masked)
            for i in range(1, len(r_masked)):
                cn[i] = cn[i-1] + trapezoid(integrand[i-1:i+1], r_masked[i-1:i+1])
            cn_results[pair_name] = (r_masked, cn)
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            for pair_name, (r_masked, cn) in cn_results.items():
                df_cn = pd.DataFrame({'r_A': r_masked, 'CN': cn})
                sheet_name = f"CN_{pair_name.replace('-', '_')}"
                df_cn.to_excel(writer, sheet_name=sheet_name, index=False)
        if output_dir is None: output_dir = self.config.output_dir / "plots"
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        n_pairs = len(cn_results)
        if n_pairs == 0: return
        ncols = 4; nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
        axes = axes.flatten() if nrows*ncols > 1 else [axes]
        for idx, (pair, (r_cn, cn)) in enumerate(cn_results.items()):
            ax = axes[idx]
            ax.plot(r_cn, cn, color='#2980b9', linewidth=1.5)
            ax.set_xlabel('r (A)'); ax.set_ylabel('CN(r)'); ax.set_title(pair); ax.grid(True, alpha=0.3)
            elements = pair.split('-')
            if len(elements) == 2:
                cut = self.FIXED_CUTOFFS.get((elements[0], elements[1]), None)
                if cut is not None:
                    idx_cut = np.argmin(np.abs(r_cn - cut))
                    cn_cut = cn[idx_cut] if idx_cut < len(cn) else np.nan
                    ax.axvline(x=cut, color='#e74c3c', linestyle='--',
                               label=f'Cutoff={cut:.2f} A, CN={cn_cut:.2f}')
                    ax.legend(fontsize=8)
        for j in range(len(cn_results), len(axes)): axes[j].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_dir / 'CN_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def export_excel(self, results, path, energy_data=None, block_energy=None,
                     sensitivity=None, final_coords=None, final_types=None, box=None):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            meta_rows = [
                ['System', f"46.1SiO2-24.4Na2O-(26.9-x)CaO-2.6P2O5-xSrO"],
                ['x_mol_percent_SrO', self.config.x],
                ['Reference', 'Xiang & Du, Chem. Mater. 2011'],
                ['N_Atoms', self.system.N_ATOMS],
                ['N_Si', self.system.counts.get('Si', 0)],
                ['N_P', self.system.counts.get('P', 0)],
                ['N_Na', self.system.counts.get('Na', 0)],
                ['N_Ca', self.system.counts.get('Ca', 0)],
                ['N_Sr', self.system.counts.get('Sr', 0)],
                ['N_O', self.system.counts.get('O', 0)],
                ['Box_A', f'{self.system.box:.6f}'],
                ['Density_g_cm3', f'{self.system.effective_density:.6f}'],
                ['Seed', self.config.seed],
                ['Cutoff_A', self.config.cutoff],
                ['Wolf_alpha', self.config.wolf_alpha],
                ['Mixing_sweeps', self.config.mixing_sweeps],
                ['Final_sweeps', self.config.final_sweeps],
                ['Mode', 'Continue' if self.config.continue_mode else 'Full'],
                ['freud', 'Yes' if FREUD_AVAILABLE else 'No']
            ]
            if energy_data is not None and len(energy_data['total']) > 0:
                meta_rows.append(['Final_Energy_eV_per_atom', f'{energy_data["total"][-1]:.6f}'])
            if block_energy is not None:
                meta_rows.append(['Block_Avg_Energy_eV_per_atom', f'{block_energy["mean"]:.6f}'])
                meta_rows.append(['Block_Avg_Std_eV_per_atom', f'{block_energy["std"]:.6f}'])
            pd.DataFrame(meta_rows, columns=['Parameter', 'Value']).to_excel(
                writer, sheet_name='Metadata', index=False)

            cn_rows = []
            for pair in self.ALL_PAIRS:
                mean_fixed, se_fixed = results['cn_fixed'].get(pair, (0.0, 0.0))
                mean_auto, se_auto = results['cn_auto'].get(pair, (0.0, 0.0))
                cut_fixed = self.FIXED_CUTOFFS[pair]
                cut_auto, _ = results['auto_cutoff'].get(pair, (cut_fixed, 0.0))
                row = {
                    'Pair': f'{pair[0]}-{pair[1]}', 'CN_fixed': mean_fixed,
                    'CN_fixed_se': se_fixed, 'Cutoff_fixed': cut_fixed,
                    'CN_auto': mean_auto, 'CN_auto_se': se_auto, 'Cutoff_auto': cut_auto
                }
                if pair == ('Si','O'):
                    extra_data = results.get('cn_extra', {}).get(pair, {})
                    for rcut in self.si_o_cut_extra:
                        mean_val, se_val = extra_data.get(rcut, (0.0, 0.0))
                        row[f'CN_{rcut:.2f}'] = mean_val
                        row[f'CN_{rcut:.2f}_se'] = se_val
                cn_rows.append(row)
            pd.DataFrame(cn_rows).to_excel(writer, sheet_name='CN', index=False)

            N_SI = max(1, self.system.counts.get('Si', 1))
            qn_rows = [{'Qn': f'Q{n}', 'Count': results['qn_si'][n][0],
                        'Count_se': results['qn_si'][n][1],
                        'Percentage': results['qn_si'][n][0]/N_SI*100.0} for n in range(5)]
            pd.DataFrame(qn_rows).to_excel(writer, sheet_name='Qn_Si', index=False)
            if self.system.counts.get('P', 0) > 0:
                N_P = max(1, self.system.counts.get('P', 1))
                qn_rows = [{'Qn': f'Q{n}', 'Count': results['qn_p'][n][0],
                            'Count_se': results['qn_p'][n][1],
                            'Percentage': results['qn_p'][n][0]/N_P*100.0} for n in range(5)]
                pd.DataFrame(qn_rows).to_excel(writer, sheet_name='Qn_P', index=False)
            qn_rows = [{'Qn': f'Q{n}', 'Percent': results['qn_combined'][n][0],
                        'Percent_se': results['qn_combined'][n][1]} for n in range(5)]
            pd.DataFrame(qn_rows).to_excel(writer, sheet_name='Qn_Combined', index=False)
            N_O = max(1, self.system.counts.get('O', 1))
            pd.DataFrame({
                'Type': ['BO','TO','NBO','FO'],
                'Count': [results['bo'][0], results['to'][0], results['nbo'][0], results['fo'][0]],
                'Count_se': [results['bo'][1], results['to'][1], results['nbo'][1], results['fo'][1]],
                'Percentage': [results['bo'][0]/N_O*100, results['to'][0]/N_O*100,
                               results['nbo'][0]/N_O*100, results['fo'][0]/N_O*100]
            }).to_excel(writer, sheet_name='O_speciation', index=False)
            pd.DataFrame({'NC': [results['nc'][0]], 'NC_se': [results['nc'][1]]}).to_excel(
                writer, sheet_name='NC', index=False)
            rxx_rows = [{'Element': e, 'R_XX': results['r_xx'][e][0],
                         'R_XX_se': results['r_xx'][e][1]}
                        for e in ['Ca', 'Na', 'Sr'] if e in results['r_xx']]
            if rxx_rows:
                pd.DataFrame(rxx_rows).to_excel(writer, sheet_name='R_XX', index=False)
            pref_rows = [{'Ratio': k, 'Value': v[0], 'Value_se': v[1]}
                         for k, v in results.get('modifier_preference', {}).items()]
            if pref_rows:
                pd.DataFrame(pref_rows).to_excel(writer, sheet_name='Modifier_Preference', index=False)
            pd.DataFrame({
                'Fnet_new': [results['fnet'][0]], 'Fnet_new_se': [results['fnet'][1]],
                'SBS_Ca': [32.0], 'SBS_Na': [20.0], 'SBS_Sr': [32.0]
            }).to_excel(writer, sheet_name='Fnet_New', index=False)
            if self.system.counts.get('P', 0) > 0:
                pd.DataFrame({
                    'Total_P-O_bonds': [results['total_p_o_bonds'][0]],
                    'P-O-Si_bonds': [results['p_si_links'][0]],
                    'Fraction_P-O-Si': [results['frac_p_si']]
                }).to_excel(writer, sheet_name='Si-O-P', index=False)

            # Sensitivity Analysis
            if sensitivity:
                pd.DataFrame(sensitivity).to_excel(writer, sheet_name='Sensitivity', index=False)

            # Energy data
            if energy_data is not None and len(energy_data['total']) > 0:
                total_steps = len(energy_data['total'])
                step_interval = max(1, total_steps // 10000)
                indices = np.arange(0, total_steps, step_interval)
                df_energy = pd.DataFrame({
                    'Step': indices + 1,
                    'Total_Energy_eV_per_atom': np.array(energy_data['total'])[indices],
                    'Short_Range_eV_per_atom': np.array(energy_data['short'])[indices],
                    'Coulomb_eV_per_atom': np.array(energy_data['coul'])[indices]
                })
                df_energy.to_excel(writer, sheet_name='Energy', index=False)

            if final_coords is not None and final_types is not None and box is not None:
                for pair in self.ALL_PAIRS:
                    ei, ej = pair
                    cut = self.FIXED_CUTOFFS[pair]
                    cn, peak, r, gr = self.compute_rdf(
                        final_coords, final_types, ei, ej, box, cut, rmax=6.0, nbins=400)
                    df_rdf = pd.DataFrame({'r_A': r, 'g_r': gr})
                    df_rdf.to_excel(writer, sheet_name=f'RDF_{ei}-{ej}', index=False)
                osi_angles, siosi_angles = self._compute_angles(final_coords, final_types, box)
                if osi_angles:
                    bins = np.linspace(60, 180, 26)
                    hist_osi, edges_osi = np.histogram(osi_angles, bins=bins)
                    centers_osi = (edges_osi[:-1] + edges_osi[1:]) / 2.0
                    df_ang = pd.DataFrame({
                        'O-Si-O_angle_bin_center': centers_osi,
                        'O-Si-O_frequency': hist_osi
                    })
                    if siosi_angles:
                        hist_siosi, edges_siosi = np.histogram(siosi_angles, bins=bins)
                        centers_siosi = (edges_siosi[:-1] + edges_siosi[1:]) / 2.0
                        df_ang['Si-O-Si_angle_bin_center'] = centers_siosi
                        df_ang['Si-O-Si_frequency'] = hist_siosi
                    df_ang.to_excel(writer, sheet_name='Angle_Distribution', index=False)
        self.compute_cn_curves(path, rmax=6.0, output_dir=self.config.output_dir / "plots")

    def plot_results(self, results, outdir):
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(4, 5, figsize=(24, 18)); axes = axes.flatten()
        for idx, pair in enumerate(self.ALL_PAIRS):
            if idx >= len(axes): break
            _, _, r, gr = self.compute_rdf(self.system.coords, self.system.type_indices,
                                           pair[0], pair[1], self.system.box,
                                           self.FIXED_CUTOFFS[pair])
            axes[idx].plot(r, gr, color='#2980b9', lw=1)
            axes[idx].axvline(self.FIXED_CUTOFFS[pair], color='#e74c3c', ls='--',
                              label=f'Cutoff={self.FIXED_CUTOFFS[pair]:.2f} A')
            axes[idx].set_xlabel('r (A)'); axes[idx].set_ylabel('g(r)')
            axes[idx].set_title(f'{pair[0]}-{pair[1]}')
            axes[idx].legend(fontsize=8); axes[idx].grid(alpha=0.3)
        for extra in range(len(self.ALL_PAIRS), len(axes)): axes[extra].axis('off')
        plt.tight_layout()
        plt.savefig(outdir / 'rdf.png', dpi=300, bbox_inches='tight'); plt.close()
        colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#27ae60']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        means = [results['qn_si'][n][0] for n in range(5)]
        errs = [results['qn_si'][n][1] for n in range(5)]
        bars = ax1.bar([f'Q{n}' for n in range(5)], means, yerr=errs,
                       color=colors, edgecolor='white', capsize=5)
        N_SI = max(1, self.system.counts.get('Si', 1))
        for bar, n in zip(bars, range(5)):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(means)*0.02,
                     f'{means[n]/N_SI*100:.1f}%', ha='center')
        ax1.set_title('Si Qn'); ax1.set_ylabel('Count'); ax1.grid(alpha=0.3, axis='y')
        if self.system.counts.get('P', 0) > 0:
            means = [results['qn_p'][n][0] for n in range(5)]
            errs = [results['qn_p'][n][1] for n in range(5)]
            bars = ax2.bar([f'Q{n}' for n in range(5)], means, yerr=errs,
                           color=colors, edgecolor='white', capsize=5)
            N_P = max(1, self.system.counts.get('P', 1))
            for bar, n in zip(bars, range(5)):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(means)*0.02,
                         f'{means[n]/N_P*100:.1f}%', ha='center')
            ax2.set_title('P Qn'); ax2.set_ylabel('Count'); ax2.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(outdir / 'qn.png', dpi=300, bbox_inches='tight'); plt.close()
        logger.info(f"Plots saved to {outdir}")


# ========================= SENSITIVITY ANALYZER =========================
class SensitivityAnalyzer:
    def __init__(self, system, config):
        self.system = system; self.config = config; self.pot = system.potential
        self.A_mat = np.zeros((6, 6), dtype=np.float64)
        self.F_mat = np.zeros((6, 6), dtype=np.float64)
        self.C_mat = np.zeros((6, 6), dtype=np.float64)
        self.R_HARD = np.full((6, 6), self.pot.r_hard_default, dtype=np.float64)
        for (e1, e2), val in self.pot.r_hard_by_pair.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.R_HARD[t1, t2] = val; self.R_HARD[t2, t1] = val
        for (e1, e2), p in self.pot.buck_params.items():
            t1 = self.system.type_map[e1]; t2 = self.system.type_map[e2]
            self.A_mat[t1, t2] = p['A']; self.F_mat[t1, t2] = p['F']; self.C_mat[t1, t2] = p['C']
            if t1 != t2:
                self.A_mat[t2, t1] = p['A']; self.F_mat[t2, t1] = p['F']; self.C_mat[t2, t1] = p['C']
        self.type_Z = self.system.get_type_Z()

    def run(self):
        logger.info("=" * 70)
        logger.info("SENSITIVITY ANALYSIS")
        logger.info("=" * 70)
        res = []
        for cut in self.config.test_cutoffs:
            for alpha in self.config.test_alphas:
                nl = NeighborList(self.system, cut, 1.0)
                n, s = nl.neighbors, nl.starts
                U = total_energy(self.system.coords, self.system.charges,
                                 self.system.type_indices, self.type_Z,
                                 self.A_mat, self.F_mat, self.C_mat, self.R_HARD,
                                 n, s, self.system.box, cut, alpha,
                                 self.pot.zbl_a0, self.pot.zbl_c, self.pot.zbl_d)
                self_energy = -KE * (alpha / sqrt(pi)) * np.sum(self.system.charges**2)
                U_tot = (U + self_energy) / self.system.N_ATOMS
                res.append({'Cutoff': cut, 'Alpha': alpha, 'Energy': U_tot})
                logger.info(f"  Cutoff={cut:.1f} A, alpha={alpha:.2f}: E={U_tot:.6f} eV/atom")
        return res


# ========================= BLOCK AVERAGING =========================
def compute_block_energy(energy_log, block_size=5000):
    """Compute block-averaged energy with standard deviation."""
    if len(energy_log) < block_size * 3:
        return {'mean': float(np.mean(energy_log)) if energy_log else 0.0, 'std': 0.0}
    prod_energy = np.array(energy_log)
    n_blocks = len(prod_energy) // block_size
    if n_blocks < 3:
        return {'mean': float(np.mean(prod_energy)), 'std': 0.0}
    block_avgs = [np.mean(prod_energy[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
    mean_e = float(np.mean(block_avgs))
    std_e = float(np.std(block_avgs, ddof=1) / np.sqrt(n_blocks))
    return {'mean': mean_e, 'std': std_e}


# ========================= SUMMARY PRINTER =========================
def print_summary(results, x_val):
    print("\n" + "=" * 70)
    print(f"  ANALYSIS SUMMARY - 45S5 + {x_val} mol% SrO")
    print("=" * 70)
    print("\n--- Bond Lengths (A) [Sim vs Paper] ---")
    for (e1, e2), bl in sorted(results.get('bond_lengths', {}).items()):
        if np.isnan(bl): continue
        paper = PAPER_BOND_LENGTHS.get((e1, e2), PAPER_BOND_LENGTHS.get((e2, e1), None))
        ps = f"{paper:.2f}" if paper else "  -  "
        flag = "  OK" if paper and abs(bl - paper) < 0.1 else ("  DIFF" if paper else "")
        print(f"  {e1}-{e2:2s}: {bl:.3f}  (paper: {ps}){flag}")
    print("\n--- Modifier CN [Sim vs Paper] ---")
    for mod, cn in results.get('modifier_cn', {}).items():
        paper = PAPER_CN.get(mod, None)
        ps = f"{paper:.1f}" if paper else "  -  "
        flag = "  OK" if paper and abs(cn - paper) < 0.5 else "  DIFF"
        print(f"  {mod}-O: {cn:.2f}  (paper: {ps}){flag}")
    print("\n--- Network Connectivity [Sim vs Paper] ---")
    for k, v in results.get('nc', {}).items():
        paper = PAPER_NC.get(k, None)
        ps = f"{paper:.2f}" if paper else "  -  "
        flag = "  OK" if paper and abs(v - paper) < 0.2 else "  DIFF"
        print(f"  NC ({k:8s}): {v:.3f}  (paper: {ps}){flag}")
    print("\n--- Oxygen Speciation ---")
    tot = sum(results.get('oxygen_speciation', {}).values())
    for k, v in results.get('oxygen_speciation', {}).items():
        print(f"  {k:4s}: {v:5d}  ({v/tot*100:5.1f}%)")
    print("\n--- Clustering R_obs/R_hom ---")
    for mod, (rxx, cn_obs, cn_hom) in results.get('rxx', {}).items():
        if rxx is not None and not np.isnan(rxx):
            print(f"  {mod}: R={rxx:.3f}  (CN_obs={cn_obs:.2f}, CN_hom={cn_hom:.2f})")
    print("\n--- Qn Distribution (%) ---")
    print("  Si:  " + "  ".join(f"{k}:{v:.1f}" for k, v in results.get('qn_si', {}).items()))
    print("  P:   " + "  ".join(f"{k}:{v:.1f}" for k, v in results.get('qn_p', {}).items()))
    print("=" * 70 + "\n")


# ========================= CLI =========================
app = typer.Typer(help="45S5/Sr Bioglass NVT MC v10.0 - ALL-IN-ONE (Simulation + Analysis)")

@app.command()
def run(
    input_file: Path = typer.Argument(..., help="Input XYZ file generated by struture.py"),
    seed: Optional[int] = typer.Option(None, help="Random seed. If omitted, read from XYZ comment."),
    density: Optional[float] = typer.Option(None, help="Optional density override in g/cm^3."),
    cores: int = typer.Option(4, help="Number of CPU cores for analysis."),
    final_sweeps: Optional[int] = typer.Option(None, help="Override production MC sweeps."),
    cutoff: float = typer.Option(10.0, help="Effective cutoff in Angstrom."),
    wolf_alpha: float = typer.Option(0.25, help="Wolf damping parameter alpha."),
    continue_mode: bool = typer.Option(False, "--continue",
                                       help="Skip mixing/annealing. Use for relaxing pre-equilibrated structure.")
):
    try:
        n_header, comment, meta = parse_xyz_header(input_file)
        x = meta.get('x', 0)
        if seed is None: seed = meta.get('seed', 42)
        if seed is None: seed = 42
        if density is None: density = meta.get('density', None)
        box_override = meta.get('box', None)
        output_dir = Path(f"Bioglass_x{x}_N{n_header}_seed{seed}")

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
        logger.info(f"File       : {input_file}")
        logger.info(f"x          : {x} mol% SrO")
        logger.info(f"N atoms    : {n_header}")
        logger.info(f"Seed       : {seed}")
        logger.info(f"Mode       : {'CONTINUE' if continue_mode else 'FULL PROTOCOL'}")
        logger.info(f"Output dir : {config.output_dir}")

        # ---- SIMULATION ----
        system = GlassSystem(config)
        sim = MCSimulator(system, config)
        sim.run()

        energy_data = {'total': sim.energy_log, 'short': sim.short_log, 'coul': sim.coul_log}
        final_coords = system.coords.copy()
        final_types = system.type_indices.copy()
        box = system.box

        # ---- BLOCK AVERAGING ----
        block_energy = compute_block_energy(sim.energy_log, config.block_size)
        logger.info(f"Block Energy: mean={block_energy['mean']:.6f}, std={block_energy['std']:.6f}")

        # ---- SAVE FINAL STRUCTURE ----
        xyz_path = config.output_dir / "final_structure.xyz"
        with open(xyz_path, 'w') as f:
            f.write(f"{system.N_ATOMS}\n")
            f.write(
                f"Final Xiang-Du 2011, x={config.x} mol% SrO, "
                f"N={system.N_ATOMS}, rho={system.effective_density:.4f} g/cm3, "
                f"box={system.box:.4f} A, seed={config.seed}\n"
            )
            for i in range(system.N_ATOMS):
                elem = system.type_to_elem[system.type_indices[i]]
                x_, y_, z_ = system.coords[i]
                f.write(f"{elem:2s} {x_:12.6f} {y_:12.6f} {z_:12.6f}\n")
        logger.info(f"Final XYZ saved to {xyz_path}")

        # ---- SAVE ENERGY LOG ----
        energy_path = config.output_dir / "energy_log.csv"
        with open(energy_path, 'w') as f:
            f.write("Step,Total_eV_per_atom,ShortRange_eV_per_atom,Coulomb_eV_per_atom\n")
            for i, (e, s, c) in enumerate(zip(sim.energy_log, sim.short_log, sim.coul_log)):
                f.write(f"{i * sim.log_freq},{e:.6f},{s:.6f},{c:.6f}\n")
        logger.info(f"Energy log saved to {energy_path}")

        # ---- ANALYSIS ----
        snapshots = sim.snapshots
        if not snapshots:
            snapshots = [{'coords': final_coords, 'types': final_types, 'box': box}]

        analyzer = Analyzer(system, config)
        results = analyzer.analyze_all(snapshots, n_cores=cores)
        results['block_energy'] = block_energy

        # Bond lengths from final structure
        bond_lengths = {}
        for (e1, e2) in analyzer.ALL_PAIRS:
            ti, tj = analyzer.type_map[e1], analyzer.type_map[e2]
            r, gr = analyzer.compute_rdf(final_coords, final_types, ti, tj,
                                         box, analyzer.FIXED_CUTOFFS[(e1, e2)],
                                         rmax=6.0, nbins=400)[2:]
            bl = analyzer._find_first_minimum(r, gr, (e1, e2))
            if bl is not None:
                bond_lengths[(e1, e2)] = bl
        results['bond_lengths'] = bond_lengths

        # Modifier CN
        modifier_cn = {}
        for mod in ['Na', 'Ca', 'Sr']:
            if system.counts.get(mod, 0) > 0:
                pair = (mod, 'O')
                cutoff = analyzer.FIXED_CUTOFFS[pair]
                ti, tj = analyzer.type_map[mod], analyzer.type_map['O']
                r, gr = analyzer.compute_rdf(final_coords, final_types, ti, tj,
                                             box, cutoff, rmax=cutoff+2.0)[:2]
                modifier_cn[mod] = coordination_number(r, gr, cutoff, system.counts.get('O', 0), box)
        results['modifier_cn'] = modifier_cn

        # ---- SENSITIVITY ANALYSIS ----
        sens = SensitivityAnalyzer(system, config).run()
        results['sensitivity'] = sens

        # ---- EXPORT ----
        excel_path = config.output_dir / "simulation_results.xlsx"
        analyzer.export_excel(results, excel_path, energy_data=energy_data,
                              block_energy=block_energy, sensitivity=sens,
                              final_coords=final_coords, final_types=final_types, box=box)
        analyzer.plot_results(results, config.output_dir / "plots")

        with open(config.output_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f)

        # ---- PRINT SUMMARY ----
        print_summary(results, x)

        logger.info(f"[OK] All results saved to {config.output_dir}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()