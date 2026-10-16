#!/usr/bin/env python3
"""
printer_stiffness.py - Frame stiffness analysis for 3D-printer CAD models (STEP),
with improvement recommendations written by a LOCAL LLM (Ollama or any
OpenAI-compatible server such as llama.cpp / vLLM / LM Studio).

What it does
------------
1. Reads a STEP assembly (every solid body) with gmsh / OpenCASCADE.
2. Classifies each body:  beam (extrusion, rail, rod) | plate | block (bracket, motor...)
   and derives section properties of beams directly from the CAD mass properties.
3. Finds which bodies touch, and builds a 3-D frame model:
     beams  -> Euler-Bernoulli beam elements split at every contact
     joints -> short flexible "joint link" elements (bolted joints are not rigid)
     blocks/plates -> hub node at the centroid with spokes to what they touch
4. Solves load cases that matter for print quality:
     accel_X / accel_Y  - inertial load of the moving gantry at the top of the frame
     torsion_Z          - twisting couple at the top (CoreXY racking)
     gravity_Z          - self weight + moving mass (sag)
   plus the lowest natural frequencies (ringing / input-shaper relevance).
5. Runs "what-if" studies so recommendations are backed by numbers:
     upsizing each of the weakest members, stiffening all joints (gussets),
     adding diagonal braces / shear panels on each side and on the top.
6. Sends the numeric summary to your local LLM, which writes a prioritised
   improvement plan.  Everything is saved in a report folder.

No cloud AI is used.  The physics is done in this script; the LLM only
interprets the numbers.

Units: mm, N, MPa (N/mm^2), tonne (modal), s.  Reported stiffness in N/um.

Usage
-----
    python printer_stiffness.py my_printer.step
    python printer_stiffness.py my_printer.step --model llama3.1:8b --accel 15000
    python printer_stiffness.py my_printer.step --no-ai            # numbers only
    python printer_stiffness.py --help

Limitations (also printed in the report): this is a reduced-order frame model,
not a solid FEA.  Joint stiffness and torsion constants of slotted extrusions
are estimates; use the results to RANK weaknesses and compare design options,
not as absolute certified values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__version__ = "1.1.0"

# ----------------------------------------------------------------------------
# Materials  (E, nu in MPa; rho in kg/mm^3).  Built-in fallback; library/*.json overrides and extends.
# ----------------------------------------------------------------------------
MATERIALS = {
    "aluminum": dict(E=69000.0, nu=0.33, rho=2.70e-6, desc="Aluminium 6063-T5 (extrusions, plates)"),
    "steel":    dict(E=200000.0, nu=0.29, rho=7.85e-6, desc="Steel (linear rails, rods, screws)"),
    "petg":     dict(E=2000.0, nu=0.38, rho=1.27e-6, printed=True, desc="PETG print"),
    "pla":      dict(E=3500.0, nu=0.36, rho=1.24e-6, printed=True, desc="PLA print"),
    "abs":      dict(E=2200.0, nu=0.35, rho=1.05e-6, printed=True, desc="ABS print"),
    "asa":      dict(E=2000.0, nu=0.35, rho=1.07e-6, printed=True, desc="ASA print"),
    "pc_cf":    dict(E=6000.0, nu=0.35, rho=1.25e-6, printed=True, desc="PC-CF print"),
    "carbon":   dict(E=70000.0, nu=0.30, rho=1.60e-6, desc="Carbon fibre tube / plate"),
    "mdf":      dict(E=3500.0, nu=0.25, rho=0.75e-6, desc="MDF / wood panel"),
    "acrylic":  dict(E=3000.0, nu=0.37, rho=1.19e-6, desc="Acrylic panel"),
}

# Built-in name rules, used when no library is found (regex on lower-case body name).
DEFAULT_PART_RULES = [
    dict(match=r"screw|bolt|\bnut\b|t-?nut|washer|insert|\bbelt|cable|wire|spring|\bm[2-8]x|din.?9[0-9]{2}|iso.?[0-9]{4}",
         role="fastener", ignore=True),
    dict(match=r"mgn|hgr|hiwin|linear.?rail|\brail|\brod\b|smooth.?rod|shaft|lead.?screw|ball.?screw|steel|stainless",
         role="steel part", material="steel"),
    dict(match=r"petg", material="petg"), dict(match=r"\babs\b", material="abs"),
    dict(match=r"\basa\b", material="asa"), dict(match=r"\bpla\b", material="pla"),
    dict(match=r"\bcf\b|carbon.?fib|nylon.?cf|pc.?cf|pa.?cf", material="pc_cf"),
    dict(match=r"carbon.?tube|cf.?tube", material="carbon"),
    dict(match=r"\bmdf\b|plywood|wood", material="mdf"), dict(match=r"acrylic|perspex|plexi", material="acrylic"),
    dict(match=r"printed|\bprint\b|_print|-print", role="printed part", material="asa"),
]
DEFAULT_PRINT_PROFILES = {
    "solid": dict(walls=99, top_bottom=99, infill=1.0, line_width_mm=0.45, source="solid"),
}

# 2020 aluminium extrusion used for brace what-ifs (overwritten by the library's measured 2020 profile)
BRACE_2020 = dict(A=175.0, I=7021.0, J=2200.0, E=69000.0, G=26000.0)


def load_library(path):
    """Merge every *.json in the library folder (alphabetical; later files override keys)."""
    lib = dict(materials={}, print_profiles={}, printed_model={}, part_rules=[], profiles={}, blocks={}, editions={})
    if not path or not os.path.isdir(path):
        return None
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(".json"))
    for f in files:
        with open(os.path.join(path, f), encoding="utf-8") as fh:
            d = json.load(fh)
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                lib.setdefault(k, {}).update(v)
            elif isinstance(v, list):
                lib.setdefault(k, []).extend(v)
    lib["files"] = files
    return lib


def apply_library(lib):
    if not lib:
        return
    for k, m in lib["materials"].items():
        MATERIALS[k] = dict(E=float(m["E"]), nu=float(m.get("nu", 0.33)), rho=float(m["rho"]),
                            printed=bool(m.get("printed", False)), creep=m.get("creep", ""), desc=m.get("desc", ""))
    p = lib["profiles"].get("2020")
    if p:
        BRACE_2020.update(A=p["A"], I=p["I_small"], J=p["J"])


MATERIAL_WORDS = [  # a material named in the body name beats the library's role rule
    (r"pet-?cf|petg-?cf", "pet_cf"), (r"pa-?cf|nylon-?cf|pa\d+-?cf", "pa_cf"), (r"pc-?cf", "pc_cf"),
    (r"\bpetg\b", "petg"), (r"\basa\b", "asa"), (r"\babs\b", "abs"), (r"\bpla\b", "pla"), (r"\btpu\b", "tpu"),
]


def named_material(n):
    for rx, mat in MATERIAL_WORDS:
        if re.search(rx, n) and mat in MATERIALS:
            return mat
    return None


def classify(name, args, lib):
    """Return dict(role, material, print_profile, ignore) for a body name."""
    n = re.sub(r"[_]+", " ", name.lower())
    for rx, mat in args.material_rules:                 # CLI overrides first
        if re.search(rx, n):
            return dict(role="user rule", material=mat, print_profile=args.print_profile if MATERIALS[mat].get("printed") else None, ignore=False)
    if args.ignore and re.search(args.ignore, n, re.I):
        return dict(role="ignored (--ignore)", material=None, print_profile=None, ignore=True)
    rules = lib["part_rules"] if lib and lib["part_rules"] else DEFAULT_PART_RULES
    for r in rules:
        if re.search(r["match"], n):
            if r.get("ignore"):
                return dict(role=r.get("role", "ignored"), material=None, print_profile=None, ignore=True)
            mat = named_material(n) or r.get("material", args.default_material)
            if mat not in MATERIALS:
                mat = args.default_material
            prof = None
            if MATERIALS[mat].get("printed"):
                prof = r.get("print_profile") or args.print_profile
            return dict(role=r.get("role", ""), material=mat, print_profile=prof, ignore=False,
                        structural=r.get("structural", True))
    mat = named_material(n) or args.default_material
    return dict(role="unmatched", material=mat,
                print_profile=args.print_profile if MATERIALS[mat].get("printed") else None, ignore=False)


def printed_factor(profile, ext, lib):
    """Effective-modulus factor for a printed part (walls + infill + layer anisotropy). Estimate."""
    profs = dict(DEFAULT_PRINT_PROFILES, **(lib["print_profiles"] if lib else {}))
    pp = profs.get(profile) if profile else None
    if not pp:
        return 1.0
    pm = (lib or {}).get("printed_model", {}) or {}
    expo, layer = float(pm.get("infill_exponent", 1.6)), float(pm.get("layer_factor", 0.8))
    t = max(float(min(ext)), 0.5)
    shell = min(1.0, 2.0 * pp.get("walls", 3) * pp.get("line_width_mm", 0.45) / t)
    infill = float(pp.get("infill", 0.2))
    return (shell + (1 - shell) * infill ** expo) * layer


def body_label(raw: str, tag: int) -> str:
    """gmsh STEP label 'Shapes/<assembly>/<label>/<label>' -> '<label>'. Labels may contain '/' (e.g.
    'base member (front/back)', 'NEMA17 A/B motor'), so look for a repeated tail instead of the last segment."""
    segs = [s for s in re.split(r"/", raw or "")]
    if segs and segs[0].lower() == "shapes":
        segs = segs[1:]
    best = ""
    for i in range(1, len(segs)):
        tail = "/".join(segs[i:])
        k = len(tail.split("/"))
        if i - k >= 0 and "/".join(segs[i - k:i]) == tail:
            best = tail
            break
    if not best and segs:
        best = segs[-1]
    best = best.strip()
    if not best or re.search(r"open cascade|step translator|^solid$|^body$|^part$", best, re.I):
        best = f"solid_{tag}"
    return best


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class Part:
    idx: int
    tag: int
    name: str
    material: str
    kind: str = "block"            # beam | plate | block
    volume: float = 0.0            # mm^3
    mass: float = 0.0              # kg
    c: np.ndarray = None           # centroid
    e: np.ndarray = None           # 3x3, columns: e1 (long), e2, e3
    ext: np.ndarray = None         # extents along e1,e2,e3
    tmin: float = 0.0
    tmax: float = 0.0
    L: float = 0.0
    A: float = 0.0
    Iy: float = 0.0                # about e2 (bending deflection along e3)
    Iz: float = 0.0                # about e3
    J: float = 0.0
    fill: float = 1.0
    bbox: tuple = None
    label: str = ""                 # label without FreeCAD's 001/002 duplicate suffix
    instance: str = ""              # full label (one part instance may be several solids)
    role: str = ""
    print_profile: str = None
    E: float = 0.0                  # effective modulus used (printed parts reduced for walls/infill)
    nu: float = 0.33
    rho: float = 0.0
    e_factor: float = 1.0
    lib_match: str = None           # matched library profile / catalog block
    structural: bool = True         # False: kept for the model check, left out of the frame model
    lib_note: str = ""
    nodes: list = field(default_factory=list)   # node ids (beam: sorted along axis)
    node_t: list = field(default_factory=list)
    conn_t: list = field(default_factory=list)  # axis positions of connections (beams)


@dataclass
class Model:
    xyz: np.ndarray                # (n,3)
    el_n: np.ndarray               # (m,2)
    E: np.ndarray
    G: np.ndarray
    A: np.ndarray
    Iy: np.ndarray
    Iz: np.ndarray
    J: np.ndarray
    yref: np.ndarray               # (m,3) reference vector for local y
    owner: list                    # per element: ("member", part_idx) | ("joint", i, j) | ("brace", label)
    node_mass: np.ndarray          # kg per node
    support: np.ndarray            # node ids fixed
    top: np.ndarray                # node ids loaded
    parts: list
    excluded: list
    floating: list


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def log(msg: str, quiet=False):
    if not quiet:
        print(msg, flush=True)


def rect_props(b: float, h: float):
    """Solid rectangle b (local y) x h (local z)."""
    b, h = max(b, 1e-3), max(h, 1e-3)
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    a, s = max(b, h), min(b, h)
    J = a * s ** 3 * (1.0 / 3.0 - 0.21 * (s / a) * (1.0 - s ** 4 / (12.0 * a ** 4)))
    return A, Iy, Iz, J


def perpendicular(v: np.ndarray) -> np.ndarray:
    a = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(v, a)
    return p / np.linalg.norm(p)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# ----------------------------------------------------------------------------
# 1. Geometry: STEP -> parts + contacts
# ----------------------------------------------------------------------------
def up_matrix(up: str) -> np.ndarray:
    """Rotation mapping model coordinates to internal coordinates with Z up."""
    up = up.lower()
    if up == "z":
        return np.eye(3)
    if up == "y":   # model Y up  -> internal Z ; model Z -> internal -Y
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    if up == "x":
        return np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    if up == "-z":
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    raise ValueError("--up must be x, y, z or -z")


def match_profile(p: Part, lib, args):
    """Replace CAD-estimated section properties of a beam by the library's (measured) profile when the
    section envelope and area match: e.g. V-slot 2020/2040, MGN9/MGN12 rails."""
    if not lib or not lib["profiles"] or args.no_library_sections or MATERIALS[p.material].get("printed"):
        return
    sec = sorted([float(p.ext[1]), float(p.ext[2])])
    best = None
    for name, pr in lib["profiles"].items():
        env = sorted(pr["envelope"])
        if abs(env[0] - sec[0]) > 0.8 or abs(env[1] - sec[1]) > 0.8:
            continue
        ratio = p.A / pr["A"]
        if not (0.75 <= ratio <= 1.30):
            continue
        score = abs(1 - ratio) + abs(env[0] - sec[0]) / 10 + abs(env[1] - sec[1]) / 10
        if best is None or score < best[0]:
            best = (score, name, pr, ratio)
    if not best:
        return
    _, name, pr, ratio = best
    # ext[1] (e2) is the larger section dimension, ext[2] (e3) the smaller.
    # Iy = int(s3^2) -> second moment along the SMALL dimension; Iz along the LARGE one.
    p.A, p.Iy, p.Iz, p.J = pr["A"], pr["I_small"], pr["I_large"], pr["J"]
    if pr.get("material") in MATERIALS and p.role in ("unmatched", "extrusion", "linear rail", "lead screw", "steel part"):
        if p.material != pr["material"]:
            p.material = pr["material"]
            p.rho, p.nu, p.E, p.e_factor = MATERIALS[p.material]["rho"], MATERIALS[p.material]["nu"], MATERIALS[p.material]["E"], 1.0
            p.mass = p.volume * p.rho
    p.lib_match = name
    p.lib_note = f"CAD area {ratio*100:.0f}% of library"


def match_block(p: Part, lib):
    if not lib or not lib["blocks"]:
        return
    bb = sorted(float(x) for x in (p.bbox[1] - p.bbox[0]))
    for name, bl in lib["blocks"].items():
        for key, note in (("bbox", ""), ("body_bbox", "main body")):
            ref = bl.get(key)
            if ref and all(abs(a - b) <= max(1.0, 0.03 * b) for a, b in zip(bb, sorted(ref))):
                p.lib_match = name
                p.lib_note = note
                if p.role == "unmatched" and bl.get("material") in MATERIALS:
                    p.material = bl["material"]
                    p.rho, p.nu, p.E, p.e_factor = (MATERIALS[p.material]["rho"], MATERIALS[p.material]["nu"],
                                                    MATERIALS[p.material]["E"], 1.0)
                    p.mass = p.volume * p.rho
                    p.role = "catalog part"
                return


def shape_rule(p: Part, lib):
    """Recognise unnamed bodies by shape (e.g. GT2 belts exported as thin 6 mm loops)."""
    if not lib or not lib.get("shape_rules") or p.role not in ("unmatched",) or not p.instance.startswith("solid_"):
        return None
    dims = sorted(float(x) for x in (p.bbox[1] - p.bbox[0]))
    fill = p.volume / max(dims[0] * dims[1] * dims[2], 1e-9)
    for r in lib["shape_rules"]:
        lo, hi = r.get("one_dim_between", [0, 1e9])
        if not any(lo <= d <= hi for d in dims):
            continue
        if fill <= r.get("max_fill", 1.0) or dims[0] <= r.get("or_thinner_than", -1):
            return r
    return None


def merge_collinear(parts, contacts, args, quiet=False):
    """Fuse bars that the CAD split into several collinear solids (same label or unnamed, same section, touching
    end to end) into one bar. Also report collinear pieces of one bar that leave a gap (e.g. a stretched rail
    whose middle was cut through a bolt hole). Returns parts, contacts, merges, gaps."""
    beams = [p for p in parts if p.kind == "beam"]
    uf = UnionFind(len(parts))
    gaps = []

    def same_family(a, b):
        if a.instance == b.instance:
            return True
        return a.instance.startswith("solid_") and b.instance.startswith("solid_")

    for ii, a in enumerate(beams):
        for b in beams[ii + 1:]:
            if a.material != b.material or not same_family(a, b):
                continue
            if abs(float(a.e[:, 0] @ b.e[:, 0])) < 0.9995:
                continue
            sa, sb = sorted(a.ext[1:]), sorted(b.ext[1:])
            if abs(sa[0] - sb[0]) > 0.3 or abs(sa[1] - sb[1]) > 0.3 or abs(a.volume / a.L - b.volume / b.L) > 0.05 * a.volume / a.L:
                continue
            ax = a.e[:, 0]
            d = b.c - a.c
            if np.linalg.norm(d - (d @ ax) * ax) > 0.5:
                continue
            a0, a1 = a.c @ ax + a.tmin, a.c @ ax + a.tmax
            sgn = 1.0 if (b.e[:, 0] @ ax) > 0 else -1.0
            bt = sorted([b.c @ ax + sgn * b.tmin, b.c @ ax + sgn * b.tmax])
            gap = max(bt[0] - a1, a0 - bt[1])
            if gap <= args.contact_tol + 0.05:
                uf.union(a.idx, b.idx)
            elif gap <= 400:
                lo, hi = (a1, bt[0]) if bt[0] >= a1 else (bt[1], a0)
                fill = [p.name for p in parts if p is not a and p is not b
                        and lo - 1 <= min(p.c @ ax + p.tmin, p.c @ ax + p.tmax) and max(p.c @ ax + p.tmin, p.c @ ax + p.tmax) <= hi + 1
                        and np.linalg.norm((p.c - a.c) - ((p.c - a.c) @ ax) * ax) < max(a.ext[1], a.ext[2])]
                gaps.append(dict(a=a.idx, b=b.idx, gap=round(gap, 1), fillers=fill))
    groups = {}
    for p in parts:
        groups.setdefault(uf.find(p.idx), []).append(p)
    merges = []
    new_parts, remap = [], {}
    for root, ps in sorted(groups.items()):
        if len(ps) == 1:
            p = ps[0]
        else:
            p0 = ps[0]
            ax = p0.e[:, 0]
            V = sum(q.volume for q in ps)
            m = sum(q.mass for q in ps)
            c = sum(q.c * q.volume for q in ps) / V
            ends = []
            for q in ps:
                sgn = 1.0 if (q.e[:, 0] @ ax) > 0 else -1.0
                ends += [q.c @ ax + sgn * q.tmin, q.c @ ax + sgn * q.tmax]
            t0, t1 = min(ends) - c @ ax, max(ends) - c @ ax
            p = Part(**{**p0.__dict__})
            p.volume, p.mass, p.c = V, m, c
            p.tmin, p.tmax = float(t0), float(t1)
            p.L = float(t1 - t0)
            p.ext = np.array([p.L, p0.ext[1], p0.ext[2]])
            if not p0.lib_match:
                p.A = V / p.L
            p.bbox = (np.min([q.bbox[0] for q in ps], axis=0), np.max([q.bbox[1] for q in ps], axis=0))
            p.lib_note = (p0.lib_note + "; " if p0.lib_note else "") + f"merged from {len(ps)} solids"
            if p0.instance.startswith("solid_"):
                p.name = p.instance = p.label = "+".join(q.instance for q in ps)
            merges.append(dict(name=p.name, pieces=[q.name for q in ps], length=round(p.L, 1)))
        for q in ps:
            remap[q.idx] = len(new_parts)
        p.idx = len(new_parts)
        new_parts.append(p)
    cmap = {}
    for i, j, pts in contacts:
        a, b = remap[i], remap[j]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        cmap.setdefault(key, []).extend(pts)
    new_contacts = [(a, b, pts) for (a, b), pts in cmap.items()]
    gaps = [g for g in gaps if remap[g["a"]] != remap[g["b"]]]       # pieces later fused into one bar
    for g in gaps:
        g["a"], g["b"] = new_parts[remap[g["a"]]].name, new_parts[remap[g["b"]]].name
    if merges:
        log(f"      merged {sum(len(m['pieces']) for m in merges)} split solids into {len(merges)} bars", quiet)
    return new_parts, new_contacts, merges, gaps


def _cache_path(args, lib):
    import hashlib
    st = os.stat(args.step)
    libsig = []
    if lib and os.path.isdir(args.library):
        for f in lib["files"]:
            libsig.append((f, os.stat(os.path.join(args.library, f)).st_mtime))
    key = repr((__version__, os.path.abspath(args.step), st.st_size, st.st_mtime, args.up, args.contact_tol,
                args.contact_spacing, args.min_volume, args.slender, args.ignore, args.material_rules,
                args.print_profile, args.default_material, args.no_library_sections, args.extrusion_torsion_factor,
                libsig))
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = os.environ.get("PS_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "printer-stiffness")
    return os.path.join(d, f"{os.path.basename(args.step)}.{h}.pkl")


def load_parts_cached(args, quiet=False, lib=None):
    """load_parts() with a disk cache: reading a big STEP and finding contacts takes minutes, the analysis
    seconds - re-runs with other loads or AI settings reuse the geometry. --no-cache to force a re-read."""
    import pickle
    path = None if getattr(args, "no_cache", False) else _cache_path(args, lib)
    if path and os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                raw = pickle.load(f)
            parts = [Part(**d) for d in raw["parts"]]
            log(f"[1-2/6] Geometry from cache ({path})", quiet)
            return parts, raw["excluded"], raw["contacts"]
        except Exception:
            pass
    data = load_parts(args, quiet, lib)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            parts, excluded, contacts = data
            # plain dicts: Part may live in __main__ or in the imported module depending on the entry point
            with open(path, "wb") as f:
                pickle.dump(dict(parts=[dict(p.__dict__) for p in parts], excluded=excluded, contacts=contacts), f)
        except Exception:
            pass
    return data


def load_parts(args, quiet=False, lib=None, keep_open=False):
    try:
        import gmsh
    except ImportError:
        sys.exit("gmsh is not installed.  Run:  pip install gmsh")

    R = up_matrix(args.up)
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 1)
    try:
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    except Exception:
        pass
    occ = gmsh.model.occ
    t0 = time.time()
    log(f"[1/6] Reading {args.step} ...", quiet)
    occ.importShapes(args.step, highestDimOnly=True)
    occ.synchronize()
    vols = [t for d, t in gmsh.model.getEntities(3)]
    if not vols:
        gmsh.finalize()
        sys.exit("No solid bodies found in the STEP file (surfaces only?). Export solids from your CAD.")
    log(f"      {len(vols)} solid bodies read in {time.time()-t0:.1f}s", quiet)

    raw_names = {}
    for t in vols:
        try:
            nm = gmsh.model.getEntityName(3, t) or ""
        except Exception:
            nm = ""
        raw_names[t] = body_label(nm, t)
    counts, seen = {}, {}
    for t in vols:
        counts[raw_names[t]] = counts.get(raw_names[t], 0) + 1

    parts, excluded = [], []
    for t in vols:
        base = raw_names[t]
        seen[base] = seen.get(base, 0) + 1
        name = f"{base} #{seen[base]}" if counts[base] > 1 else base
        label = re.sub(r"\d{3}$", "", base).strip()       # FreeCAD duplicate suffix 001, 002 ...
        V = occ.getMass(3, t)
        if V <= 0:
            excluded.append((name, "zero volume"))
            continue
        cl = classify(base, args, lib)
        if cl["ignore"]:
            excluded.append((name, f"ignored: {cl['role']}"))
            continue
        if V < args.min_volume:
            excluded.append((name, f"smaller than --min-volume ({V:.0f} mm3)"))
            continue
        mat = cl["material"] if cl["material"] in MATERIALS else args.default_material
        c = R @ np.array(occ.getCenterOfMass(3, t))
        Jm = R @ np.array(occ.getMatrixOfInertia(3, t)).reshape(3, 3) @ R.T
        Jm = 0.5 * (Jm + Jm.T)
        w, vec = np.linalg.eigh(Jm)
        off = np.abs(Jm - np.diag(np.diag(Jm))).max()
        if off <= 1e-6 * np.abs(np.diag(Jm)).max():
            vec = np.eye(3)            # axis-aligned body: keep global axes (avoids 45 deg ambiguity of square sections)
        # snap near-axis-aligned eigenvectors (cleans up degenerate cases)
        for k in range(3):
            v = vec[:, k]
            j = int(np.argmax(np.abs(v)))
            if abs(v[j]) > 0.999:
                v2 = np.zeros(3)
                v2[j] = math.copysign(1.0, v[j])
                vec[:, k] = v2
        # points for extents
        bnd = gmsh.model.getBoundary([(3, t)], combined=False, oriented=False, recursive=True)
        pts = [gmsh.model.getValue(0, abs(tag), []) for d, tag in bnd if d == 0]
        bb = occ.getBoundingBox(3, t)
        corners = np.array([[x, y, z] for x in (bb[0], bb[3]) for y in (bb[1], bb[4]) for z in (bb[2], bb[5])])
        pts = np.array(pts) if pts else corners
        P = (R @ pts.T).T
        Cg = (R @ corners.T).T
        proj = (P - c) @ vec
        ext = proj.max(0) - proj.min(0)
        # if axis aligned, bbox extents are exact (handles round parts)
        aligned = np.all(np.max(np.abs(vec), axis=0) > 0.999)
        if aligned:
            projc = (Cg - c) @ vec
            ext = projc.max(0) - projc.min(0)
            proj = projc
        order = np.argsort(-ext)
        vec, ext, proj = vec[:, order], ext[order], proj[:, order]
        # Section axes: square/round sections have two equal inertia eigenvalues, so eigh can return any rotation
        # (a 2020 bar then looks 21.5 x 21.5 or 28 x 28). Pick the in-plane rotation with the smallest
        # bounding rectangle, preferring global axes.
        e1 = vec[:, 0]
        u = vec[:, 1] - (vec[:, 1] @ e1) * e1
        u /= np.linalg.norm(u)
        w_ = np.cross(e1, u)
        cands = list(np.radians(np.arange(0.0, 90.0, 0.5)))
        for g in np.eye(3):
            gp = g - (g @ e1) * e1
            if np.linalg.norm(gp) > 0.5:
                gp /= np.linalg.norm(gp)
                cands.append(math.atan2(gp @ w_, gp @ u) % (math.pi / 2))
        Q = P - c
        best = None
        for th in cands:
            a_ = math.cos(th) * u + math.sin(th) * w_
            b_ = -math.sin(th) * u + math.cos(th) * w_
            ea, eb = np.ptp(Q @ a_), np.ptp(Q @ b_)
            area = ea * eb
            if best is None or area < best[0] * (1 - 1e-6):
                best = (area, a_, b_, ea, eb)
        _, a_, b_, ea, eb = best
        if eb > ea:
            a_, b_, ea, eb = b_, -a_, eb, ea
        vec = np.column_stack([e1, a_, b_])
        for k in range(3):
            v = vec[:, k]
            j = int(np.argmax(np.abs(v)))
            if abs(v[j]) > 0.9999:
                v2 = np.zeros(3); v2[j] = math.copysign(1.0, v[j]); vec[:, k] = v2
        if np.linalg.det(vec) < 0:
            vec[:, 2] *= -1
        if np.all(np.max(np.abs(vec), axis=0) > 0.9999):
            proj = (Cg - c) @ vec           # axis aligned: bbox is exact (also for round parts)
        else:
            proj = (P - c) @ vec
        ext = proj.max(0) - proj.min(0)
        if ext[0] < ext[1]:                 # keep e1 the longest
            vec = vec[:, [1, 0, 2]]; vec[:, 2] *= -1; proj = proj[:, [1, 0, 2]]; ext = ext[[1, 0, 2]]
        if ext[1] < ext[2]:
            vec = vec[:, [0, 2, 1]]; vec[:, 2] *= -1; proj = proj[:, [0, 2, 1]]; ext = ext[[0, 2, 1]]
        p = Part(idx=len(parts), tag=t, name=name, material=mat, volume=V,
                 mass=V * MATERIALS[mat]["rho"], c=c, e=vec, ext=ext, label=label, instance=base,
                 role=cl["role"], print_profile=cl["print_profile"], structural=cl.get("structural", True))
        p.rho, p.nu = MATERIALS[mat]["rho"], MATERIALS[mat]["nu"]
        p.e_factor = printed_factor(p.print_profile, ext, lib) if MATERIALS[mat].get("printed") else 1.0
        p.E = MATERIALS[mat]["E"] * p.e_factor
        p.tmin, p.tmax = float(proj[:, 0].min()), float(proj[:, 0].max())
        lo = Cg.min(0); hi = Cg.max(0)
        p.bbox = (lo, hi)
        L, b, h = ext
        slender = L / max(b, 1e-6)
        if slender >= args.slender and L > 20:
            p.kind = "beam"
            p.L = L
            A = V / L
            # Section second moments from the inertia tensor without subtracting V*L^2/12 (ill-conditioned on
            # real CAD with holes/chamfers):  J11 = L(Iy+Iz),  J22 - J33 = L(Iy - Iz)
            J11 = vec[:, 0] @ Jm @ vec[:, 0]
            J22 = vec[:, 1] @ Jm @ vec[:, 1]
            J33 = vec[:, 2] @ Jm @ vec[:, 2]
            Iy = (J11 + J22 - J33) / (2 * L)    # about e2: integral of s3^2 (deflection along e3)
            Iz = (J11 - J22 + J33) / (2 * L)    # about e3: integral of s2^2
            # rectangular envelope: local y (e2) width b, local z (e3) height h -> Iy = b h^3/12
            Ar, Iyr, Izr, Jr = rect_props(b, h)
            if not (0 < Iy <= 1.5 * Iyr):
                Iy = Iyr * min(1.0, A / Ar)
            if not (0 < Iz <= 1.5 * Izr):
                Iz = Izr * min(1.0, A / Ar)
            fill = min(1.0, A / Ar)
            Jsv = A ** 4 / (4 * math.pi ** 2 * (Iy + Iz))      # Saint-Venant compact-section estimate
            Jsv = min(Jsv, Iy + Iz, Jr * 1.05)
            if fill < 0.7:   # slotted / hollow extrusion: open slots kill torsion stiffness
                Jsv *= args.extrusion_torsion_factor
            p.A, p.Iy, p.Iz, p.J, p.fill = A, Iy, Iz, Jsv, fill
            match_profile(p, lib, args)
        elif h / max(b, 1e-6) < 0.2:
            p.kind = "plate"
        else:
            p.kind = "block"
        if p.kind != "beam":
            match_block(p, lib)
        sr = shape_rule(p, lib)
        if sr and sr.get("ignore"):
            excluded.append((name, f"ignored by shape: {sr['name']}"))
            continue
        parts.append(p)
    log(f"      {len(parts)} structural bodies, {len(excluded)} ignored", quiet)

    # -------- contacts
    log(f"[2/6] Finding contacts (tolerance {args.contact_tol} mm) ...", quiet)
    t0 = time.time()
    tol = args.contact_tol
    raw_bb = {p.idx: occ.getBoundingBox(3, p.tag) for p in parts}
    contacts = []   # (i, j, [points internal coords])
    n = len(parts)
    for a in range(n):
        ba = raw_bb[a]
        for b_ in range(a + 1, n):
            bb_ = raw_bb[b_]
            if (ba[0] - tol > bb_[3] or bb_[0] - tol > ba[3] or ba[1] - tol > bb_[4] or
                    bb_[1] - tol > ba[4] or ba[2] - tol > bb_[5] or bb_[2] - tol > ba[5]):
                continue
            try:
                d = occ.getDistance(3, parts[a].tag, 3, parts[b_].tag)
            except Exception:
                continue
            if d[0] > tol:
                continue
            p1 = np.array(d[1:4]); p2 = np.array(d[4:7])
            pc = R @ (0.5 * (p1 + p2))
            # overlap region of (expanded) bounding boxes -> spread points along long contacts
            lo = np.maximum(np.array(ba[:3]), np.array(bb_[:3])) - tol
            hi = np.minimum(np.array(ba[3:]), np.array(bb_[3:])) + tol
            lo_i, hi_i = R @ lo, R @ hi
            lo_i, hi_i = np.minimum(lo_i, hi_i), np.maximum(lo_i, hi_i)
            span = hi_i - lo_i
            k = int(np.argmax(span))
            pts = [pc]
            if span[k] > args.contact_spacing * 1.5:
                m = int(span[k] // args.contact_spacing) + 1
                pts = []
                for s in np.linspace(lo_i[k] + 0.5 * span[k] / m, hi_i[k] - 0.5 * span[k] / m, m):
                    q = pc.copy()
                    q[k] = s
                    pts.append(q)
            contacts.append((a, b_, pts))
    log(f"      {len(contacts)} touching pairs found in {time.time()-t0:.1f}s", quiet)
    if not keep_open:
        gmsh.finalize()
    return parts, excluded, contacts


# ----------------------------------------------------------------------------
# 2. Frame model
# ----------------------------------------------------------------------------
def link_section(p: Part):
    if p.kind == "beam":
        return p.ext[1], p.ext[2]
    if p.kind == "plate":
        return min(p.ext[1], 50.0), p.ext[2]
    return p.ext[1], p.ext[2]


def build_model(parts, excluded, contacts, args, quiet=False) -> Model:
    log("[3/6] Building frame model ...", quiet)
    keep = [p for p in parts if p.structural]
    if len(keep) < len(parts):
        drop = [p for p in parts if not p.structural]
        excluded = excluded + [(p.name, f"not structural ({p.role})") for p in drop]
        remap = {p.idx: k for k, p in enumerate(keep)}
        contacts = [(remap[i], remap[j], pts) for i, j, pts in contacts if i in remap and j in remap]
        for k, p in enumerate(keep):
            p.idx = k
        parts = keep
    xyz = []
    # axis parameters for beams
    beam_t = {p.idx: [p.tmin, p.tmax] for p in parts if p.kind == "beam"}
    conn_pts = []   # (i, j, point)
    for i, j, pts in contacts:
        for q in pts:
            conn_pts.append((i, j, q))
            for k in (i, j):
                p = parts[k]
                if p.kind == "beam":
                    beam_t[k].append(float(np.clip((q - p.c) @ p.e[:, 0], p.tmin, p.tmax)))
    # create nodes
    for p in parts:
        if p.kind == "beam":
            ts = sorted(beam_t[p.idx])
            merged = [ts[0]]
            for t in ts[1:]:
                if t - merged[-1] > args.node_merge:
                    merged.append(t)
            if merged[-1] < p.tmax - 1e-9 and p.tmax - merged[-1] <= args.node_merge:
                merged[-1] = p.tmax
            # always keep a mid node on long unconnected spans for modal accuracy
            refined = [merged[0]]
            for t in merged[1:]:
                base = refined[-1]
                gap = t - base
                nsub = int(gap // args.max_element)
                for s in range(1, nsub + 1):
                    refined.append(base + gap * s / (nsub + 1))
                refined.append(t)
            p.node_t = refined
            p.conn_t = merged
            p.nodes = []
            for t in refined:
                p.nodes.append(len(xyz))
                xyz.append(p.c + p.e[:, 0] * t)
        else:
            p.nodes = [len(xyz)]
            p.node_t = [0.0]
            xyz.append(p.c.copy())
    xyz = np.array(xyz)
    nn = len(xyz)
    uf = UnionFind(nn)

    def node_of(p: Part, q):
        if p.kind != "beam":
            return p.nodes[0]
        t = float(np.clip((q - p.c) @ p.e[:, 0], p.tmin, p.tmax))
        k = int(np.argmin(np.abs(np.array(p.node_t) - t)))
        return p.nodes[k]

    els = []   # dicts
    for p in parts:
        if p.kind != "beam":
            continue
        Emod, G = p.E, p.E / (2 * (1 + p.nu))
        for a, b in zip(p.nodes[:-1], p.nodes[1:]):
            els.append(dict(n=(a, b), E=Emod, G=G, A=p.A, Iy=p.Iy, Iz=p.Iz, J=p.J,
                            y=p.e[:, 1].copy(), owner=("member", p.idx)))

    seen_links = set()
    for i, j, q in conn_pts:
        pi, pj = parts[i], parts[j]
        na, nb = node_of(pi, q), node_of(pj, q)
        if na == nb:
            continue
        d = np.linalg.norm(xyz[na] - xyz[nb])
        if d < 0.5:
            uf.union(na, nb)
            continue
        key = (min(na, nb), max(na, nb))
        if key in seen_links:
            continue
        seen_links.add(key)
        Ei, Ej = pi.E, pj.E
        Emod = min(Ei, Ej) * args.joint_factor
        si, sj = link_section(pi), link_section(pj)
        b, h = si if si[0] * si[1] <= sj[0] * sj[1] else sj
        A, Iy, Iz, Jt = rect_props(b, h)
        v = xyz[nb] - xyz[na]
        els.append(dict(n=(na, nb), E=Emod, G=Emod / 2.6, A=A, Iy=Iy, Iz=Iz, J=Jt,
                        y=perpendicular(v / np.linalg.norm(v)), owner=("joint", min(i, j), max(i, j))))

    # merge coincident nodes
    rep = np.array([uf.find(k) for k in range(nn)])
    uniq, newid = np.unique(rep, return_inverse=True)
    xyz2 = np.zeros((len(uniq), 3))
    cnt = np.zeros(len(uniq))
    for k in range(nn):
        xyz2[newid[k]] += xyz[k]
        cnt[newid[k]] += 1
    xyz2 /= cnt[:, None]
    for p in parts:
        p.nodes = [int(newid[k]) for k in p.nodes]
    els = [dict(e, n=(int(newid[e["n"][0]]), int(newid[e["n"][1]]))) for e in els]
    els = [e for e in els if e["n"][0] != e["n"][1]]
    xyz = xyz2
    nn = len(xyz)

    # masses
    node_mass = np.zeros(nn)
    for p in parts:
        if p.kind == "beam" and len(p.nodes) > 1:
            ts = np.array(p.node_t)
            w = np.zeros(len(ts))
            seg = np.diff(ts)
            w[:-1] += seg / 2
            w[1:] += seg / 2
            w = w / w.sum() if w.sum() > 0 else np.ones(len(ts)) / len(ts)
            for k, nid in enumerate(p.nodes):
                node_mass[nid] += p.mass * w[k]
        else:
            node_mass[p.nodes[0]] += p.mass

    # connectivity -> find supports / floating groups
    uf2 = UnionFind(nn)
    for e in els:
        uf2.union(*e["n"])
    used = np.zeros(nn, bool)
    for e in els:
        used[list(e["n"])] = True
    for p in parts:
        if p.kind != "beam":
            used[p.nodes[0]] = True
    z = xyz[:, 2]
    zmin, zmax = z[used].min(), z[used].max()
    H = max(zmax - zmin, 1.0)
    stol = args.support_tol if args.support_tol is not None else max(5.0, 0.03 * H)
    support = np.where(used & (z <= zmin + stol))[0]
    sup_roots = {uf2.find(k) for k in support}
    keep = np.array([used[k] and uf2.find(k) in sup_roots for k in range(nn)])
    floating = sorted({p.name for p in parts if not keep[p.nodes[0]]})
    if floating:
        log(f"      WARNING: {len(floating)} bodies are not connected to the supported frame "
            f"(check --contact-tol): {', '.join(floating[:8])}{' ...' if len(floating) > 8 else ''}", quiet)

    # compact: drop unused / floating nodes
    keep_ids = np.where(keep)[0]
    remap = -np.ones(nn, int)
    remap[keep_ids] = np.arange(len(keep_ids))
    els = [dict(e, n=(int(remap[e["n"][0]]), int(remap[e["n"][1]])))
           for e in els if keep[e["n"][0]] and keep[e["n"][1]]]
    xyz = xyz[keep_ids]
    node_mass = node_mass[keep_ids]
    for p in parts:
        p.nodes = [int(remap[k]) for k in p.nodes if remap[k] >= 0]
    support = remap[support[keep[support]]]
    z = xyz[:, 2]
    zmin, zmax = z.min(), z.max()
    H = max(zmax - zmin, 1.0)
    if args.top_z is not None:
        top = np.where(z >= args.top_z)[0]
    else:
        top = np.where(z >= zmax - args.top_band * H)[0]
    if len(top) == 0:
        top = np.array([int(np.argmax(z))])

    m = len(els)
    model = Model(
        xyz=xyz,
        el_n=np.array([e["n"] for e in els], int).reshape(m, 2),
        E=np.array([e["E"] for e in els]), G=np.array([e["G"] for e in els]),
        A=np.array([e["A"] for e in els]), Iy=np.array([e["Iy"] for e in els]),
        Iz=np.array([e["Iz"] for e in els]), J=np.array([e["J"] for e in els]),
        yref=np.array([e["y"] for e in els]).reshape(m, 3),
        owner=[e["owner"] for e in els], node_mass=node_mass,
        support=np.array(sorted(set(support.tolist())), int), top=top,
        parts=parts, excluded=excluded, floating=floating)
    nb = sum(1 for p in parts if p.kind == "beam")
    nj = sum(1 for o in model.owner if o[0] == "joint")
    log(f"      {len(xyz)} nodes, {m} elements ({nb} beam bodies, {nj} joint links), "
        f"{len(model.support)} support nodes, {len(model.top)} loaded top nodes", quiet)
    return model


# ----------------------------------------------------------------------------
# 3. FE solver
# ----------------------------------------------------------------------------
def element_matrices(model: Model, scale=None):
    """Return list of (dof indices, 12x12 global stiffness) for all elements."""
    xyz = model.xyz
    out = []
    E, G, A, Iy, Iz, J = model.E, model.G, model.A, model.Iy, model.Iz, model.J
    if scale is not None:
        E = E * scale.get("E", 1.0)
        G = G * scale.get("E", 1.0)
        A = A * scale.get("A", 1.0)
        Iy = Iy * scale.get("I", 1.0)
        Iz = Iz * scale.get("I", 1.0)
        J = J * scale.get("I", 1.0)
    for k, (a, b) in enumerate(model.el_n):
        d = xyz[b] - xyz[a]
        L = np.linalg.norm(d)
        ex = d / L
        y = model.yref[k] - (model.yref[k] @ ex) * ex
        if np.linalg.norm(y) < 1e-6:
            y = perpendicular(ex)
        ey = y / np.linalg.norm(y)
        ez = np.cross(ex, ey)
        Rm = np.vstack([ex, ey, ez])
        kl = np.zeros((12, 12))
        EA, GJ = E[k] * A[k] / L, G[k] * J[k] / L
        kl[0, 0] = kl[6, 6] = EA; kl[0, 6] = kl[6, 0] = -EA
        kl[3, 3] = kl[9, 9] = GJ; kl[3, 9] = kl[9, 3] = -GJ
        ei = E[k] * Iz[k]   # bending in local xy plane
        a1, b1, c1, d1 = 12 * ei / L ** 3, 6 * ei / L ** 2, 4 * ei / L, 2 * ei / L
        for (r, c_, v) in [(1, 1, a1), (1, 5, b1), (1, 7, -a1), (1, 11, b1), (5, 5, c1), (5, 7, -b1),
                           (5, 11, d1), (7, 7, a1), (7, 11, -b1), (11, 11, c1)]:
            kl[r, c_] = kl[c_, r] = v
        ei = E[k] * Iy[k]   # bending in local xz plane
        a1, b1, c1, d1 = 12 * ei / L ** 3, 6 * ei / L ** 2, 4 * ei / L, 2 * ei / L
        for (r, c_, v) in [(2, 2, a1), (2, 4, -b1), (2, 8, -a1), (2, 10, -b1), (4, 4, c1), (4, 8, b1),
                           (4, 10, d1), (8, 8, a1), (8, 10, b1), (10, 10, c1)]:
            kl[r, c_] = kl[c_, r] = v
        T = np.zeros((12, 12))
        for q in range(4):
            T[3 * q:3 * q + 3, 3 * q:3 * q + 3] = Rm
        kg = T.T @ kl @ T
        dofs = np.r_[6 * a:6 * a + 6, 6 * b:6 * b + 6]
        out.append((dofs, kg))
    return out


def assemble(n_nodes, emats):
    rows, cols, vals = [], [], []
    for dofs, kg in emats:
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(kg.ravel())
    N = 6 * n_nodes
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsc()
    return K


def free_dofs(model: Model, support_mode: str):
    N = 6 * len(model.xyz)
    fixed = []
    for s in model.support:
        fixed += list(range(6 * s, 6 * s + 3)) if support_mode == "pinned" else list(range(6 * s, 6 * s + 6))
    mask = np.ones(N, bool)
    mask[fixed] = False
    return np.where(mask)[0]


def load_cases(model: Model, args):
    xyz, top = model.xyz, model.top
    nn = len(xyz)
    moving_kg = args.moving_mass
    F = moving_kg * args.accel / 1000.0          # N  (kg * m/s^2)
    cases = {}
    for name, ax in (("accel_X", 0), ("accel_Y", 1)):
        f = np.zeros(6 * nn)
        f[6 * top + ax] = F / len(top)
        cases[name] = dict(f=f, total=F, dir=ax, kind="lin",
                           desc=f"{moving_kg} kg moving mass x {args.accel/1000:.0f} m/s^2 in {'XY'[ax]} at top of frame")
    # torsion about vertical axis through the centroid of top nodes
    ctr = xyz[top].mean(0)
    r = xyz[top] - ctr
    rr = np.hypot(r[:, 0], r[:, 1])
    f = np.zeros(6 * nn)
    ok = rr > 1e-6
    if ok.sum() >= 2:
        arm = float(np.median(rr[ok]))
        T = F * arm                                   # N*mm  (inertial load acting off-centre)
        w = rr[ok] ** 2
        fk = T / w.sum()
        t_idx = top[ok]
        f[6 * t_idx + 0] = -fk * r[ok, 1]
        f[6 * t_idx + 1] = fk * r[ok, 0]
        cases["torsion_Z"] = dict(f=f, total=T, dir=None, kind="rot", ctr=ctr, ok=t_idx,
                                  desc=f"twisting couple {T/1000:.1f} N*m at top (moving load x {arm:.0f} mm arm)")
    f = np.zeros(6 * nn)
    f[6 * np.arange(nn) + 2] = -model.node_mass * 9.81
    f[6 * top + 2] += -moving_kg * 9.81 / len(top)
    cases["gravity_Z"] = dict(f=f, total=float(-f[2::6].sum()), dir=2, kind="lin",
                              desc="self weight + moving mass, downward")
    return cases


def solve_all(model: Model, args, cases, scale_mask=None, extra=None):
    """Solve all load cases. scale_mask: (bool mask, scale dict). extra: list of extra elements."""
    emats = element_matrices(model)
    if scale_mask is not None:
        mask, sc = scale_mask
        emats2 = element_matrices(model, scale=sc)
        emats = [emats2[k] if mask[k] else emats[k] for k in range(len(emats))]
    if extra:
        emats = emats + extra
    nn = len(model.xyz)
    K = assemble(nn, emats)
    fd = free_dofs(model, args.support)
    Kf = K[fd][:, fd].tocsc()
    diag = Kf.diagonal()
    reg = 1e-10 * float(np.max(diag))
    Kf = Kf + sp.diags(np.full(len(fd), reg))     # tiny stabiliser for free-spinning rotations
    lu = spla.splu(Kf)
    results = {}
    for name, c in cases.items():
        u = np.zeros(6 * nn)
        u[fd] = lu.solve(c["f"][fd])
        results[name] = u
    return results, emats, K, fd


def summarise_case(model: Model, case, u):
    top = model.top
    disp = u.reshape(-1, 6)[:, :3]
    if case["kind"] == "lin":
        ax = case["dir"]
        d = float(np.mean(np.abs(disp[top, ax])))
        stiff = abs(case["total"]) / max(d, 1e-12)                # N/mm
        return dict(defl_um=d * 1000, max_um=float(np.max(np.linalg.norm(disp[top], axis=1))) * 1000,
                    stiffness_N_per_um=stiff / 1000)
    ctr, idx = case["ctr"], case["ok"]
    r = model.xyz[idx] - ctr
    rr2 = r[:, 0] ** 2 + r[:, 1] ** 2
    th = float(np.sum(r[:, 0] * disp[idx, 1] - r[:, 1] * disp[idx, 0]) / np.sum(rr2))   # rad
    kt = case["total"] / max(abs(th), 1e-15)                       # N*mm/rad
    return dict(rot_deg=math.degrees(th), max_um=float(np.max(np.linalg.norm(disp[top], axis=1))) * 1000,
                stiffness_Nm_per_deg=kt / 1000 * math.pi / 180)


def energy_breakdown(model: Model, emats, u):
    """Strain energy grouped by member / joint. Returns dict key->fraction."""
    groups = {}
    tot = 0.0
    for k, (dofs, kg) in enumerate(emats[:len(model.owner)]):
        ue = u[dofs]
        e = 0.5 * float(ue @ kg @ ue)
        o = model.owner[k]
        key = o if o[0] != "member" else ("member", o[1])
        groups[key] = groups.get(key, 0.0) + e
        tot += e
    for k in range(len(model.owner), len(emats)):
        dofs, kg = emats[k]
        ue = u[dofs]
        tot += 0.5 * float(ue @ kg @ ue)
    return {k: v / tot for k, v in groups.items()} if tot > 0 else {}


def modal(model: Model, K, fd, emats, nmodes, quiet=False):
    nn = len(model.xyz)
    m = model.node_mass * 1e-3                       # tonnes
    m = np.maximum(m, 1e-9)
    Mdiag = np.zeros(6 * nn)
    for k in range(3):
        Mdiag[k::6] = m
    rot = m * 25.0                                   # small rotary inertia (5 mm radius of gyration)
    for k in range(3, 6):
        Mdiag[k::6] = np.maximum(rot, 1e-9)
    Kf = K[fd][:, fd].tocsc()
    Mf = sp.diags(Mdiag[fd]).tocsc()
    k = min(nmodes, len(fd) - 2)
    frame_nodes = sorted({n for p in model.parts for n in p.nodes      # metal bars = the frame
                          if p.kind == "beam" and not MATERIALS[p.material].get("printed")
                          and p.role not in ("lead screw",) and p.E > 20000})
    def solve_modes(k):
        try:
            w2, phi = spla.eigsh(Kf, k=k, M=Mf, sigma=-1.0, which="LM")
        except Exception as ex:
            log(f"      modal analysis failed: {ex}", quiet)
            return []
        order = np.argsort(w2)
        out = []
        for i in order:
            f = math.sqrt(max(w2[i], 0)) / (2 * math.pi)
            u = np.zeros(6 * nn)
            u[fd] = phi[:, i]
            disp = u.reshape(-1, 6)[:, :3]
            top = model.top
            dt = disp[top]
            tx, ty, tz = (float(np.sum(np.abs(dt[:, a]))) for a in range(3))
            ctr = model.xyz[top].mean(0)
            r = model.xyz[top] - ctr
            tw = float(np.sum(np.abs(r[:, 0] * dt[:, 1] - r[:, 1] * dt[:, 0]) / np.maximum(np.hypot(r[:, 0], r[:, 1]), 1)))
            lab = max((("X sway", tx), ("Y sway", ty), ("Z bounce", tz), ("twist", tw)), key=lambda t: t[1])[0]
            en = energy_breakdown(model, emats, u)
            # kinetic-energy share of the frame (extrusions + rails) vs local sub-assemblies (bed on spacers, motors...)
            ke = model.node_mass * np.sum(disp ** 2, axis=1)
            tot = float(ke.sum()) or 1.0
            frame_frac = float(ke[frame_nodes].sum()) / tot if len(frame_nodes) else 1.0
            per_part = {}
            for p in model.parts:
                if p.nodes:
                    per_part[p.name] = float(ke[list(set(p.nodes))].sum())
            dominant = max(per_part, key=per_part.get) if per_part else ""
            local = frame_frac < 0.25
            if local:
                lab = f"local: {dominant}"
            elif frame_frac < 0.5:
                lab = f"{lab}, coupled with {dominant}"
            out.append(dict(freq_hz=f, shape=lab, energy=en, u=u, frame_share=frame_frac, local=local, dominant=dominant))

        return out

    # Soft sub-assemblies (bed on silicone spacers, motors, the bed carriers) can fill the lowest modes;
    # keep asking for more until a few frame modes show up.
    out = []
    while True:
        out = solve_modes(k)
        nf = sum(1 for m in out if not m.get("local"))
        if nf >= 4 or k >= min(200, len(fd) - 2):
            break
        k = min(k * 2, 200, len(fd) - 2)
    if k > nmodes:
        log(f"      computed {k} modes to get past {sum(1 for m in out if m.get('local'))} local sub-assembly modes", quiet)
    return out


# ----------------------------------------------------------------------------
# 4. Analysis driver incl. what-ifs
# ----------------------------------------------------------------------------
def owner_label(model: Model, o):
    P = model.parts
    if o[0] == "member":
        return P[o[1]].name
    if o[0] == "joint":
        return f"joint: {P[o[1]].name} <-> {P[o[2]].name}"
    return f"brace: {o[1]}"


def brace_elements(model: Model, a, b, label):
    xyz = model.xyz
    d = xyz[b] - xyz[a]
    L = np.linalg.norm(d)
    if L < 50 or np.hypot(d[0], d[1]) < 50 or a == b:
        return None     # degenerate (corners coincide in plan view)
    ex = d / L
    y = perpendicular(ex)
    tmp = Model(xyz=xyz, el_n=np.array([[a, b]]), E=np.array([BRACE_2020["E"]]), G=np.array([BRACE_2020["G"]]),
                A=np.array([BRACE_2020["A"]]), Iy=np.array([BRACE_2020["I"]]), Iz=np.array([BRACE_2020["I"]]),
                J=np.array([BRACE_2020["J"]]), yref=y[None, :], owner=[("brace", label)], node_mass=None,
                support=None, top=None, parts=None, excluded=None, floating=None)
    return element_matrices(tmp)


def corner_nodes(model: Model):
    xyz = model.xyz
    lo, hi = xyz.min(0), xyz.max(0)
    sup, top = model.support, model.top
    def near(cands, x, y, z):
        d = np.linalg.norm(xyz[cands] - np.array([x, y, z]), axis=1)
        return int(cands[int(np.argmin(d))])
    cn = {}
    for nx, x in (("L", lo[0]), ("R", hi[0])):
        for ny, y in (("F", lo[1]), ("B", hi[1])):
            cn["b" + nx + ny] = near(sup, x, y, lo[2])
            cn["t" + nx + ny] = near(top, x, y, hi[2])
    return cn


def run_analysis(model: Model, args, quiet=False):
    log("[4/6] Solving load cases and modal analysis ...", quiet)
    cases = load_cases(model, args)
    res, emats, K, fd = solve_all(model, args, cases)
    summary = {}
    energies = {}
    for name, c in cases.items():
        s = summarise_case(model, c, res[name])
        s["description"] = c["desc"]
        summary[name] = s
        energies[name] = energy_breakdown(model, emats, res[name])
    modes = modal(model, K, fd, emats, args.modes, quiet)

    # mechanism check: absurdly large displacement relative to size
    size = float(np.linalg.norm(model.xyz.max(0) - model.xyz.min(0)))
    mech = any(s.get("max_um", 0) / 1000 > 0.05 * size for s in summary.values())

    # ----- what-ifs
    log("[5/6] Running what-if studies ...", quiet)
    lat = [k for k in ("accel_X", "accel_Y", "torsion_Z") if k in cases]

    def stiff_of(r):
        out = {}
        for k in lat:
            s = summarise_case(model, cases[k], r[k])
            out[k] = s.get("stiffness_N_per_um", s.get("stiffness_Nm_per_deg"))
        return out

    base = stiff_of(res)

    def gains(new):
        return {k: (new[k] / base[k] - 1) * 100 if base[k] > 0 else 0.0 for k in lat}

    whatifs = []
    # rank members by average lateral energy share
    member_score = {}
    for k in lat:
        for o, f in energies[k].items():
            if o[0] == "member":
                member_score[o[1]] = member_score.get(o[1], 0) + f / len(lat)
    top_members = sorted(member_score, key=lambda i: -member_score[i])[:args.whatif_members]
    owner_member = np.array([o[1] if o[0] == "member" else -1 for o in model.owner])
    for pi in top_members:
        mask = owner_member == pi
        r, *_ = solve_all(model, args, cases, scale_mask=(mask, {"A": 2.0, "I": 3.0}))
        whatifs.append(dict(change=f"Upsize '{model.parts[pi].name}' one profile size (~3x bending stiffness, e.g. 2020->2040/3030)",
                            target=model.parts[pi].name, kind="upsize", gain_pct=gains(stiff_of(r))))
    jmask = np.array([o[0] == "joint" for o in model.owner])
    if jmask.any():
        r, *_ = solve_all(model, args, cases, scale_mask=(jmask, {"E": 5.0}))
        whatifs.append(dict(change="Stiffen ALL joints 5x (corner gussets, blind joints/end-tapped bolts, bigger printed brackets)",
                            target="all joints", kind="joints", gain_pct=gains(stiff_of(r))))
        # top joints only
        zt = model.xyz[:, 2].max() - 0.25 * (model.xyz[:, 2].max() - model.xyz[:, 2].min())
        tmask = jmask & (model.xyz[model.el_n[:, 0], 2] >= zt)
        if tmask.any() and tmask.sum() < jmask.sum():
            r, *_ = solve_all(model, args, cases, scale_mask=(tmask, {"E": 5.0}))
            whatifs.append(dict(change="Stiffen only the upper-frame joints 5x (top-corner gussets)",
                                target="upper joints", kind="joints", gain_pct=gains(stiff_of(r))))
    try:
        cn = corner_nodes(model)
        sides = {
            "back (Y max) side":   [("bLB", "tRB"), ("bRB", "tLB")],
            "front (Y min) side":  [("bLF", "tRF"), ("bRF", "tLF")],
            "left (X min) side":   [("bLF", "tLB"), ("bLB", "tLF")],
            "right (X max) side":  [("bRF", "tRB"), ("bRB", "tRF")],
            "top plane":           [("tLF", "tRB"), ("tRF", "tLB")],
        }
        for label, pairs in sides.items():
            extra = []
            for a, b in pairs:
                be = brace_elements(model, cn[a], cn[b], label)
                if be:
                    extra += be
            if not extra:
                continue
            r, *_ = solve_all(model, args, cases, extra=extra)
            whatifs.append(dict(change=f"Add X-diagonal braces (2020 profile) or a bolted shear panel on the {label}",
                                target=label, kind="brace", gain_pct=gains(stiff_of(r))))
    except Exception as ex:
        log(f"      brace what-ifs skipped: {ex}", quiet)
    for w in whatifs:
        w["avg_gain_pct"] = float(np.mean(list(w["gain_pct"].values())))
    whatifs.sort(key=lambda w: -w["avg_gain_pct"])
    return dict(cases=cases, summary=summary, energies=energies, modes=modes, whatifs=whatifs,
                disp=res, mechanism=mech, member_score=member_score, base=base)


# ----------------------------------------------------------------------------
# 5. Findings (deterministic)
# ----------------------------------------------------------------------------
def member_table(model: Model):
    rows = []
    for p in model.parts:
        row = dict(name=p.name, type=p.kind, material=p.material, role=p.role, library_match=p.lib_match,
                   print_profile=p.print_profile, E_used_MPa=round(p.E), mass_g=round(p.mass * 1000, 1),
                   size_mm=[round(float(x), 1) for x in p.ext])
        if p.kind == "beam":
            r = math.sqrt(min(p.Iy, p.Iz) / p.A) if p.A > 0 else 0
            # longest unsupported span between connection nodes
            ts = np.array(p.conn_t or p.node_t)
            Imin = min(p.Iy, p.Iz)
            f_span = None
            span = float(np.max(np.diff(ts))) if len(ts) > 1 else p.L
            if span > 0:
                f_span = (math.pi / 2) / span ** 2 * math.sqrt(p.E * Imin / (p.rho * 1e-3 * p.A))
            row.update(length_mm=round(p.L, 1), area_mm2=round(p.A, 1), Iy_mm4=round(p.Iy), Iz_mm4=round(p.Iz),
                       J_mm4=round(p.J), slenderness=round(p.L / r, 1) if r else None,
                       longest_span_mm=round(span, 1), span_freq_hz=round(f_span, 1) if f_span else None)
        rows.append(row)
    return rows


def findings(model: Model, A, args):
    out = []
    S = A["summary"]
    lat = [k for k in ("accel_X", "accel_Y") if k in S]
    if len(lat) == 2:
        kx, ky = S["accel_X"]["stiffness_N_per_um"], S["accel_Y"]["stiffness_N_per_um"]
        ratio = max(kx, ky) / max(min(kx, ky), 1e-12)
        if ratio > 1.5:
            weak = "X" if kx < ky else "Y"
            out.append(f"Stiffness is unbalanced: {weak} direction is {ratio:.1f}x softer than the other.")
    for k in ("accel_X", "accel_Y", "torsion_Z"):
        if k not in A["energies"]:
            continue
        jf = sum(f for o, f in A["energies"][k].items() if o[0] == "joint")
        if jf > 0.35:
            out.append(f"{k}: joints account for {jf*100:.0f}% of the flexibility -> joint quality is the bottleneck, not profile size.")
        topm = sorted(((f, o) for o, f in A["energies"][k].items() if o[0] == "member"), reverse=True)[:3]
        if topm:
            out.append(f"{k}: most flexible members: " +
                       ", ".join(f"{model.parts[o[1]].name} ({f*100:.0f}%)" for f, o in topm))
        # printed parts: their own beam energy + half of every joint link they take part in
        share = {}
        for o, f in A["energies"][k].items():
            ids = [o[1]] if o[0] == "member" else ([o[1], o[2]] if o[0] == "joint" else [])
            for i in ids:
                if MATERIALS[model.parts[i].material].get("printed"):
                    share[i] = share.get(i, 0) + (f if o[0] == "member" else f / 2)
        for i, f in sorted(share.items(), key=lambda t: -t[1])[:4]:
            if f > 0.05:
                p = model.parts[i]
                creep = MATERIALS[p.material].get("creep", "")
                out.append(f"{k}: printed part '{p.name}' ({p.material}, {p.print_profile or 'solid'}, E~{p.E:.0f} MPa) "
                           f"takes {f*100:.0f}% of the flexibility; consider metal, more walls/infill or a stiffer shape"
                           + (f" (creep: {creep})" if creep else "") + ".")
    fm = [m for m in A["modes"] if not m.get("local")]
    loc = [m for m in A["modes"] if m.get("local")]
    if loc:
        out.append("Local modes below the first frame mode (sub-assemblies moving on their own, not the frame): " +
                   ", ".join(f"{m['freq_hz']:.0f} Hz {m['dominant']}" for m in loc[:5]) +
                   ". Very low values usually mean a part held by one contact or by soft parts (silicone spacers) in the model.")
    if fm:
        f1 = fm[0]["freq_hz"]
        rating = "LOW (expect ringing; frame mode may sit inside input-shaper range)" if f1 < 40 else \
            "MODERATE" if f1 < 80 else "GOOD"
        out.append(f"First frame mode {f1:.0f} Hz ({fm[0]['shape']}) - {rating} (heuristic bands: <40 low, 40-80 moderate, >80 good).")
    elif A["modes"]:
        out.append(f"No frame mode among the first {len(A['modes'])} modes - raise --modes.")
    for row in member_table(model):
        if row["type"] == "beam" and row.get("role") not in ("lead screw",) and row.get("span_freq_hz") \
                and row["span_freq_hz"] < 60 and row["longest_span_mm"] > 150:
            out.append(f"'{row['name']}' has a {row['longest_span_mm']:.0f} mm unsupported span (local bending mode ~{row['span_freq_hz']:.0f} Hz).")
    if A["whatifs"]:
        w = A["whatifs"][0]
        out.append(f"Best single change tested: {w['change']} -> average +{w['avg_gain_pct']:.0f}% lateral/torsional stiffness.")
    if A["mechanism"]:
        out.append("WARNING: very large displacements - the model has a near-mechanism (parts only touching at one point, "
                   "or missing joints). Check --contact-tol and that brackets/plates are in the STEP.")
    if model.floating:
        out.append(f"{len(model.floating)} bodies are not connected to the supported frame and were left out: "
                   + ", ".join(model.floating[:10]))
    return out


# ----------------------------------------------------------------------------
# 6. Local LLM
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior mechanical engineer who designs high-speed 3D printer frames
(CoreXY, Voron, Rat Rig, Prusa style). You receive the numeric output of a frame stiffness
analysis (beam/joint finite-element model built from the printer's CAD) and write a practical,
prioritised improvement plan for a hobbyist builder.

Rules:
- Base every claim on the numbers given. Quote the what-if gains and energy shares you rely on.
- Never invent part names, measurements or results that are not in the data.
- Prioritise by (stiffness gain) / (cost + effort + loss of build volume). Diagonal braces on the
  front side block access; the back side and top plane are usually acceptable; panels can replace braces.
- Distinguish problems of joints (gussets, better brackets, end-tapped blind joints, more bolts)
  from problems of member size (bigger profiles) and of missing triangulation (braces/panels).
- Mention the first natural frequency and what it means for ringing / input shaping.
- Note the model limitations briefly at the end (estimated joint stiffness, no solid FEA)."""

USER_TEMPLATE = """Printer CAD file: {file}

Analysis results (JSON):
{data}

Write the report in Markdown with these sections:
## Verdict  (3-4 sentences: how stiff the frame is overall and the main weakness)
## Top improvements  (numbered, most effective first; for each: what to change, where, expected gain from the data, cost/effort)
## Secondary improvements
## What to verify in CAD / on the machine
"""


def ai_payload(model, A, args, finds):
    lat_e = {}
    for k, en in A["energies"].items():
        items = sorted(en.items(), key=lambda t: -t[1])[:10]
        lat_e[k] = [dict(item=owner_label(model, o), share_pct=round(f * 100, 1)) for o, f in items]
    data = dict(
        settings=dict(moving_mass_kg=args.moving_mass, accel_mm_s2=args.accel, support=args.support,
                      joint_factor=args.joint_factor),
        frame=dict(
            size_mm=[round(float(x)) for x in (model.xyz.max(0) - model.xyz.min(0))],
            structural_bodies=len(model.parts),
            beams=sum(p.kind == "beam" for p in model.parts),
            total_mass_kg=round(float(model.node_mass.sum()), 2)),
        load_cases={k: {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in s.items()}
                    for k, s in A["summary"].items()},
        modes=[dict(freq_hz=round(m["freq_hz"], 1), shape=m["shape"], local_subassembly=m.get("local", False),
                    top_contributors=[owner_label(model, o) for o, _ in
                                      sorted(m["energy"].items(), key=lambda t: -t[1])[:3]])
               for m in ([x for x in A["modes"] if not x.get("local")][:4] + [x for x in A["modes"] if x.get("local")][:4])],
        flexibility_share_per_case=lat_e,
        what_if_studies=[dict(change=w["change"], avg_gain_pct=round(w["avg_gain_pct"], 1),
                              gain_pct={k: round(v, 1) for k, v in w["gain_pct"].items()})
                         for w in A["whatifs"][:12]],
        automatic_findings=finds,
        beams=[r for r in member_table(model) if r["type"] == "beam"][:40],
    )
    return data


def call_llm(args, payload, quiet=False):
    data = json.dumps(payload, indent=1)
    if len(data) > args.max_prompt_chars:
        payload = dict(payload)
        payload["beams"] = payload["beams"][:10]
        data = json.dumps(payload, indent=1)[:args.max_prompt_chars]
    user = USER_TEMPLATE.format(file=os.path.basename(args.step), data=data)
    host = args.ai_host.rstrip("/")
    if args.api == "ollama":
        url = host + "/api/chat"
        body = dict(model=args.model, stream=False,
                    messages=[dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=user)],
                    options=dict(temperature=args.temperature, num_ctx=args.num_ctx))
    else:
        url = host + ("/chat/completions" if host.endswith("/v1") else "/v1/chat/completions")
        body = dict(model=args.model, temperature=args.temperature,
                    messages=[dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=user)])
    log(f"[6/6] Asking local AI ({args.api}: {args.model} @ {host}) - this can take a minute ...", quiet)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if args.api == "openai" and os.environ.get("OPENAI_API_KEY"):
        req.add_header("Authorization", "Bearer " + os.environ["OPENAI_API_KEY"])
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=args.ai_timeout) as r:
        resp = json.loads(r.read().decode())
    text = resp["message"]["content"] if args.api == "ollama" else resp["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models
    log(f"      AI answered in {time.time()-t0:.0f}s", quiet)
    return text


# ----------------------------------------------------------------------------
# 7. Output
# ----------------------------------------------------------------------------
def plot(model: Model, A, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    panels = [(k, A["disp"][k]) for k in ("accel_X", "accel_Y", "torsion_Z") if k in A["disp"]]
    if A["modes"]:
        fm = [m for m in A["modes"] if not m.get("local")] or A["modes"]
        panels.append((f"first frame mode: {fm[0]['freq_hz']:.0f} Hz ({fm[0]['shape']})", fm[0]["u"]))
    fig = plt.figure(figsize=(6 * len(panels), 6))
    xyz = model.xyz
    size = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    cmap = plt.get_cmap("inferno_r")
    for i, (title, u) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        d = u.reshape(-1, 6)[:, :3]
        sc = 0.06 * size / max(np.abs(d).max(), 1e-12)
        en = energy_breakdown(model, element_matrices(model), u)
        emax = max([v for k, v in en.items()] + [1e-12])
        for k, (a, b) in enumerate(model.el_n):
            o = model.owner[k]
            key = o if o[0] != "member" else ("member", o[1])
            frac = en.get(key, 0) / emax
            p0, p1 = xyz[[a, b]], xyz[[a, b]] + sc * d[[a, b]]
            ax.plot(*p0.T, color="0.8", lw=0.8)
            ax.plot(*p1.T, color=cmap(0.15 + 0.85 * frac), lw=1.2 + 2.5 * frac,
                    ls="--" if o[0] == "joint" else "-")
        ax.scatter(*xyz[model.support].T, marker="^", color="tab:blue", s=12)
        ax.set_title(f"{title}\n(deformation x{sc:.0f}; dark = most flexible)", fontsize=9)
        ax.set_box_aspect(np.ptp(xyz, axis=0) + 1)
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(model, A, args, finds, ai_text, outdir, plot_path):
    S = A["summary"]
    L = []
    L.append(f"# Frame stiffness report - {os.path.basename(args.step)}")
    L.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by printer_stiffness.py v{__version__}_\n")
    sz = model.xyz.max(0) - model.xyz.min(0)
    L.append(f"**Frame envelope (node centres):** {sz[0]:.0f} x {sz[1]:.0f} x {sz[2]:.0f} mm, "
             f"{len(model.parts)} structural bodies, structural mass {model.node_mass.sum():.2f} kg  ")
    L.append(f"**Loads:** moving mass {args.moving_mass} kg at {args.accel:.0f} mm/s^2; supports: {args.support} "
             f"at {len(model.support)} bottom nodes; joint factor {args.joint_factor}\n")
    L.append("## Results\n")
    L.append("| Load case | Description | Top deflection | Stiffness |")
    L.append("|---|---|---|---|")
    for k, s in S.items():
        if "stiffness_N_per_um" in s:
            L.append(f"| {k} | {s['description']} | {s['defl_um']:.1f} um (max {s['max_um']:.1f}) | {s['stiffness_N_per_um']:.2f} N/um |")
        else:
            L.append(f"| {k} | {s['description']} | {s['rot_deg']*3600:.1f} arcsec (max {s['max_um']:.1f} um) | {s['stiffness_Nm_per_deg']:.0f} N*m/deg |")
    if A["modes"]:
        L.append("\n**Natural frequencies (frame):** " + (", ".join(f"{m['freq_hz']:.0f} Hz ({m['shape']})" for m in [x for x in A["modes"] if not x.get("local")][:8]) or "none in the computed range"))
        loc = [m for m in A["modes"] if m.get("local")]
        if loc:
            L.append(f"\n**Local modes (sub-assemblies, not the frame; {len(loc)} found, first 10):** " + ", ".join(f"{m['freq_hz']:.0f} Hz ({m['dominant']})" for m in loc[:10]))
    L.append("\n## Automatic findings\n")
    L += [f"- {f}" for f in finds]
    L.append("\n## What-if studies (stiffness gain vs current design)\n")
    keys = [k for k in ("accel_X", "accel_Y", "torsion_Z") if k in S]
    L.append("| Change | " + " | ".join(keys) + " | Average |")
    L.append("|---|" + "---|" * (len(keys) + 1))
    for w in A["whatifs"]:
        L.append(f"| {w['change']} | " + " | ".join(f"{w['gain_pct'][k]:+.0f}%" for k in keys) + f" | **{w['avg_gain_pct']:+.0f}%** |")
    L.append("\n## Where the flexibility is (share of strain energy)\n")
    for k, en in A["energies"].items():
        items = sorted(en.items(), key=lambda t: -t[1])[:8]
        L.append(f"**{k}:** " + "; ".join(f"{owner_label(model, o)} {f*100:.0f}%" for o, f in items) + "\n")
    if plot_path:
        L.append(f"![deformed shapes]({os.path.basename(plot_path)})\n")
    L.append("## Recommendations from local AI\n")
    L.append(ai_text if ai_text else "_AI step skipped or unavailable - see findings and what-if table above._")
    L.append("\n## Member inventory\n")
    L.append("| Body | Type | Library match | Material (print) | E used (MPa) | Size (mm) | Mass (g) | Iy / Iz (mm^4) | Longest span (mm) | Span mode (Hz) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in member_table(model):
        mat = r['material'] + (f" ({r['print_profile']})" if r.get('print_profile') else "")
        L.append(f"| {r['name']} | {r['type']} | {r.get('library_match') or '-'} | {mat} | {r['E_used_MPa']} | "
                 f"{' x '.join(f'{x:.0f}' for x in r['size_mm'])} | "
                 f"{r['mass_g']:.0f} | {(str(r.get('Iy_mm4'))+' / '+str(r.get('Iz_mm4'))) if r['type']=='beam' else '-'} | "
                 f"{r.get('longest_span_mm', '-')} | {r.get('span_freq_hz', '-')} |")
    if model.excluded:
        L.append(f"\n<details><summary>{len(model.excluded)} bodies ignored</summary>\n")
        L += [f"- {n}: {why}" for n, why in model.excluded]
        L.append("\n</details>")
    L.append("""
## Assumptions and limitations
- Reduced-order frame model: slender bodies are beams (section properties from CAD mass properties), plates and
  blocks are hubs connected by flexible spokes. It is not a solid (tetrahedral) FEA.
- Bolted joints are modelled as short links with `--joint-factor` x the weaker part's modulus. Real joints vary a lot;
  the joint share and the "stiffen joints" what-if tell you whether this matters for your frame.
- Torsion constants of slotted extrusions are estimates (`--extrusion-torsion-factor`).
- Materials come from body-name rules in library/base_library.json (override with `--part-material "regex=material"`).
  Printed parts use an effective modulus for their walls/infill (library print_profiles); this is an estimate.
- Beams whose section matches a library profile (V-slot 2020/2040, MGN9/MGN12 rail) use the library's measured
  section properties; others use values computed from the CAD.
- Use the results to rank weaknesses and compare options; validate important changes on the machine
  (accelerometer / input-shaper resonance test).
""")
    path = os.path.join(outdir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def write_json(model, A, args, finds, ai_text, outdir):
    def ser_en(en):
        return [dict(item=owner_label(model, o), share=f) for o, f in sorted(en.items(), key=lambda t: -t[1])]
    data = dict(
        file=os.path.abspath(args.step), version=__version__, settings=vars(args).copy(),
        summary=A["summary"], findings=finds,
        modes=[dict(freq_hz=m["freq_hz"], shape=m["shape"], local=m.get("local", False), frame_share=m.get("frame_share"),
                    energy=ser_en(m["energy"])[:10]) for m in A["modes"][:40]],
        energies={k: ser_en(v) for k, v in A["energies"].items()},
        whatifs=A["whatifs"], members=member_table(model),
        excluded=[dict(name=n, reason=r) for n, r in model.excluded], floating=model.floating,
        ai_recommendations=ai_text)
    data["settings"]["material_rules"] = [list(r) for r in args.material_rules]
    path = os.path.join(outdir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    return path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
DEFAULT_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")


def preload_library(argv):
    """Load the library before the parser is built (its materials become valid --part-material choices)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--library", default=DEFAULT_LIBRARY)
    pre.add_argument("--no-library", action="store_true")
    known, _ = pre.parse_known_args(argv)
    lib = None if known.no_library else load_library(known.library)
    apply_library(lib)
    return lib


def add_model_args(ap):
    g = ap.add_argument_group("parts, materials and library")
    g.add_argument("--library", default=DEFAULT_LIBRARY,
                   help="folder of part-library JSON files (materials, print profiles, name rules, profiles)")
    g.add_argument("--no-library", action="store_true", help="ignore the library; use built-in rules only")
    g.add_argument("--no-library-sections", action="store_true",
                   help="always use section properties computed from the CAD instead of matched library profiles")
    g.add_argument("--default-material", default="aluminum", choices=list(MATERIALS))
    g.add_argument("--part-material", action="append", default=[], metavar="REGEX=MATERIAL",
                   help="assign material by body-name regex, e.g. 'corner|bracket=pet_cf' (repeatable)")
    g.add_argument("--print-profile", default="duender_load_bearing",
                   help="print settings for printed parts no library rule gives one (see library print_profiles)")
    g.add_argument("--ignore", default="", help="extra regex of body names to ignore (library rules already skip fasteners)")
    g.add_argument("--min-volume", type=float, default=50.0, help="ignore bodies smaller than this, mm^3 (fasteners are skipped by name anyway)")
    g.add_argument("--contact-tol", type=float, default=0.3, help="gap below which two bodies count as connected, mm")
    g.add_argument("--contact-spacing", type=float, default=60.0, help="spacing of connection points along long contacts, mm")
    g.add_argument("--slender", type=float, default=3.0, help="length/width ratio above which a body is a beam")
    g.add_argument("--extrusion-torsion-factor", type=float, default=0.35, help="torsion reduction for slotted profiles (CAD-estimated sections)")
    g.add_argument("--up", default="z", help="vertical axis of the CAD model: x, y, z or -z")
    g.add_argument("--no-cache", action="store_true", help="re-read the STEP even if a cached geometry exists")


def finish_args(ap, args):
    rules = []
    for s in args.part_material:
        if "=" not in s:
            ap.error(f"--part-material needs REGEX=MATERIAL, got '{s}'")
        rx, mat = s.rsplit("=", 1)
        mat = mat.strip().lower()
        if mat not in MATERIALS:
            ap.error(f"unknown material '{mat}'. Choose from: {', '.join(MATERIALS)}")
        rules.append((rx.strip().lower(), mat))
    args.material_rules = rules
    if not os.path.isfile(args.step):
        ap.error(f"file not found: {args.step}")
    return args


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Frame stiffness analysis of a 3D printer STEP model, with improvement advice from a local LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("step", help="STEP file (.step/.stp) of the printer frame or full printer")
    ap.add_argument("-o", "--out", help="output folder (default: <step name>_stiffness next to the STEP file)")
    g = ap.add_argument_group("loads and supports")
    g.add_argument("--moving-mass", type=float, default=1.5, help="moving gantry + toolhead mass, kg")
    g.add_argument("--accel", type=float, default=10000, help="print acceleration, mm/s^2")
    g.add_argument("--support", choices=["fixed", "pinned"], default="fixed", help="how the bottom nodes are held")
    g.add_argument("--support-tol", type=float, default=None, help="nodes within this height of the lowest node are supports, mm (default 3%% of height, min 5)")
    g.add_argument("--top-band", type=float, default=0.10, help="fraction of height (from the top) whose nodes receive the gantry load")
    g.add_argument("--top-z", type=float, default=None, help="explicit height (internal Z, mm) above which nodes are loaded")
    add_model_args(ap)
    g = ap.add_argument_group("model check (runs first, see validate_model.py)")
    g.add_argument("--edition", default="auto", help="Duender edition for the parts-list check: c300, c400, c500, auto or none")
    g.add_argument("--length-warn", type=float, default=0.5, help="length difference that gives a warning, mm")
    g.add_argument("--length-fail", type=float, default=2.0, help="length difference that fails, mm")
    g = ap.add_argument_group("frame model")
    g.add_argument("--joint-factor", type=float, default=0.3, help="joint link stiffness factor (1 = as stiff as the parts)")
    g.add_argument("--node-merge", type=float, default=2.0, help="merge connection points closer than this along a beam, mm")
    g.add_argument("--max-element", type=float, default=100.0, help="max beam element length, mm")
    g.add_argument("--modes", type=int, default=20, help="number of natural frequencies (local sub-assembly modes are labelled)")
    g.add_argument("--whatif-members", type=int, default=6, help="how many weakest members get an upsize what-if")
    g = ap.add_argument_group("local AI")
    g.add_argument("--no-ai", action="store_true", help="skip the LLM step")
    g.add_argument("--api", choices=["ollama", "openai"], default=os.environ.get("PS_AI_API", "ollama"),
                   help="ollama native API, or any OpenAI-compatible local server")
    g.add_argument("--ai-host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"),
                   help="AI server URL (e.g. http://my-gpu-box:11434)")
    g.add_argument("--model", default=os.environ.get("PS_AI_MODEL", "llama3.1:8b"), help="model name on the AI server")
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--num-ctx", type=int, default=16384, help="context window requested from Ollama")
    g.add_argument("--max-prompt-chars", type=int, default=40000)
    g.add_argument("--ai-timeout", type=float, default=900, help="seconds")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version=__version__)
    return finish_args(ap, ap.parse_args(argv))


def main(argv=None):
    lib = preload_library(argv if argv is not None else sys.argv[1:])
    args = parse_args(argv)
    q = args.quiet
    outdir = args.out or os.path.splitext(os.path.abspath(args.step))[0] + "_stiffness"
    os.makedirs(outdir, exist_ok=True)
    if lib:
        log(f"Library: {args.library} ({', '.join(lib['files'])})", q)

    parts, excluded, contacts = load_parts_cached(args, q, lib)
    parts, contacts, merges, gaps = merge_collinear(parts, contacts, args, q)
    check = None
    try:
        import model_check as mc
        ed = args.edition.lower()
        ed = mc.detect_edition(args.step, parts, excluded, lib) if ed == "auto" else (None if ed == "none" else ed)
        check = mc.run_checks(parts, excluded, contacts, lib, args, ed, None, merges, gaps)
        with open(os.path.join(outdir, "model_check.md"), "w", encoding="utf-8") as f:
            f.write(mc.markdown(check, os.path.basename(args.step)))
        c = check["counts"]
        log(f"      model check ({ed or 'generic'}): {c['FAIL']} fail, {c['WARN']} warn, {c['PASS']} pass "
            f"-> model_check.md", q)
    except ImportError:
        pass
    if not parts:
        sys.exit("Nothing structural left after filtering. Check --ignore / --min-volume.")
    model = build_model(parts, excluded, contacts, args, q)
    if len(model.el_n) == 0 or len(model.support) == 0:
        sys.exit("Could not build a connected frame (no elements or no supports). Try a larger --contact-tol or check --up.")
    A = run_analysis(model, args, q)
    finds = findings(model, A, args)
    if check:
        fails = [x for x in check["checks"] if x["level"] == "FAIL"]
        if fails:
            finds.insert(0, f"MODEL CHECK: {len(fails)} problem(s) in the CAD itself - fix these before trusting the "
                            f"numbers (see model_check.md): " + "; ".join(f"{x['item']}: {x['message']}" for x in fails[:5]))

    ai_text = None
    if not args.no_ai:
        try:
            ai_text = call_llm(args, ai_payload(model, A, args, finds), q)
        except urllib.error.URLError as ex:
            log(f"      Could not reach the AI server at {args.ai_host} ({ex}). Report written without AI section.\n"
                f"      Is Ollama running?  ('ollama serve', and 'ollama pull {args.model}')", q)
        except Exception as ex:
            log(f"      AI step failed: {ex}. Report written without AI section.", q)

    plot_path = plot(model, A, os.path.join(outdir, "deformed_shapes.png"))
    rep = write_report(model, A, args, finds, ai_text, outdir, plot_path)
    js = write_json(model, A, args, finds, ai_text, outdir)

    if not q:
        print("\n==== SUMMARY ====")
        for k, s in A["summary"].items():
            if "stiffness_N_per_um" in s:
                print(f"  {k:10s} {s['stiffness_N_per_um']:8.2f} N/um   ({s['defl_um']:.1f} um)")
            else:
                print(f"  {k:10s} {s['stiffness_Nm_per_deg']:8.0f} N*m/deg")
        if A["modes"]:
            fm = [m for m in A["modes"] if not m.get("local")]
        print("  frame modes: " + (", ".join(f"{m['freq_hz']:.0f} Hz {m['shape']}" for m in fm[:4]) or "none found"))
        nl = sum(1 for m in A["modes"] if m.get("local"))
        if nl:
            print(f"  ({nl} local sub-assembly modes below/among them - see report)")
        print("  best changes:")
        for w in A["whatifs"][:5]:
            print(f"    {w['avg_gain_pct']:+6.0f}%  {w['change']}")
        print(f"\nReport: {rep}\nData:   {js}" + (f"\nPlot:   {plot_path}" if plot_path else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
