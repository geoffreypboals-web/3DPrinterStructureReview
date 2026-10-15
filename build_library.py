#!/usr/bin/env python3
"""Build library/duender_generated.json from this repo's Duender CAD (T-015).

Reads (no FreeCAD needed, only gmsh):
  cad/designs/duender-large/lib/profile_2020.brep, profile_2040.brep, rail_mgn12.brep  (real sections)
  cad/designs/duender-large/lib/lego_mgn9.brep, carriage_mgn12h.brep, nema17.brep, coupler.brep
  cad/designs/duender-large/out/c{300,400,500}-mgn9x-steps.json   (key letters, labels, quantities, key dims)
Writes:
  tools/printer-stiffness/library/duender_generated.json
     profiles  - section properties measured from the real profile geometry
     blocks    - bounding-box signatures of catalog parts (carriages, motor, coupler)
     editions  - expected parts per edition (key, label, profile, length, qty) + design rules

The .brep files are in Git LFS: run `git lfs pull` first.  Re-run this after build_lego_mgn9.py changes a size.

    python tools/printer-stiffness/build_library.py
"""
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CAD = os.path.join(REPO, "cad", "designs", "duender-large")
OUT = os.path.join(HERE, "library", "duender_generated.json")


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def need(path):
    if not os.path.isfile(path):
        sys.exit(f"missing {path}")
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            sys.exit(f"{rel(path)} is a Git LFS pointer. Run:  git -C {REPO} lfs pull")
    return path


def torsion_estimate(A, I1, I2, slotted):
    j = A ** 4 / (4 * math.pi ** 2 * (I1 + I2))
    return round(j * (0.35 if slotted else 1.0), 1)


def face_section(gmsh, path):
    gmsh.clear()
    gmsh.model.occ.importShapes(need(path))
    gmsh.model.occ.synchronize()
    faces = gmsh.model.getEntities(2)
    t = max(faces, key=lambda e: gmsh.model.occ.getMass(2, e[1]))[1]
    A = gmsh.model.occ.getMass(2, t)
    bb = gmsh.model.occ.getBoundingBox(2, t)
    ext = np.array(bb[3:]) - np.array(bb[:3])
    J = np.array(gmsh.model.occ.getMatrixOfInertia(2, t)).reshape(3, 3)
    ax = sorted(np.argsort(ext)[1:].tolist())               # the two in-plane axes (bbox has a tolerance pad)
    a, b = ax
    # For a planar face in the a-b plane: J[a,a] = int(b^2) dA and J[b,b] = int(a^2) dA.
    # "along x" = integral of x^2 = the second moment that resists bending deflection along x.
    along_b, along_a = J[a, a], J[b, b]
    dims = {a: ext[a], b: ext[b]}
    small, large = (a, b) if ext[a] <= ext[b] else (b, a)
    along = {a: along_a, b: along_b}
    return dict(A=round(A, 2), envelope=[round(float(dims[small]), 2), round(float(dims[large]), 2)],
                I_small=round(float(along[small]), 1), I_large=round(float(along[large]), 1))


def solid_section(gmsh, V, L, Jm, e_long, e_small, e_large):
    """Prismatic solid: section second moments from the mass inertia tensor (density 1).
    Uses J_long = L*(Is+Il) and J_small - J_large = L*(Il - Is): the V*L^2/12 term cancels, so holes and end
    chamfers don't wreck the result (subtracting V*L^2/12 directly is ill-conditioned on real CAD)."""
    J1 = e_long @ Jm @ e_long
    Js = e_small @ Jm @ e_small     # = V L^2/12 + L * int(s_large^2)
    Jl = e_large @ Jm @ e_large     # = V L^2/12 + L * int(s_small^2)
    I_large = (J1 + Js - Jl) / (2 * L)
    I_small = (J1 - Js + Jl) / (2 * L)
    return I_small, I_large


def main():
    try:
        import gmsh
    except ImportError:
        sys.exit("pip install gmsh")
    gmsh.initialize(["-noenv"])
    gmsh.option.setNumber("General.Terminal", 0)
    lib = os.path.join(CAD, "lib")
    profiles, blocks = {}, {}

    # ---- extrusions (real V-slot sections)
    for name in ("2020", "2040"):
        s = face_section(gmsh, os.path.join(lib, f"profile_{name}.brep"))
        slotted = True
        profiles[name] = dict(kind="extrusion", material="aluminum", envelope=s["envelope"], A=s["A"],
                              I_small=s["I_small"], I_large=s["I_large"],
                              J=torsion_estimate(s["A"], s["I_small"], s["I_large"], slotted), J_is_estimate=True,
                              kg_per_m=round(s["A"] * 2.70e-6 * 1000, 3),
                              source=f"{rel(os.path.join(lib, f'profile_{name}.brep'))} (Andizzle's real V-slot section)")
    # ---- MGN12 rail (section face)
    s = face_section(gmsh, os.path.join(lib, "rail_mgn12.brep"))
    profiles["MGN12"] = dict(kind="linear rail", material="steel", envelope=s["envelope"], A=s["A"],
                             I_small=s["I_small"], I_large=s["I_large"],
                             J=torsion_estimate(s["A"], s["I_small"], s["I_large"], False), J_is_estimate=True,
                             kg_per_m=round(s["A"] * 7.85e-6 * 1000, 3), hole_pitch_mm=25, screw="M3",
                             source=rel(os.path.join(lib, "rail_mgn12.brep")))
    # ---- MGN9 rail + MGN9H carriage (Sergei's MGN9H kit)
    gmsh.clear()
    gmsh.model.occ.importShapes(need(os.path.join(lib, "lego_mgn9.brep")))
    gmsh.model.occ.synchronize()
    rail_done = False
    carr_lo, carr_hi = np.full(3, 1e9), np.full(3, -1e9)
    for _, t in gmsh.model.getEntities(3):
        bb = gmsh.model.occ.getBoundingBox(3, t)
        ext = np.array(bb[3:]) - np.array(bb[:3])
        if ext.max() > 100:
            if not rail_done:
                V = gmsh.model.occ.getMass(3, t)
                Jm = np.array(gmsh.model.occ.getMatrixOfInertia(3, t)).reshape(3, 3)
                o = np.argsort(ext)          # small, large, long
                E = np.eye(3)
                L = ext[o[2]]
                I_small, I_large = solid_section(gmsh, V, L, Jm, E[o[2]], E[o[0]], E[o[1]])
                A = V / L
                profiles["MGN9"] = dict(kind="linear rail", material="steel",
                                        envelope=[round(float(ext[o[0]]), 2), round(float(ext[o[1]]), 2)],
                                        A=round(A, 2), I_small=round(I_small, 1), I_large=round(I_large, 1),
                                        J=torsion_estimate(A, I_small, I_large, False), J_is_estimate=True,
                                        kg_per_m=round(A * 7.85e-6 * 1000, 3), hole_pitch_mm=20, screw="M3",
                                        source=rel(os.path.join(lib, "lego_mgn9.brep")) + " (Sergei's MGN9H kit)")
                rail_done = True
        elif ext.max() < 50:        # block body, end caps and seals (skip the 70 mm retainer strips)
            carr_lo = np.minimum(carr_lo, bb[:3]); carr_hi = np.maximum(carr_hi, bb[3:])
    blocks["MGN9H carriage"] = dict(bbox=sorted(np.round(carr_hi - carr_lo, 1).tolist()), material="steel",
                                    source=rel(os.path.join(lib, "lego_mgn9.brep")))
    # ---- block signatures
    for name, f, mat in (("MGN12H carriage", "carriage_mgn12h.brep", "steel"),
                         ("NEMA17 42-40", "nema17.brep", "motor"),
                         ("coupler 5 x 8", "coupler.brep", "aluminum")):
        gmsh.clear()
        gmsh.model.occ.importShapes(need(os.path.join(lib, f)))
        gmsh.model.occ.synchronize()
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for _, t in gmsh.model.getEntities(3):
            bb = gmsh.model.occ.getBoundingBox(3, t)
            lo = np.minimum(lo, bb[:3]); hi = np.maximum(hi, bb[3:])
        blocks[name] = dict(bbox=sorted(np.round(hi - lo, 1).tolist()), material=mat, source=rel(os.path.join(lib, f)))
    gmsh.finalize()
    # T8 lead screw: nominal 8 mm solid rod (the tool only uses it for mass and as a frame-free member)
    profiles["T8"] = dict(kind="lead screw", material="steel", envelope=[8.0, 8.0], A=round(math.pi * 16, 2),
                          I_small=round(math.pi * 8 ** 4 / 64, 1), I_large=round(math.pi * 8 ** 4 / 64, 1),
                          J=round(math.pi * 8 ** 4 / 32, 1), J_is_estimate=True, lead_mm=8, starts=4,
                          source="nominal T8x8 (8 mm major diameter); thread root not modelled")

    # ---- editions from the generator outputs
    editions = {}
    for size, donor_ed in (("300", "c300-v3"), ("400", "c400-s4"), ("500", "c500-s5")):
        f = os.path.join(CAD, "out", f"c{size}-mgn9x-steps.json")
        if not os.path.isfile(f):
            print("skip (missing):", rel(f)); continue
        d = json.load(open(f, encoding="utf-8"))
        parts = {}
        for st in d["steps"]:
            if st["step"] == "ALL":
                continue
            for it in st.get("new", []):
                k = (it["key"], it["label"])
                if k not in parts:
                    parts[k] = dict(key=it["key"], label=it["label"], qty=0, step=st["step"])
                parts[k]["qty"] += it["qty"]
        for p in parts.values():
            lab = p["label"]
            m = re.match(r"(2020|2040)(?: X beam)? x (\d+(?:\.\d+)?)", lab)
            if m:
                p["profile"], p["length"] = m.group(1), float(m.group(2))
            m = re.match(r"(MGN12H?|MGN9) (?:X )?rail x (\d+(?:\.\d+)?)", lab)
            if m:
                p["profile"], p["length"] = ("MGN9" if m.group(1) == "MGN9" else "MGN12"), float(m.group(2))
            m = re.match(r"T8 lead screw x (\d+)", lab)
            if m:
                p["profile"], p["length"] = "T8", float(m.group(1))
        rules = dict(
            rail_not_longer_than=[["T", "S"], ["Q", "F"], ["H", "E"], ["P", "U"]],
            equal_length_keys=["A", "B", "C", "D", "E", "F", "G", "H", "K", "Q"],
            relations=([{"check": "C == K + 35", "why": "upright = lead screw + 35 (generator rule)"}] if size != "300"
                       else [{"check": "C == 550", "why": "C300 keeps Andizzle's 550 uprights"}]),
        )
        editions[f"c{size}"] = dict(
            manual=f"manual/editions/{donor_ed}",
            step_file=rel(os.path.join(CAD, "out", f"c{size}-duender-3z-mgn9x.step")),
            source=rel(f), dims=d["dims"], parts=sorted(parts.values(), key=lambda p: (p["step"], p["key"])),
            rules=rules)

    data = dict(_about="GENERATED by build_library.py from the repo CAD - do not edit by hand; edit base_library.json "
                       "or re-run the generator.", profiles=profiles, blocks=blocks, editions=editions)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", rel(OUT))
    for k, v in profiles.items():
        print(f"  {k:6s} A={v['A']:7.1f}  I={v['I_small']:9.1f} / {v['I_large']:9.1f}  {v.get('kg_per_m', '-')} kg/m")
    for k, v in blocks.items():
        print(f"  {k:16s} bbox {v['bbox']}")
    for k, v in editions.items():
        print(f"  {k}: {len(v['parts'])} part lines")


if __name__ == "__main__":
    main()
