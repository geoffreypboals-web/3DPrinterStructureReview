#!/usr/bin/env python3
"""Generate a simple CoreXY-style demo frame STEP (2020 cube + MGN rails + gantry)
to test printer_stiffness.py.   Usage:  python make_demo_frame.py demo_frame.step"""
import os
import sys
import gmsh

out = sys.argv[1] if len(sys.argv) > 1 else "demo_frame.step"
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
W, D, H, S = 400.0, 400.0, 450.0, 20.0     # outer width, depth, height, profile size

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
o = gmsh.model.occ
named = []


def box(name, x, y, z, dx, dy, dz):
    t = o.addBox(x, y, z, dx, dy, dz)
    named.append((t, name))


# 4 posts
for i, (x, y) in enumerate([(0, 0), (W - S, 0), (0, D - S), (W - S, D - S)]):
    box(f"post_2020x{H:.0f}", x, y, 0, S, S, H)
# bottom + top rings (X rails front/back, Y rails left/right)
for z, lvl in ((0, "bottom"), (H - S, "top")):
    box(f"{lvl}_front_2020", S, 0, z, W - 2 * S, S, S)
    box(f"{lvl}_back_2020", S, D - S, z, W - 2 * S, S, S)
    box(f"{lvl}_left_2020", 0, S, z, S, D - 2 * S, S)
    box(f"{lvl}_right_2020", W - S, S, z, S, D - 2 * S, S)
# MGN12 rails on top of the left/right top extrusions
box("mgn12_rail_left", 4, 0, H, 12, D, 8)
box("mgn12_rail_right", W - 16, 0, H, 12, D, 8)
# carriages + X gantry beam
box("mgn12_carriage_left", 0, D / 2 - 20, H + 8, 20, 40, 10)
box("mgn12_carriage_right", W - 20, D / 2 - 20, H + 8, 20, 40, 10)
box("x_gantry_2020", 0, D / 2 - 10, H + 18, W, S, S)
# printed corner brackets under the top ring (front corners only - back left open on purpose)
box("corner_bracket_petg_FL", S, 0, H - 2 * S, S, S, S)
box("corner_bracket_petg_FR", W - 2 * S, 0, H - 2 * S, S, S, S)
# a bed support plate on the bottom ring
box("bed_plate", S, S, S, W - 2 * S, D - 2 * S, 6)

o.synchronize()
gmsh.write(out)
gmsh.finalize()

# gmsh's STEP writer does not store body names; patch them into the PRODUCT entities
import re
txt = open(out, encoding="utf-8", errors="ignore").read()
names = [n for _, n in named]
txt = re.sub(r"'Open CASCADE STEP translator [0-9.]+ 1\.(\d+)'",
             lambda m: f"'{names[int(m.group(1)) - 1]}'" if int(m.group(1)) <= len(names) else m.group(0), txt)
open(out, "w", encoding="utf-8").write(txt)
print("wrote", out)
