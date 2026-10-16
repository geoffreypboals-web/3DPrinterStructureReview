#!/usr/bin/env python3
"""Check a printer STEP model before trusting any analysis of it.

    python validate_model.py printer.step                 # generic checks; Duender edition auto-detected
    python validate_model.py printer.step --edition c500  # force the C500 parts list
    python validate_model.py printer.step --interference  # also measure metal-to-metal overlaps (slower)

Writes <step>_check/model_check.md and model_check.json next to the STEP file.
Exit code: 0 = no FAIL, 1 = at least one FAIL.
"""
import argparse
import json
import os
import sys

import printer_stiffness as ps
import model_check as mc


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    lib = ps.preload_library(argv)
    ap = argparse.ArgumentParser(description="Validate a 3D-printer STEP model against the part library and the "
                                             "Duender manuals' parts lists.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("step", help="STEP file")
    ap.add_argument("-o", "--out", help="output folder (default: <step>_check next to the file)")
    ap.add_argument("--edition", default="auto", help="Duender edition to check against: c300, c400, c500, auto or none")
    ap.add_argument("--interference", action="store_true", help="measure overlaps between touching metal bars (slower)")
    ap.add_argument("--length-warn", type=float, default=0.5, help="length difference that gives a warning, mm")
    ap.add_argument("--length-fail", type=float, default=2.0, help="length difference that fails, mm")
    ap.add_argument("-q", "--quiet", action="store_true")
    ps.add_model_args(ap)
    args = ps.finish_args(ap, ap.parse_args(argv))
    q = args.quiet
    if lib:
        ps.log(f"Library: {args.library} ({', '.join(lib['files'])})", q)

    inter = None
    if args.interference:
        parts, excluded, contacts = ps.load_parts(args, q, lib, keep_open=True)
        import gmsh
        ps.log("      measuring metal-to-metal overlaps ...", q)
        inter = mc.interference_volumes(gmsh, parts, contacts)
        gmsh.finalize()
    else:
        parts, excluded, contacts = ps.load_parts_cached(args, q, lib)
    parts, contacts, merges, gaps = ps.merge_collinear(parts, contacts, args, q)

    ed = args.edition.lower()
    if ed == "auto":
        ed = mc.detect_edition(args.step, parts, excluded, lib)
    elif ed == "none":
        ed = None
    if ed and ed not in ((lib or {}).get("editions") or {}):
        sys.exit(f"unknown edition '{ed}'. Known: {', '.join(((lib or {}).get('editions') or {}))}")
    result = mc.run_checks(parts, excluded, contacts, lib, args, ed, inter, merges, gaps)

    outdir = args.out or os.path.splitext(os.path.abspath(args.step))[0] + "_check"
    os.makedirs(outdir, exist_ok=True)
    md = os.path.join(outdir, "model_check.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(mc.markdown(result, os.path.basename(args.step)))
    with open(os.path.join(outdir, "model_check.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    c = result["counts"]
    if not q:
        print(f"\n==== MODEL CHECK ({ed or 'generic'}) ====  {c['FAIL']} fail, {c['WARN']} warn, {c['PASS']} pass")
        for lvl in ("FAIL", "WARN"):
            for x in [x for x in result["checks"] if x["level"] == lvl][:25]:
                print(f"  {lvl}  [{x['area']}] {x['item']}: {x['message']}")
        print(f"\nReport: {md}")
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
