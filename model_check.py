"""Model validation for printer STEP files (T-015).

Generic checks (any printer):
  units / size sanity, duplicate bodies, bodies that touch nothing, section of a body vs the profile its name
  claims (e.g. "2020 x 400" that is not a 20x20 V-slot), length in the name vs measured length,
  optional metal-to-metal interference.
Edition checks (Duender C300/C400/C500, from library/duender_generated.json):
  every key letter of the manual's picture list is present the right number of times, each frame/rail/screw part
  has the right profile and cut length, and the design rules hold (rail no longer than its extrusion,
  same-key parts equal, upright = lead screw + 35 ...).

Used by validate_model.py (stand-alone) and by printer_stiffness.py --edition.
"""
import math
import re

import numpy as np

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _key_of(label, keys):
    tok = label.split()[0] if label.split() else ""
    return tok if tok in keys else None


def _strip(name):
    return re.sub(r" #\d+$", "", name)


def _base_label(instance):
    return re.sub(r"\d{3}$", "", instance).strip()


def detect_edition(step_path, parts, excluded, lib):
    eds = (lib or {}).get("editions") or {}
    if not eds:
        return None
    low = step_path.lower().replace("\\", "/")
    for name in eds:
        if re.search(rf"(^|[^0-9a-z]){name}([^0-9]|$)", low.split("/")[-1]):
            return name
    labels = {_base_label(p.instance) for p in parts} | {_base_label(_strip(n)) for n, _ in excluded}
    best, score = None, 0
    for name, ed in eds.items():
        exp = {f"{p['key']} {p['label']}" for p in ed["parts"]}
        sc = len(exp & labels)
        if sc > score:
            best, score = name, sc
    return best if score >= 5 else None


def run_checks(parts, excluded, contacts, lib, args, edition=None, interference=None, merges=None, gaps=None):
    """Return dict(edition, checks=[dict(level, area, item, message)], counts)."""
    checks = []

    def add(level, area, item, msg):
        checks.append(dict(level=level, area=area, item=item, message=msg))

    lib = lib or {}
    profiles = lib.get("profiles", {})
    tol_w, tol_f = args.length_warn, args.length_fail

    # ---------------- generic
    if parts:
        lo = np.min([p.bbox[0] for p in parts], axis=0)
        hi = np.max([p.bbox[1] for p in parts], axis=0)
        size = hi - lo
        if size.max() < 60:
            add(FAIL, "units", "model size", f"model is only {size.max():.1f} mm across - exported in metres or inches?")
        elif size.max() > 5000:
            add(FAIL, "units", "model size", f"model is {size.max():.0f} mm across - exported in micrometres?")
        else:
            add(PASS, "units", "model size", f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm")

    # duplicates: same volume and centroid
    seen = {}
    for p in parts:
        k = (round(p.volume, 0), *np.round(p.c, 1))
        if k in seen:
            add(FAIL, "duplicates", p.name, f"identical to '{seen[k]}' (same volume and position) - a doubled body "
                                            f"doubles its stiffness in the analysis; delete one")
        else:
            seen[k] = p.name

    touched = set()
    for i, j, _ in contacts:
        touched.add(i); touched.add(j)
    for p in parts:
        if p.idx not in touched and p.kind in ("beam", "plate") or (p.idx not in touched and p.volume > 2000):
            add(WARN, "connections", p.name, f"touches no other part within {args.contact_tol} mm - floating in "
                                             f"the model (gap, or a missing bracket)")

    # section vs claimed profile, and length in the name
    for p in parts:
        if p.kind != "beam":
            continue
        claim = None
        m = re.search(r"\b(2020|2040|3030|4040|2060|2080)\b", p.label)
        if m:
            claim = m.group(1)
        m2 = re.search(r"\bmgn ?(9|12|15)", p.label.lower())
        if m2 and "rail" in p.label.lower():
            claim = f"MGN{m2.group(1)}"
        if claim and claim in profiles:
            if p.lib_match != claim:
                env = sorted(profiles[claim]["envelope"])
                add(FAIL, "profile", p.name, f"name says {claim} but the section is {p.ext[1]:.1f} x {p.ext[2]:.1f} mm"
                                             f" (expected {env[0]:g} x {env[1]:g}, area {profiles[claim]['A']:.0f} mm2;"
                                             f" CAD {p.A:.0f} mm2) - wrong profile or a simplified solid bar")
            else:
                add(PASS, "profile", p.name, f"{claim} section matches the library ({p.lib_note})")
        m = re.search(r"\bx (\d+(?:\.\d+)?)\b", p.label)
        if m and not edition:
            want = float(m.group(1))
            if abs(want - p.L) > 50 and abs(want - max(p.ext[1], p.ext[2])) < 5:
                continue            # the number is a section size, not a length
            d = p.L - want
            if abs(d) > tol_f:
                add(FAIL, "length", p.name, f"name says {want:g} mm, measured {p.L:.1f} mm ({d:+.1f})")
            elif abs(d) > tol_w:
                add(WARN, "length", p.name, f"name says {want:g} mm, measured {p.L:.1f} mm ({d:+.1f})")

    # split and broken bars
    for m in merges or []:
        add(WARN, "split solids", m["name"], f"one bar ({m['length']:g} mm) is {len(m['pieces'])} separate solids in the "
                                             f"CAD: {', '.join(m['pieces'])}. Merged for the analysis; a single body "
                                             f"(and a name) in the CAD is cleaner")
    for g in gaps or []:
        add(FAIL, "broken bar", f"{g['a']} / {g['b']}",
            f"two collinear pieces of the same bar section leave a {g['gap']:g} mm gap"
            + (f", filled only by thinner bodies ({', '.join(g['fillers'][:4])})" if g["fillers"] else "")
            + ". In the real part this is one continuous bar: the model's section is incomplete there")

    # interference
    if interference:
        for i, j, vol in interference:
            add(FAIL if vol > 50 else WARN, "interference", f"{parts[i].name} / {parts[j].name}",
                f"{vol:.1f} mm3 of metal overlap (parts intersect in the CAD)")

    # ---------------- edition
    ed = (lib.get("editions") or {}).get(edition) if edition else None
    lengths = {}
    if ed:
        keys = {p["key"] for p in ed["parts"]}
        inst = {}          # instance label -> list(parts)
        for p in parts:
            inst.setdefault(p.instance, []).append(p)
        names_all = list(inst) + [_strip(n) for n, _ in excluded]
        by_key = {}
        for name in set(names_all):
            k = _key_of(name, keys)
            if k:
                by_key.setdefault(k, set()).add(name)
        rules = ed.get("rules", {})
        skip = rules.get("not_checked", {})
        exp_qty = {}
        for line in ed["parts"]:
            if line["key"] in skip:
                continue
            exp_qty[line["key"]] = exp_qty.get(line["key"], 0) + line["qty"]
        # unnamed bodies: recognise them by shape (bar profile + length, or catalog block)
        catalog = rules.get("catalog", {})
        unnamed = sorted({p.instance for p in parts if p.instance.startswith("solid_")})
        for name in unnamed:
            ps = inst[name]
            p = max(ps, key=lambda q: q.volume)
            need = lambda k: exp_qty.get(k, 0) - len(by_key.get(k, ()))
            hit = None
            if p.kind == "beam" and p.lib_match:
                for line in ed["parts"]:
                    if line.get("profile") == p.lib_match and abs(p.L - line.get("length", -1e9)) <= tol_f and need(line["key"]) > 0:
                        hit = line["key"]; break
            elif p.lib_match:
                for k, blk in catalog.items():
                    if blk == p.lib_match and need(k) > 0:
                        hit = k; break
            if hit:
                by_key.setdefault(hit, set()).add(name)
                add(WARN, "names", f"{hit} {name}", f"recognised by shape as key {hit} "
                                                    f"({p.lib_match}{', %.0f mm' % p.L if p.kind == 'beam' else ''}); "
                                                    f"it has no name in the CAD")
        for k, why in skip.items():
            add(PASS, "parts list", k, f"not checked: {why}")
        for k, q in sorted(exp_qty.items()):
            got = len(by_key.get(k, ()))
            labels = "; ".join(sorted({l["label"] for l in ed["parts"] if l["key"] == k}))
            if got == q:
                add(PASS, "parts list", k, f"{got} x {labels}")
            else:
                add(FAIL if got < q else WARN, "parts list", k,
                    f"expected {q}, found {got} ({labels})")
        assigned = set().union(*by_key.values()) if by_key else set()
        rest = [n for n in inst if not _key_of(n, keys) and n not in assigned]
        big = sorted({_base_label(n) for n in rest if max(p.volume for p in inst[n]) > 2000})
        small = len(rest) - len([n for n in rest if max(p.volume for p in inst[n]) > 2000])
        if big:
            add(WARN, "parts list", "not in the manual",
                f"{len(big)} part(s) have no key letter from the manual: " + ", ".join(big[:15])
                + (" ..." if len(big) > 15 else ""))
        if small:
            add(PASS, "parts list", "small unnamed bodies",
                f"{small} small unnamed bodies (carriage end caps, seals and similar) not checked")
        # profile + length per keyed part
        for line in ed["parts"]:
            if "length" not in line:
                continue
            k, want, prof = line["key"], line["length"], line.get("profile")
            for name in sorted(by_key.get(k, ())):
                ps = [p for p in inst.get(name, []) if p.kind == "beam"]
                if not ps:
                    add(WARN, "length", f"{k} {name}", "not analysed as a bar (too small or filtered out)")
                    continue
                p = max(ps, key=lambda q: q.L)
                lengths.setdefault(k, []).append(p.L)
                if prof in profiles and prof != "T8" and p.lib_match != prof:
                    add(FAIL, "profile", f"{k} {p.name}", f"should be {prof}; section {p.ext[1]:.1f} x {p.ext[2]:.1f} mm "
                                                         f"matches {p.lib_match or 'no library profile'}")
                d = p.L - want
                lvl = PASS if abs(d) <= tol_w else (WARN if abs(d) <= tol_f else FAIL)
                add(lvl, "length", f"{k} {p.name}", f"cut list {want:g} mm, model {p.L:.1f} mm ({d:+.1f})")
        for k in rules.get("equal_length_keys", []):
            v = lengths.get(k, [])
            if len(v) > 1:
                spread = max(v) - min(v)
                add(PASS if spread <= tol_w else FAIL, "rules", f"{k} equal lengths",
                    f"{len(v)} parts, spread {spread:.1f} mm")
        for rail, carrier in rules.get("rail_not_longer_than", []):
            if lengths.get(rail) and lengths.get(carrier):
                r, c = max(lengths[rail]), min(lengths[carrier])
                add(PASS if r <= c + 0.2 else FAIL, "rules", f"{rail} rail <= {carrier}",
                    f"rail {r:.1f} mm on a {c:.1f} mm extrusion" + ("" if r <= c + 0.2 else " - rail overhangs"))
        for rel in rules.get("relations", []):
            expr = rel["check"]
            toks = set(re.findall(r"[A-Z][A-Z0-9]*", expr))
            if not re.fullmatch(r"[A-Z0-9 .+\-*/=()]+", expr) or not all(lengths.get(t) for t in toks):
                continue
            env = {t: float(np.mean(lengths[t])) for t in toks}
            lhs, rhs = expr.split("==")
            a, b = eval(lhs, {"__builtins__": {}}, env), eval(rhs, {"__builtins__": {}}, env)
            add(PASS if abs(a - b) <= tol_w else FAIL, "rules", expr,
                f"{a:.1f} vs {b:.1f} mm - {rel.get('why', '')}")

    counts = {lvl: sum(1 for c in checks if c["level"] == lvl) for lvl in (PASS, WARN, FAIL)}
    return dict(edition=edition, checks=checks, counts=counts, lengths={k: v for k, v in lengths.items()})


def interference_volumes(gmsh, parts, contacts, metals=("aluminum", "steel")):
    """Intersection volume for touching metal bar pairs (call while gmsh still holds the model)."""
    occ = gmsh.model.occ
    out = []
    for i, j, _ in contacts:
        a, b = parts[i], parts[j]
        if a.material not in metals or b.material not in metals or "beam" not in (a.kind, b.kind):
            continue
        try:
            res, _m = occ.intersect([(3, a.tag)], [(3, b.tag)], removeObject=False, removeTool=False)
            occ.synchronize()
            vol = sum(occ.getMass(d, t) for d, t in res if d == 3)
            if res:
                occ.remove(res, recursive=True)
                occ.synchronize()
        except Exception:
            continue
        if vol > 1.0:
            out.append((i, j, vol))
    return out


def markdown(result, step_name):
    c = result["counts"]
    L = [f"# Model check - {step_name}", ""]
    L.append(f"Edition: **{result['edition'] or 'none (generic checks only)'}** - "
             f"{c['FAIL']} fail, {c['WARN']} warn, {c['PASS']} pass\n")
    icon = {"FAIL": "❌", "WARN": "⚠️", "PASS": "✅"}
    for lvl in ("FAIL", "WARN", "PASS"):
        rows = [x for x in result["checks"] if x["level"] == lvl]
        if not rows:
            continue
        L.append(f"## {icon[lvl]} {lvl} ({len(rows)})\n")
        L.append("| Area | Item | Detail |")
        L.append("|---|---|---|")
        for x in rows:
            L.append(f"| {x['area']} | {x['item']} | {x['message']} |")
        L.append("")
    return "\n".join(L)
