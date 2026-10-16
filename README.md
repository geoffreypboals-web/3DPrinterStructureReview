# printer-stiffness (T-020)

Stand-alone tools (no Claude needed) for checking and improving a printer frame from its STEP file.
The physics runs in Python; the improvement write-up comes from the **local** LLM (Ollama on lambda2).

| Script | What it does |
|---|---|
| `validate_model.py` | Checks the CAD before any analysis: units, duplicate bodies, floating parts, profiles vs the part library, bars split into pieces or with gaps, and for the Duender editions every part of the manual's picture list (key letters, quantities, cut lengths, design rules). |
| `printer_stiffness.py` | Runs the same check, then a frame finite-element model (gantry inertia X/Y, twist, gravity, natural frequencies), what-if studies (bigger profiles, stiffer joints, braces/panels) and asks the local LLM for a prioritised improvement plan. |
| `build_library.py` | Regenerates `library/duender_generated.json` from this repo's CAD (real V-slot/rail sections, carriage/motor signatures, the C300/C400/C500 parts lists). Re-run after `build_lego_mgn9.py` changes a size. |

## Library (`library/`)

Every `*.json` in the folder is merged, so you can drop in more (e.g. a `mcmaster.json` with parts you use).

- `base_library.json` (hand-edited): materials, print profiles from the Duender manuals (ASA/ABS, 5 walls,
  60 % gyroid for load-bearing parts; MedusaHC parts 98 %; PET-CF for docks/bed mounts; PLA only for boxes),
  name rules (which body is a rail, a motor, a printed part, a fastener...) and shape rules (unnamed belts).
- `duender_generated.json` (generated, don't edit): sections measured from `cad/designs/duender-large/lib/`
  (2020: A 175 mm², I 7021 mm⁴; 2040: A 295 mm², I 11781 / 48822 mm⁴; MGN12 and MGN9 rails), carriage /
  NEMA17 / coupler signatures, and the three editions' parts lists from `out/c*-mgn9x-steps.json`.

Printed parts get an effective modulus for their walls and infill
(`E_eff = E x (shell + (1 - shell) x infill^1.6) x 0.8`), so a 20 mm ASA bracket at 5 walls / 60 % counts as
roughly 45 % of solid ASA. It's an estimate; change `printed_model` or the profiles if you print differently.

## Install (lambda2)

```
cd /home/$USER/repos/Toolchanger-Research        # wherever the repo is cloned
git -C /home/$USER/repos/Toolchanger-Research lfs pull
python3 -m venv /home/$USER/repos/Toolchanger-Research/tools/printer-stiffness/.venv
/home/$USER/repos/Toolchanger-Research/tools/printer-stiffness/.venv/bin/pip install -r /home/$USER/repos/Toolchanger-Research/tools/printer-stiffness/requirements.txt
```
Headless Linux also needs `sudo apt install -y libglu1-mesa libxcursor1 libxinerama1 libxft2` for gmsh.
Windows: same with `D:\repos\Toolchanger-Research\tools\printer-stiffness\.venv\Scripts\python`.

## Run

```
PY=/home/$USER/repos/Toolchanger-Research/tools/printer-stiffness/.venv/bin/python
T=/home/$USER/repos/Toolchanger-Research/tools/printer-stiffness
$PY $T/validate_model.py /home/$USER/repos/Toolchanger-Research/cad/designs/duender-large/out/c500-duender-3z-mgn9x.step
$PY $T/printer_stiffness.py /home/$USER/repos/Toolchanger-Research/cad/designs/duender-large/out/c500-duender-3z-mgn9x.step --model llama3.1:8b
```
Results go next to the STEP file (`<name>_check/`, `<name>_stiffness/`). Those folders are outputs; don't commit
them. A 45 MB Duender STEP takes about 4 minutes (reading + contact search), the analysis itself seconds.

Useful options: `--edition c300|c400|c500|none`, `--interference` (validator: metal overlaps),
`--moving-mass`, `--accel`, `--part-material "regex=pet_cf"`, `--print-profile medusahc`, `--joint-factor`,
`--ai-host http://lambda02:11434`, `--no-ai`. `--help` lists everything.

## Limits

Reduced-order frame model (beams + joint links + hub nodes), not solid FEA. Joint stiffness and the torsion of
slotted extrusions are estimates. Use it to rank weaknesses and compare options, then confirm with an
accelerometer resonance test.
