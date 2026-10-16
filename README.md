# 3D Printer Structure Review

Find the weak spots in a 3D printer frame from its CAD file, before you build it.

Give it a STEP export of your printer. It works out which bodies are extrusions, rails, plates and printed brackets, connects them where they touch, and runs a frame model of the load cases that matter for print quality:

- **Gantry acceleration in X and Y**: how far the top of the frame moves when the toolhead changes direction.
- **Twist (racking)**: what a CoreXY frame feels from the belts.
- **Sag under gravity**.
- **Natural frequencies**: where ringing comes from, and what input shaping has to cancel.

Then it tries changes and measures each one: a bigger profile on the weakest members, stiffer joints, diagonal braces or bolted panels on each side. Each option is ranked by how much stiffness it actually buys. A **local** LLM (Ollama or any OpenAI-compatible server on your own machine) can optionally turn the numbers into a prioritised write-up. No cloud AI is used, and the physics never depends on the LLM.

Example on the bundled demo (a 400 × 400 × 450 mm 2020 cube with MGN rails, which runs in about 6 seconds):

```
  accel_Y        0.39 N/um   (38.2 um)
  torsion_Z       555 N*m/deg
  gravity_Z     41.32 N/um   (2.3 um)
  frame modes: 50 Hz X sway, 54 Hz Y sway, 70 Hz twist, 152 Hz X sway
  best changes:
       +78%  Add X-diagonal braces (2020 profile) or a bolted shear panel on the left (X min) side
       +77%  Add X-diagonal braces (2020 profile) or a bolted shear panel on the front (Y min) side
       +25%  Upsize 'post_2020x450 #1' one profile size (~3x bending stiffness, e.g. 2020->2040/3030)
```

## Install

You need Python 3.10 or newer and git.

**Windows (PowerShell):**

```powershell
cd $env:USERPROFILE; git clone https://github.com/geoffreypboals-web/3DPrinterStructureReview.git; cd 3DPrinterStructureReview; python -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements.txt
```

**Linux / macOS:**

```bash
cd ~ && git clone https://github.com/geoffreypboals-web/3DPrinterStructureReview.git && cd 3DPrinterStructureReview && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

A headless Linux machine (no desktop) also needs gmsh's graphics libraries, even though nothing is drawn:

```bash
sudo apt install -y libglu1-mesa libxcursor1 libxinerama1 libxft2
```

Check the install with the smoke test. It builds the demo frame and analyses it, and prints `ok` if everything works.

**Windows:**

```powershell
cd $env:USERPROFILE\3DPrinterStructureReview; .\.venv\Scripts\python test_smoke.py
```

**Linux / macOS:**

```bash
cd ~/3DPrinterStructureReview && .venv/bin/python test_smoke.py
```

## Analyse your printer

1. **Export a STEP of the whole printer** from your CAD tool (FreeCAD, Fusion, Onshape, SolidWorks...), as an assembly with every body. Body names help: a body called `2040 x 500` or `MGN12 rail` is recognised by name, and unnamed bodies are recognised by shape. Leave screws in or out; fasteners are skipped.
2. **Check the model**, then **run the analysis**. Replace the path with your own file. The `--no-ai` flag gives numbers only; to get the AI write-up as well, see the next section.

**Windows:**

```powershell
cd $env:USERPROFILE\3DPrinterStructureReview; .\.venv\Scripts\python validate_model.py C:\path\to\my_printer.step --edition none; .\.venv\Scripts\python printer_stiffness.py C:\path\to\my_printer.step --edition none --no-ai
```

**Linux / macOS:**

```bash
cd ~/3DPrinterStructureReview && .venv/bin/python validate_model.py /path/to/my_printer.step --edition none && .venv/bin/python printer_stiffness.py /path/to/my_printer.step --edition none --no-ai
```

Results are written to folders next to your STEP file:

| Folder | What's in it |
|---|---|
| `my_printer_check/` | The model check: wrong units, duplicate or floating bodies, bars split into pieces or with gaps, profiles it couldn't identify. Fix FAIL lines before trusting the stiffness numbers. |
| `my_printer_stiffness/report.md` | The readable report: stiffness per load case, which members carry the flex, the mode shapes, and the ranked what-if changes. |
| `my_printer_stiffness/results.json` | The same numbers, for scripts. |
| `my_printer_stiffness/deformed_shapes.png` | A picture of how the frame bends in each load case. |

A large assembly takes a while to read: a 45 MB STEP takes about 4 minutes, but the analysis itself takes seconds. Geometry is cached, so a second run is fast.

### Settings that change the answer

The defaults suit a typical CoreXY printer: 1.5 kg moving mass at 10 m/s². Set these to match yours:

| Option | Meaning |
|---|---|
| `--moving-mass 0.8` | Mass of the toolhead plus whatever moves with it, in kg. |
| `--accel 20000` | Acceleration in mm/s² (20000 = 20 m/s²). |
| `--part-material "bracket=pet_cf"` | Material for bodies whose name matches the regex. Choices: aluminum, steel, asa, abs, petg, pla, pet_cf, pa_cf, pc_cf, tpu, carbon, glass, silicone, motor. |
| `--print-profile solid` | Walls and infill of your printed parts: `duender_load_bearing` (default: 5 walls, 60 % gyroid), `medusahc`, `box` or `solid`. This changes how stiff printed brackets count. |
| `--joint-factor 0.6` | Bolted-joint stiffness relative to the parts themselves (default 0.3; 1 = as stiff as the parts). |
| `--interference` | (validator only) Report metal parts that overlap. |

`--help` on either script lists every option.

## Optional: the local AI write-up

The LLM only interprets the numbers; it never changes them. You need [Ollama](https://ollama.com/download) running on this machine or another machine you can reach.

1. Install Ollama from https://ollama.com/download and start it. On Windows and macOS it runs in the tray after install; on Linux the installer starts a service.
2. Download a model (about 5 GB). The default is `llama3.1:8b`; any instruction-tuned model works.

```bash
ollama pull llama3.1:8b
```

3. Run the analysis without `--no-ai`.

**Windows:**

```powershell
cd $env:USERPROFILE\3DPrinterStructureReview; .\.venv\Scripts\python printer_stiffness.py C:\path\to\my_printer.step --edition none --model llama3.1:8b
```

**Linux / macOS:**

```bash
cd ~/3DPrinterStructureReview && .venv/bin/python printer_stiffness.py /path/to/my_printer.step --edition none --model llama3.1:8b
```

Ollama on another machine: add `--ai-host http://that-machine:11434`, or set the `OLLAMA_HOST_URL` environment variable. For llama.cpp, vLLM or LM Studio, add `--api openai --ai-host http://localhost:PORT`. `PS_AI_MODEL` and `PS_AI_API` set the default model and API.

## The part library

`library/` holds what the tool knows about parts. **Every `*.json` file in that folder is merged**, so you can add your own (for example a `my_parts.json` listing the profiles and printed parts you use) without editing the shipped ones. Later files override earlier keys.

- `base_library.json` (hand-edited) holds:
  - materials (modulus and density)
  - print profiles: walls and infill for load-bearing printed parts
  - name rules that decide which body is a rail, a motor, a printed part or a fastener
  - shape rules for unnamed parts such as belts
- `duender_generated.json` (generated) holds the measured sections of real 2020 and 2040 V-slot, MGN9 and MGN12 rails, bounding-box signatures for carriages, NEMA17 motors and couplers, and the parts lists of three Duender-style printer editions.

Printed parts get an effective stiffness from their walls and infill: `E_eff = E × (shell + (1 − shell) × infill^1.6) × 0.8`. For example, a 20 mm ASA bracket at 5 walls and 60 % infill counts as about 45 % of solid ASA.

### Checking a Duender-style C300 / C400 / C500 build

These scripts came out of the [Toolchanger-Research](https://github.com/geoffreypboals-web/Toolchanger-Research) project. For its C300 / C400 / C500 printers, `--edition c300|c400|c500` (or `auto`, the default) also checks the STEP against that edition's manual: every part's key letter, quantity and cut length. For any other printer, pass `--edition none`.

`build_library.py` regenerates `duender_generated.json` from that project's CAD. You only need it if you change those designs:

```bash
cd ~/3DPrinterStructureReview && TOOLCHANGER_REPO=~/Toolchanger-Research .venv/bin/python build_library.py
```

## How it works, and its limits

The STEP is read with gmsh/OpenCASCADE. Each body is classified as a beam, plate or block. Beam section properties come from the body's own geometry, or from the library when a profile is recognised.

Bodies that touch are connected:

- **Beams** become 3-D beam elements, split at every contact.
- **Bolted joints** become short flexible links, because bolted joints aren't rigid.
- **Blocks and plates** become a hub node with spokes to whatever they touch.

The load cases are solved as a sparse linear system, and the frequencies come from a modal solve.

This is a **reduced-order frame model, not solid FEA**. Joint stiffness and the torsion of slotted extrusions are estimates. Use the results to **rank** weaknesses and compare design options, not as certified absolute values. Confirm on the real printer with an accelerometer resonance test (for example Klipper's `SHAPER_CALIBRATE`).

## Licence

Apache License 2.0 (see `LICENSE`). See `NOTICE.md` for where the library data comes from, and `SECURITY.md` for reporting a vulnerability.
