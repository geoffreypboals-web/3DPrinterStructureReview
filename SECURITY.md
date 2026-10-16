# Security

Report a vulnerability privately through GitHub: **Security → Report a vulnerability** on this repo
(https://github.com/geoffreypboals-web/3DPrinterStructureReview/security/advisories/new). Please don't open a
public issue for it.

What the tool trusts:

- **STEP files** are parsed by gmsh/OpenCASCADE. Only open CAD from sources you'd open in your CAD program.
- **Library JSON files** (`library/*.json`) are data only. Rule expressions are evaluated as plain
  arithmetic on part lengths, never as Python code.
- **The geometry cache** (`~/.cache/printer-stiffness`, or `PS_CACHE_DIR`) uses Python pickle. On Linux/macOS
  a cache file is only loaded if you own it and nobody else can write to it. Don't point `PS_CACHE_DIR` at a
  shared folder.
- **The AI step** sends the numeric summary (no CAD geometry) to `--ai-host`. With `--api openai`,
  `OPENAI_API_KEY` is sent to that host if it's set, so only point it at a server you trust.
