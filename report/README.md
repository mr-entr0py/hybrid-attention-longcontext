# Pre-project report (IE624)

- `main.pdf` is the compiled report.
- `main.tex` is the source of the 6-page submission; `report_long.tex` is the longer draft. `iereport.sty` is the house style: a NeurIPS-like layout set in STIX Two.
- `refs.bib` holds the bibliography; every entry was verified against its primary source (see `citation_check.md`).
- `figures/` holds the figures. `make_report_figures.py` regenerates every data figure from the repository's raw experiment files. `iitb_logo.svg` is the source of `iitb_logo.pdf`.

Build: `tectonic -X compile main.tex` (or `latexmk -xelatex main.tex`). XeTeX is required for the OpenType fonts.
