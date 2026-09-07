# MFXerrorsSuppl



This repository contains the Python code used to generate the quantitative
figures in:

> Doughty, L.A., Mebus, V.H., Alibhai, D., Brewer, J.R. *From MINFLUX
> localisation precision to biological positional accuracy: a critical
> review on error sources, diagnostics, and reporting.*

## Contents

| File | Produces |
|---|---|
| `fig_precision_retention.py` | Figure 4 (precision-retention curve) |
| `master_figures.py` | Figures 1, 2, 3 and 5 (panel b) |
| `input/xy_nanoparticle_alignment_donut.tif` | Raw beam back-scatter image used by `master_figures.py` |

## Requirements

Python 3.9 or later, with the following packages loaded:

**macOS / Linux:**
 
```bash
python3 -m pip install numpy scipy matplotlib tifffile
```
 
**Windows:**
 
```bash
py -m pip install numpy scipy matplotlib tifffile
```

## Folder Structure
Before running, organise folders/files as follows:
```
your_folder/
├── master_figures.py
├── fig_precision_retention.py
├── input/
│   └── xy_nanoparticle_alignment_donut.tif
└── output/          (created automatically if it does not exist)
```


## Usage
From within `your_folder/`, run:

```bash
python master_figures.py
python fig_precision_retention.py
```

Each script produces a short confirmation message per figure when it is 
complete e.g.:

```
✓ fig2a_centrepull
✓ fig2b_degradation
✓ figure3 (panels a + b)
✓ fig4b_distributions
✓ donut_comparison (fig2 beam quality)
```

All output figures are written to `output/` in PDF, PNG and SVG (where 
applicable).

## Figure Descriptions

## Data

`xy_nanoparticle_alignment_donut.tif` is a real experimental 2D (XY) back-scatter
image of a 640 nm MINFLUX excitation beam, acquired via a 25 nm gold
nanoparticle fiducial at 100 px/µm and 8% laser power. The excitiation beam has
been intentionally abberated by a wavefront distortion via the spatial 
light modulator to cause a small residual intensity for non-perfect 
illustrative purposes. Replace this file with any experimentally obtained 
2D back-scatter image from an Abberior MINFLUX instrument.


## Notes

- Random seeds are fixed in both scripts (`np.random.default_rng(...)`)
  for reproducibility of the simulated figures.
- If re-running produces a `FileNotFoundError`, check that the `INPUT`
  and `OUT` path variables at the top of `master_figures.py` correctly
  point to your local `input/` and `output/` folders.
