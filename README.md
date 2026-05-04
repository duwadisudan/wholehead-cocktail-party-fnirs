# Whole-head cocktail-party fNIRS spatial-attention decoding

Publication code for **"Decoding Spatial Attention in the Cocktail Party Problem
Using Wearable Whole-head High-Density fNIRS"** (preprint forthcoming).

Whole-head high-density fNIRS data, cocktail-party paradigm, Random Forest
decoder.

- **Paper:** preprint forthcoming on bioRxiv — citation will be added here on release.
- **Raw data (BIDS):** [OpenNeuro ds007738](https://openneuro.org/datasets/ds007738) — DOI `10.18112/openneuro.ds007738.v1.0.0`
- **Derivatives:** Zenodo deposit forthcoming *(will enable the "moderate" and "fast" reviewer modes — see [Reviewer modes](#reviewer-modes))*

> **Always use the latest version of the OpenNeuro dataset.** The DOI above
> resolves to a specific version snapshot. If a newer version is available
> on the [dataset page](https://openneuro.org/datasets/ds007738), use that
> version — bug fixes and metadata corrections are released without a code
> change here.

> **About cedalion.** This repo includes a frozen, vendored copy of the
> [cedalion](https://github.com/ibs-lab/cedalion) fNIRS analysis library
> under [`cedalion/`](cedalion/). It is intentionally pinned to the exact
> version used to produce the paper figures and is **not** kept in sync
> with upstream. Please install it via the editable-install command in
> [Step 3](#3-install-the-vendored-cedalion) below. Cedalion is third-party
> software authored by collaborators and is cited separately in the
> manuscript — vendoring it here does not replace that citation.

---

## Status of this README

This is the **setup and end-to-end run** guide. It covers cloning the repo,
building the environment, configuring paths, and running the full pipeline
from raw BIDS data.

The shortcut paths — downloading shipped derivatives so reviewers can skip
preprocessing or skip classification — will be added once the Zenodo
deposit is finalized. See [Reviewer modes](#reviewer-modes) for the planned
structure.

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11, macOS (Intel and Apple Silicon — see note below), Linux |
| Python | 3.11 |
| Package manager | `conda` (Miniconda or Anaconda) |
| Disk | ~20 GB recommended (raw data + derivatives + figures) — exact size depends on the OpenNeuro release version |
| RAM | 16 GB minimum, 32 GB recommended |

> Apple Silicon: the pinned environment is expected to work but has not yet
> been independently verified end-to-end on M-series Macs. Please report
> install or runtime issues if you hit them on Apple Silicon.

Cross-platform compatibility is a hard requirement of this repo; if a script
fails on macOS or Windows, that is a bug — please [open an issue](#issues-and-questions).

---

## 1. Clone the repository

```bash
git clone https://github.com/duwadisudan/wholehead-cocktail-party-fnirs.git
cd wholehead-cocktail-party-fnirs
```

The repo includes a vendored copy of [cedalion](https://github.com/ibs-lab/cedalion)
inside [`cedalion/`](cedalion/) and a vendored copy of the
[`whichscript`](whichscript/) provenance helper. Both are intentionally
frozen — **do not** run `git pull` or any update command inside those
folders.

## 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate cedalion
```

If the solver fails on your platform, please [report it](#issues-and-questions)
rather than loosening pins — the version pins are part of the
reproducibility contract for the paper. Raw and per-package snapshots of the
exact build that produced the figures are kept in [`env_snapshot/`](env_snapshot/)
for reference.

## 3. Install the vendored cedalion

From the repo root, with the `cedalion` conda env activated:

```bash
pip install -e ./cedalion
```

This installs cedalion in editable mode from the frozen local copy. Do not
replace it with a release from PyPI or upstream GitHub — the figures in the
paper were produced against this exact snapshot.

## 4. Install the vendored whichscript

```bash
pip install -e ./whichscript
```

`whichscript` provides per-output script provenance (a hidden script-snapshot
sidecar next to every figure / table the pipeline saves). Some scripts
import it directly.

## 5. Install this project

Still from the repo root:

```bash
pip install -e .
```

This makes the shared modules in [`src/wholehead_cocktail_party/`](src/wholehead_cocktail_party/)
importable from any of the entry-point scripts in [`scripts/`](scripts/).

## 6. Verify the install

```bash
python -c "import cedalion, wholehead_cocktail_party, whichscript; print('ok')"
```

A smoke test in [`tests/`](tests/) will be added alongside a shipped
single-subject fixture in a later update.

---

## Configure paths

All scripts read input/output locations from a single config file:

```
config/paths.yml
```

A template will be provided in the next update. You will edit this file
**once** to point at your local copy of the data; no other path changes
should be needed.

The expected on-disk layout under your chosen `data_root` is:

```
<data_root>/
├── raw/                        # from OpenNeuro (BIDS) — use latest version
└── derivatives/
    ├── preprocessed/           # produced by scripts/preprocessing/
    └── classifier/             # produced by scripts/classification/
```

Scripts will never write outside `data_root` or the repo's [`figures/`](figures/) folder.

---

## Reviewer modes

The pipeline is designed around three entry points so a reviewer can choose
how much of the pipeline to re-run:

| Mode | Starts from | Approx. time | What it does |
|---|---|---|---|
| **Full** | Raw BIDS (OpenNeuro) | hours – ~1 day | Runs everything end-to-end |
| **Moderate** | Preprocessed derivatives (Zenodo) | ~hours | Re-runs classifier + figures |
| **Fast** | Classifier outputs (Zenodo) | ~5 minutes | Re-renders figures only |

**Currently this README documents the Full mode only.** Moderate and Fast
will be enabled once the derivatives are deposited on Zenodo; the
corresponding sections will be added here in the next update. Three
top-level driver scripts (`01_preprocess.py`, `02_classify.py`,
`03_make_figures.py`) are also planned so each mode reduces to a single
command.

---

## Run the pipeline (Full mode)

After paths are configured, run the stages in order from the repo root.
Master `01_/02_/03_` driver scripts are not yet in place; for now the
stages are run as the individual scripts below.

### Stage 1 — Preprocessing

Per-condition gaze quality control followed by per-trial SNR computation:

```bash
python scripts/preprocessing/gaze_outlier_detection_covert.py
python scripts/preprocessing/gaze_outlier_detection_overt_orient.py
python scripts/preprocessing/gaze_outlier_detection_overt_control.py
python scripts/preprocessing/gaze_outlier_detection_tobii.py
python scripts/preprocessing/compute_trial_snr.py
```

`gaze_outlier_detection_matlab_variant.py` is an alternate implementation
kept for cross-checking against the original MATLAB pipeline; it is not on
the figure-producing path.

### Stage 2 — Classification

Per-subject Random Forest classification (all-channels and angular-gyrus
ROI), plus the permutation test:

```bash
python scripts/classification/classify_rf_all_channels.py
python scripts/classification/classify_rf_angular_gyrus.py
python scripts/classification/permutation_test_rf.py
```

### Stage 3 — Figures

```bash
# Figure 2 — hemodynamic response
python scripts/figure2_hemodynamic_response/group_level_brodmann_grid.py
python scripts/figure2_hemodynamic_response/plot_rois_scalp_2d.py

# Figure 3 — accuracy over time (all channels)
python scripts/figure3_accuracy_over_time/plot_accuracy_over_time.py

# Figure 4 — accuracy scatter + latency CI
python scripts/figure4_accuracy_scatter_latency/scatter_overt_with_latency_CI.py
python scripts/figure4_accuracy_scatter_latency/scatter_covert.py
python scripts/figure4_accuracy_scatter_latency/make_accuracy_summary_table.py

# Figure 5 — PC contributions
python scripts/figure5_pc_contributions/analyze_pc_contributions_histogram.py
python scripts/figure5_pc_contributions/plot_channel_contributions_scalp.py

# Figure 6 — overt vs covert
python scripts/figure6_overt_vs_covert_scatter/scatter_overt_vs_covert_LR_top5.py
python scripts/figure6_overt_vs_covert_scatter/pairwise_accuracy_overt_vs_covert.py

# Figure 7 — accuracy over time (angular gyrus ROI)
# scripts in scripts/figure7_accuracy_over_time_ang_gyr/

# Supplementary figures
# scripts in scripts/supplementary/

# MNI / angular-gyrus utilities
# scripts in scripts/misc/
```

A full per-figure recipe and a single end-to-end driver will be added in
`REPRODUCE.md` in the next update.

---

## Repository layout

```
.
├── README.md                          # this file
├── pyproject.toml                     # makes src/ importable
├── environment.yml                    # pinned conda env
├── env_snapshot/                      # raw env snapshots used to build environment.yml
├── cedalion/                          # vendored, frozen — do not update
├── whichscript/                       # vendored provenance helper (separate package)
├── src/
│   └── wholehead_cocktail_party/      # shared modules imported by scripts
├── scripts/
│   ├── preprocessing/                 # gaze QC, SNR
│   ├── classification/                # Random Forest training + permutation test
│   ├── figure2_hemodynamic_response/
│   ├── figure3_accuracy_over_time/
│   ├── figure4_accuracy_scatter_latency/
│   ├── figure5_pc_contributions/
│   ├── figure6_overt_vs_covert_scatter/
│   ├── figure7_accuracy_over_time_ang_gyr/
│   ├── misc/                          # MNI / angular-gyrus extraction utilities
│   └── supplementary/                 # supplementary-figure scripts
├── config/
│   └── paths.yml                      # (template forthcoming) reviewer edits this once
├── tests/                             # smoke tests on a shipped fixture (forthcoming)
├── data/                              # not under version control; populated by user
└── figures/                           # rendered outputs
```

---

## License

MIT. The vendored [`cedalion/`](cedalion/) and [`whichscript/`](whichscript/)
folders are also MIT-licensed under their original copyrights — see
[`cedalion/LICENSE.md`](cedalion/LICENSE.md) and
[`whichscript/LICENSE`](whichscript/LICENSE).

A top-level `LICENSE` file will be added alongside the public release.

---

## Citation

If you use this code in academic work, please cite the paper (citation will
be filled in on bioRxiv preprint release):

> Decoding Spatial Attention in the Cocktail Party Problem Using Wearable
> Whole-head High-Density fNIRS. *bioRxiv* (forthcoming).

A `CITATION.cff` will be added so GitHub's "Cite this repository" button
resolves to the published paper and the archived release DOI.

---

## Issues and questions

Please [open an issue](https://github.com/duwadisudan/wholehead-cocktail-party-fnirs/issues)
for bugs, install failures, or anything that does not reproduce. For
questions about the science, contact Sudan Duwadi (sudan@bu.edu).
