# Whole-head cocktail-party fNIRS spatial-attention decoding

Code for the paper "Decoding Spatial Attention in the Cocktail Party Problem
Using Wearable Whole-head High-Density fNIRS".

Raw data: [OpenNeuro ds007738](https://openneuro.org/datasets/ds007738)
(DOI `10.18112/openneuro.ds007738.v1.0.0`). Always use the latest version
available on the dataset page.

The [`cedalion/`](cedalion/) folder is a frozen, vendored copy of the
[cedalion](https://github.com/ibs-lab/cedalion) fNIRS analysis library,
pinned to the version used for the paper. Install it editable in step 3 and
do not update or replace it.

## Requirements

| | |
|---|---|
| Suggested machine | Windows |
| Package manager | conda (Miniconda or Anaconda) |
| Disk | ~20 GB |
| RAM | 16 GB minimum, 32 GB recommended |

## Setup

1. Clone:
   ```bash
   git clone https://github.com/duwadisudan/wholehead-cocktail-party-fnirs.git
   cd wholehead-cocktail-party-fnirs
   ```

2. Create the conda environment from the Windows lockfile:
   ```bash
   conda env create -f environment-win.lock.yml -n cedalion
   conda activate cedalion
   ```
   `environment-win.lock.yml` pins exact builds and reproduces the binary
   stack used to produce the paper figures. `environment.yml` is a
   higher-level spec kept for reference; do not use it directly.

3. Install the vendored cedalion:
   ```bash
   pip install -e ./cedalion
   ```

4. (Optional) Install the vendored whichscript:
   ```bash
   pip install -e ./whichscript
   ```
   [whichscript](whichscript/) writes a hidden script-snapshot sidecar next
   to every output file, for provenance. Several figure scripts call it. If
   you do not want it, comment out the `from whichscript import ...`,
   `configure(...)`, and `enable_auto_logging()` lines in those scripts.

5. Install this project:
   ```bash
   pip install -e .
   ```

6. Verify:
   ```bash
   python -c "import cedalion, wholehead_cocktail_party; print('ok')"
   ```

7. Configure paths and run options. Open [`config/paths.yml`](config/paths.yml)
   and replace each `EDIT_ME` value with the absolute path on your machine.
   Open [`config/run.yml`](config/run.yml) to choose:
     - `condition: overt` or `covert`
     - `mode: full` (raw -> preprocess -> classify -> figures) or
       `mode: from-derivatives` (skip preprocessing if you have shipped pickles)
     - `subjects: all` (paper cohort), `test` (sub-10 only, for a quick
       end-to-end sanity check), or an explicit list like `['10', '20']`

   Sanity-check:
   ```bash
   python -c "from wholehead_cocktail_party.paths import load_paths; print(load_paths())"
   python -c "from wholehead_cocktail_party.run_config import load_run_config; print(load_run_config())"
   ```

## Run

Edit `config/run.yml` to choose your condition / mode / subject list, then
invoke whichever script you want. Examples:

```bash
# Quick smoke check on a single subject (sub-10), full pipeline from raw:
# (set subjects: test, mode: full in config/run.yml first)
python scripts/classification/classify_rf_all_channels.py

# Full paper cohort:
# (set subjects: all in config/run.yml first)
python scripts/classification/classify_rf_all_channels.py
```

No command-line flags. Every reviewer-tunable knob lives in `config/run.yml`,
so you do not need to edit any `.py` source. Figure scripts under
[`scripts/`](scripts/) read the classifier outputs and respect the same
`config/run.yml`.

## Repository layout

```
.
├── README.md
├── pyproject.toml
├── environment.yml             # high-level spec; not used directly
├── environment-win.lock.yml    # exact-build Windows lockfile (use this)
├── env_snapshot/               # raw env snapshots used to build environment.yml
├── cedalion/                   # vendored, frozen. Do not update.
├── whichscript/                # vendored provenance helper (optional).
├── config/
│   ├── paths.yml               # edit once with your local data paths
│   └── run.yml                 # condition / mode / subjects knobs
├── src/wholehead_cocktail_party/
└── scripts/
    ├── classification/
    ├── preprocessing/
    ├── figure2_hemodynamic_response/
    ├── figure3_accuracy_over_time/
    ├── figure4_accuracy_scatter_latency/
    ├── figure5_pc_contributions/
    ├── figure6_overt_vs_covert_scatter/
    ├── figure7_accuracy_over_time_ang_gyr/
    ├── misc/
    └── supplementary/
```

## License

MIT.

## Citation

> Decoding Spatial Attention in the Cocktail Party Problem Using Wearable
> Whole-head High-Density fNIRS.

## Issues

[Open an issue](https://github.com/duwadisudan/wholehead-cocktail-party-fnirs/issues)
or contact Sudan Duwadi (sudan@bu.edu).
