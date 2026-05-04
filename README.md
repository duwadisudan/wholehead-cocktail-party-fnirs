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

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate cedalion
   ```

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

## Run

The classification script runs end to end on the raw BIDS data:

```bash
python scripts/classification/classify_rf_all_channels.py
```

Figure scripts under [`scripts/`](scripts/) use its outputs.

## Repository layout

```
.
├── README.md
├── pyproject.toml
├── environment.yml
├── env_snapshot/        # raw env snapshots used to build environment.yml
├── cedalion/            # vendored, frozen. Do not update.
├── whichscript/         # vendored provenance helper (optional).
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
