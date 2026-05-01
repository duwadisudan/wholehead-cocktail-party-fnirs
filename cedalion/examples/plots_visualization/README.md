# Plots & Visualization Examples

- `mock_hr_gui.py` — Simple PyQt + Matplotlib GUI that simulates and plots mock heart-rate (HR) data in real time.

## Run (Windows PowerShell)

Using your existing conda env (as defined in `environment_dev.yml`):

```powershell
conda activate cedalion
python examples\plots_visualization\mock_hr_gui.py
```

Notes:
- Requires Qt-backed Matplotlib; the repo env includes `PyQt` and `matplotlib`.
- If `PyQt5` isn't available, the script falls back to `PySide6` when installed.
- If you see a backend error, ensure the conda env is active before launching.
