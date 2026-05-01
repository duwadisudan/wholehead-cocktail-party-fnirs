"""
Mock HR (Heart Rate) real-time visualization GUI.

This example opens a small PyQt application that simulates heart rate data
and plots it live using Matplotlib. Use it as a starting point for quick
interactive demos.

Controls:
- Start / Stop: begin or pause the simulation
- Reset: clear the buffer and restart the timer
- Baseline (bpm): mean heart rate
- Variability (bpm): periodic modulation amplitude
- Noise (bpm std): Gaussian noise standard deviation per sample
- Window (s): seconds to display in the rolling window
- Update (ms): timer period; smaller = smoother updates
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import deque

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

try:  # Prefer PyQt5 if available (conda package is typically named PyQt)
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import Qt
except Exception:  # Fallback to PySide6 if PyQt5 isn't installed
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import Qt


class HRSimConfig:
    """Configuration for HR simulation."""

    def __init__(
        self,
        baseline_bpm: float = 70.0,
        variability_bpm: float = 8.0,
        noise_std_bpm: float = 1.2,
        window_seconds: int = 30,
        update_ms: int = 100,
    ) -> None:
        self.baseline_bpm = baseline_bpm
        self.variability_bpm = variability_bpm
        self.noise_std_bpm = noise_std_bpm
        self.window_seconds = window_seconds
        self.update_ms = update_ms


class HRGui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mock HR Visualizer")
        self.resize(900, 600)

        self.cfg = HRSimConfig()

        # Timing / data buffers
        self._t0 = time.perf_counter()
        self._last_update = self._t0
        # Deques bounded by samples in window (computed dynamically)
        self.times: deque[float] = deque()
        self.hr_values: deque[float] = deque()

        self._build_ui()
        self._connect_signals()
        self._setup_timer()
        self._recompute_maxlen()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Matplotlib figure + toolbar
        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("HR (bpm)")
        self.ax.grid(True, alpha=0.3)
        self._line, = self.ax.plot([], [], color="#1f77b4", lw=1.8, label="HR")
        self._avg_line, = self.ax.plot([], [], color="#ff7f0e", lw=1.0, alpha=0.8, label="MA(5s)")
        self.ax.legend(loc="upper right")

        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas, stretch=1)

        # Controls
        form = QtWidgets.QGridLayout()

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.reset_btn = QtWidgets.QPushButton("Reset")

        self.baseline_spin = QtWidgets.QDoubleSpinBox()
        self.baseline_spin.setRange(30.0, 220.0)
        self.baseline_spin.setValue(self.cfg.baseline_bpm)
        self.baseline_spin.setSuffix(" bpm")
        self.baseline_spin.setSingleStep(1.0)

        self.variability_spin = QtWidgets.QDoubleSpinBox()
        self.variability_spin.setRange(0.0, 60.0)
        self.variability_spin.setValue(self.cfg.variability_bpm)
        self.variability_spin.setSuffix(" bpm")
        self.variability_spin.setSingleStep(0.5)

        self.noise_spin = QtWidgets.QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 20.0)
        self.noise_spin.setValue(self.cfg.noise_std_bpm)
        self.noise_spin.setSuffix(" bpm")
        self.noise_spin.setSingleStep(0.2)

        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(5, 600)
        self.window_spin.setValue(self.cfg.window_seconds)
        self.window_spin.setSuffix(" s")

        self.update_spin = QtWidgets.QSpinBox()
        self.update_spin.setRange(10, 2000)
        self.update_spin.setValue(self.cfg.update_ms)
        self.update_spin.setSuffix(" ms")

        self.curr_hr_label = QtWidgets.QLabel("Current HR: –")
        f = self.curr_hr_label.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.curr_hr_label.setFont(f)

        # Layout grid
        r = 0
        form.addWidget(self.start_btn, r, 0)
        form.addWidget(self.stop_btn, r, 1)
        form.addWidget(self.reset_btn, r, 2)
        r += 1
        form.addWidget(QtWidgets.QLabel("Baseline"), r, 0)
        form.addWidget(self.baseline_spin, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel("Variability"), r, 0)
        form.addWidget(self.variability_spin, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel("Noise"), r, 0)
        form.addWidget(self.noise_spin, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel("Window"), r, 0)
        form.addWidget(self.window_spin, r, 1)
        r += 1
        form.addWidget(QtWidgets.QLabel("Update"), r, 0)
        form.addWidget(self.update_spin, r, 1)
        r += 1
        form.addWidget(self.curr_hr_label, r, 0, 1, 3)
        layout.addLayout(form)

    def _connect_signals(self) -> None:
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.reset_btn.clicked.connect(self._on_reset)

        self.baseline_spin.valueChanged.connect(self._apply_cfg)
        self.variability_spin.valueChanged.connect(self._apply_cfg)
        self.noise_spin.valueChanged.connect(self._apply_cfg)
        self.window_spin.valueChanged.connect(self._on_window_changed)
        self.update_spin.valueChanged.connect(self._on_update_interval_changed)

    def _setup_timer(self) -> None:
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(self.cfg.update_ms)
        self.timer.timeout.connect(self._on_tick)

    # ---------- Config / events ----------
    def _apply_cfg(self) -> None:
        self.cfg.baseline_bpm = float(self.baseline_spin.value())
        self.cfg.variability_bpm = float(self.variability_spin.value())
        self.cfg.noise_std_bpm = float(self.noise_spin.value())

    def _on_window_changed(self, val: int) -> None:
        self.cfg.window_seconds = int(val)
        self._recompute_maxlen()
        self._trim_to_window()
        self._redraw()

    def _on_update_interval_changed(self, val: int) -> None:
        self.cfg.update_ms = int(val)
        self.timer.setInterval(self.cfg.update_ms)
        self._recompute_maxlen()

    def _on_start(self) -> None:
        if not self.timer.isActive():
            self._t0 = time.perf_counter()
            self._last_update = self._t0
            self.timer.start()

    def _on_stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()

    def _on_reset(self) -> None:
        self._on_stop()
        self.times.clear()
        self.hr_values.clear()
        self._t0 = time.perf_counter()
        self._last_update = self._t0
        self._redraw()

    # ---------- Simulation / plotting ----------
    def _recompute_maxlen(self) -> None:
        # Estimate samples per second from update interval
        sps = max(1, int(round(1000.0 / max(1, self.cfg.update_ms))))
        maxlen = max(sps * max(1, int(self.cfg.window_seconds)), 2)
        # Deques don't support changing maxlen directly; rebuild if needed
        def rebuild(dq: deque[float]) -> deque[float]:
            newdq: deque[float] = deque(dq, maxlen=maxlen)
            return newdq

        cur_max = self.times.maxlen if self.times.maxlen is not None else 0
        if cur_max != maxlen:
            self.times = rebuild(self.times)
            self.hr_values = rebuild(self.hr_values)

    def _on_tick(self) -> None:
        now = time.perf_counter()
        t = now - self._t0

        hr = self._simulate_hr(t)
        self.times.append(t)
        self.hr_values.append(hr)
        self._last_update = now

        self._update_status(hr)
        self._redraw()

    def _simulate_hr(self, t: float) -> float:
        # Breathing-like low-frequency modulation (~0.25 Hz)
        breathe_f = 0.25
        circ = 2.0 * math.pi
        mod1 = math.sin(circ * breathe_f * t)
        # Very low-frequency drift (~0.03 Hz)
        mod2 = 0.5 * math.sin(circ * 0.03 * t + 1.2)
        hr = self.cfg.baseline_bpm + self.cfg.variability_bpm * (mod1 + mod2)
        # Add Gaussian noise
        hr += random.gauss(0.0, self.cfg.noise_std_bpm)
        # Bound plausible HR range
        return float(max(30.0, min(220.0, hr)))

    def _moving_average(self, xs: list[float], ys: list[float], win_s: float = 5.0) -> tuple[list[float], list[float]]:
        if not xs:
            return xs, ys
        # Use simple time-based MA with sliding window of win_s seconds
        out_x: list[float] = []
        out_y: list[float] = []
        j = 0
        n = len(xs)
        for i in range(n):
            t_i = xs[i]
            while j < n and xs[j] < t_i - win_s:
                j += 1
            if j <= i:
                seg = ys[j : i + 1]
                out_x.append(t_i)
                out_y.append(sum(seg) / max(1, len(seg)))
        return out_x, out_y

    def _redraw(self) -> None:
        xs = list(self.times)
        ys = list(self.hr_values)
        if not xs:
            self._line.set_data([], [])
            self._avg_line.set_data([], [])
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()
            return

        t0 = xs[0]
        xs0 = [x - t0 for x in xs]  # normalize time axis to window start

        self._line.set_data(xs0, ys)
        # Moving average line over 5 seconds
        avg_xs, avg_ys = self._moving_average(xs0, ys, win_s=5.0)
        self._avg_line.set_data(avg_xs, avg_ys)

        # Keep a nice y-margin around current data
        y_min = min(ys)
        y_max = max(ys)
        y_pad = max(2.0, 0.05 * max(1.0, y_max - y_min))
        self.ax.set_xlim(left=0.0, right=max(5.0, xs0[-1]))
        self.ax.set_ylim(bottom=y_min - y_pad, top=y_max + y_pad)
        self.canvas.draw_idle()

    def _trim_to_window(self) -> None:
        # Deques already bound by maxlen, but also trim by absolute time window
        if not self.times:
            return
        t_end = self.times[-1]
        t_start = t_end - float(self.cfg.window_seconds)
        while self.times and self.times[0] < t_start:
            self.times.popleft()
            self.hr_values.popleft()

    def _update_status(self, hr: float) -> None:
        self.curr_hr_label.setText(f"Current HR: {hr:.1f} bpm")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = HRGui()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
