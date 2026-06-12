"""
app_qt.py

Native desktop GUI for the microgpt visualizer using Qt (via QtPy).
Three tabs: Architecture, Training (live), Sampling.

Run:
    python app_qt.py

Requires:
    pip install qtpy PySide6 matplotlib numpy

    PyQt5, PyQt6, and PySide2 also work in place of PySide6 — QtPy auto-detects
    whichever Qt binding is installed.
"""

import sys
import math
import numpy as np

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QSlider,
    QLineEdit, QGroupBox, QSizePolicy, QTextEdit,
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from microgpt_core import (
    Trainer, load_dataset, build_tokenizer, apply_sampling,
    N_LAYER, N_EMBD, BLOCK_SIZE, N_HEAD, HEAD_DIM,
)


# =====================================================================
# Reusable matplotlib canvas (works with both PyQt and PySide)
# =====================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(5, 4)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# =====================================================================
# Constants
# =====================================================================
MATRIX_NAMES = [
    'wte', 'wpe',
    'layer0.attn_wq', 'layer0.attn_wk',
    'layer0.attn_wv', 'layer0.attn_wo',
    'layer0.mlp_fc1', 'layer0.mlp_fc2',
    'lm_head',
]

MATRIX_DESCRIPTIONS = {
    'wte': "Token embedding. Each row is the learned vector for one token. "
           "Watch this train — characters that play similar roles end up with similar rows.",
    'wpe': "Position embedding. Row i is added to the token at position i. "
           "Often develops smooth low-frequency structure across positions.",
    'layer0.attn_wq': "Attention: query projection. Determines what each token is 'looking for'.",
    'layer0.attn_wk': "Attention: key projection. Determines what each token 'advertises'.",
    'layer0.attn_wv': "Attention: value projection. The information actually mixed across tokens.",
    'layer0.attn_wo': "Attention: output projection. Mixes per-head outputs back into the residual stream.",
    'layer0.mlp_fc1': "MLP fc1: expand n_embd → 4·n_embd.",
    'layer0.mlp_fc2': "MLP fc2: contract 4·n_embd → n_embd.",
    'lm_head': "Final projection from residual stream to vocabulary logits.",
}


def row_labels_for(name, trainer):
    if name in ('wte', 'lm_head'):
        return list(trainer.uchars) + ['·BOS·']
    if name == 'wpe':
        return [f'pos {i}' for i in range(BLOCK_SIZE)]
    return None


def matrix_to_array(state_dict, name):
    return np.array([[v.data for v in row] for row in state_dict[name]])


# =====================================================================
# Tab 1 — Architecture
# =====================================================================
class ArchitectureTab(QWidget):
    def __init__(self, trainer, parent=None):
        super().__init__(parent)
        self.trainer = trainer

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.canvas = MplCanvas(self, figsize=(7, 10))
        self.canvas.setMinimumWidth(450)
        layout.addWidget(self.canvas, 3)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setHtml(self._explanation_html())
        text.setMinimumWidth(320)
        layout.addWidget(text, 2)

        self._draw_diagram()

    def _explanation_html(self):
        return """
        <h2>Forward pass</h2>
        <p>A single token (here, one character) flows through:</p>
        <p><b>Embedding.</b> The token id and its position are looked up as 16-dim
        vectors and added. The vocabulary is the unique characters in the dataset,
        plus a special <code>BOS</code> token.</p>
        <p><b>Attention.</b> Each token attends to itself and earlier tokens via
        4 parallel heads, each operating on a 4-dim slice. Within a head: project
        to <code>Q, K, V</code>, compute scaled dot-product attention, mix the values.</p>
        <p><b>MLP.</b> A 2-layer feedforward with a ReLU. Expands to 4× the residual
        width, then contracts back.</p>
        <p><b>LM head.</b> Project from residual stream to vocabulary; softmax.</p>
        <p><b>Residuals (dashed).</b> Each block's output is <i>added</i> to its input.
        This is what lets gradients flow through deep stacks.</p>
        <p><b>RMSNorm.</b> Normalizes a vector by its root-mean-square.</p>
        <hr>
        <h3>How training works</h3>
        <ol>
        <li>Forward pass at every prefix of the document → next-char probability.</li>
        <li>Average negative-log-probability of the actual next char = the loss.</li>
        <li>Backpropagate through every operation (the autograd in <code>Value</code>).</li>
        <li>Update each weight matrix via Adam.</li>
        </ol>
        <p>Watch this happen on the next tab.</p>
        """

    def _draw_diagram(self):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 16)
        ax.set_aspect('equal')
        ax.axis('off')

        vocab = self.trainer.vocab_size

        def box(x, y, w, h, txt, color='#cce5ff', fontsize=9):
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                linewidth=1.2, edgecolor='#222', facecolor=color,
            )
            ax.add_patch(patch)
            ax.text(x, y, txt, ha='center', va='center', fontsize=fontsize)

        def arrow(x1, y1, x2, y2, dashed=False, color='#222'):
            style = (0, (4, 3)) if dashed else '-'
            ap = FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle='->',
                mutation_scale=14, linewidth=1.2,
                color=color, linestyle=style,
            )
            ax.add_patch(ap)

        IN, EMB, NORM, BLK, ADD, OUT = (
            '#fff5cc', '#cce5ff', '#d4edda', '#ffe0b2', '#e0e0e0', '#fff5cc'
        )

        box(5, 15.0, 3.5, 0.7, f"input token id  (0…{vocab - 1})", IN)
        box(2.6, 13.4, 3.0, 0.95, f"wte\n[{vocab}×{N_EMBD}]\ntoken embedding", EMB)
        box(7.4, 13.4, 3.0, 0.95, f"wpe\n[{BLOCK_SIZE}×{N_EMBD}]\npos embedding", EMB)
        box(5, 12.0, 0.6, 0.6, "+", ADD, fontsize=12)
        box(5, 11.0, 1.8, 0.55, "RMSNorm", NORM)
        box(5, 9.4, 4.6, 1.6,
            f"Multi-Head Attention\n"
            f"Wq · Wk · Wv : [{N_EMBD}×{N_EMBD}] each\n"
            f"Wo : [{N_EMBD}×{N_EMBD}]\n"
            f"{N_HEAD} heads × {HEAD_DIM} dims", BLK)
        box(5, 7.9, 0.6, 0.6, "+", ADD, fontsize=12)
        box(5, 6.9, 1.8, 0.55, "RMSNorm", NORM)
        box(5, 5.4, 4.6, 1.4,
            f"MLP\nfc1 : [{4 * N_EMBD}×{N_EMBD}]\nReLU\n"
            f"fc2 : [{N_EMBD}×{4 * N_EMBD}]", BLK)
        box(5, 4.0, 0.6, 0.6, "+", ADD, fontsize=12)
        box(5, 2.9, 3.0, 0.8, f"lm_head  [{vocab}×{N_EMBD}]", EMB)
        box(5, 1.7, 1.6, 0.55, "softmax", NORM)
        box(5, 0.5, 4.0, 0.7, f"next-token probabilities  ({vocab} tokens)", OUT)

        # Spine arrows (top-to-bottom)
        arrow(5, 14.65, 4.0, 13.88)
        arrow(5, 14.65, 6.0, 13.88)
        arrow(2.6, 12.93, 4.7, 12.20)
        arrow(7.4, 12.93, 5.3, 12.20)
        arrow(5, 11.70, 5, 11.28)
        arrow(5, 10.72, 5, 10.20)
        arrow(5, 8.60, 5, 8.20)
        arrow(5, 7.60, 5, 7.18)
        arrow(5, 6.62, 5, 6.10)
        arrow(5, 4.70, 5, 4.30)
        arrow(5, 3.70, 5, 3.30)
        arrow(5, 2.50, 5, 1.97)
        arrow(5, 1.42, 5, 0.85)

        # Residual loops on the right (dashed)
        for top_y, bot_y in [(11.0, 7.9), (6.9, 4.0)]:
            arrow(7.6 if top_y == 11.0 else 6.9, top_y, 8.5, top_y, color='#888')
            ax.add_patch(FancyArrowPatch(
                (8.5, top_y), (8.5, bot_y),
                arrowstyle='-', linestyle=(0, (4, 3)),
                linewidth=1.2, color='#888',
            ))
            arrow(8.5, bot_y, 5.3, bot_y, dashed=True, color='#888')
            ax.text(8.7, (top_y + bot_y) / 2, 'residual',
                    fontsize=8, color='#666', rotation=90, va='center')

        self.canvas.draw_idle()


# =====================================================================
# Tab 2 — Training (live)
# =====================================================================
class TrainingTab(QWidget):
    def __init__(self, trainer, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.show_grad = False
        self.selected_matrix = 'wte'

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Top button row ----------------------------------------------
        btn_row = QHBoxLayout()
        self.btn_step1 = QPushButton("Train 1 step")
        self.btn_step10 = QPushButton("Train 10 steps")
        self.btn_auto = QPushButton("▶ Auto-train")
        self.btn_auto.setCheckable(True)
        self.btn_reset = QPushButton("Reset model")
        self.label_step = QLabel(self._step_label())
        self.label_step.setStyleSheet("font-weight: bold; padding: 4px 14px;")

        for b in (self.btn_step1, self.btn_step10, self.btn_auto, self.btn_reset):
            btn_row.addWidget(b)
        btn_row.addWidget(self.label_step, 1)
        layout.addLayout(btn_row)

        self.btn_step1.clicked.connect(self._step1)
        self.btn_step10.clicked.connect(self._step10)
        self.btn_auto.toggled.connect(self._auto_toggled)
        self.btn_reset.clicked.connect(self._reset)

        # --- Selector + gradient toggle ----------------------------------
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Inspect:"))
        self.combo_matrix = QComboBox()
        self.combo_matrix.addItems(MATRIX_NAMES)
        self.combo_matrix.setCurrentText(self.selected_matrix)
        self.combo_matrix.currentTextChanged.connect(self._matrix_changed)
        sel_row.addWidget(self.combo_matrix, 1)

        self.chk_grad = QCheckBox("Show gradients (∂loss/∂W) instead of weights")
        self.chk_grad.toggled.connect(self._grad_toggled)
        sel_row.addWidget(self.chk_grad)
        layout.addLayout(sel_row)

        self.label_desc = QLabel(MATRIX_DESCRIPTIONS[self.selected_matrix])
        self.label_desc.setWordWrap(True)
        self.label_desc.setStyleSheet(
            "color: #555; font-style: italic; padding: 2px 4px;"
        )
        layout.addWidget(self.label_desc)

        # --- Heatmap + loss canvas side-by-side --------------------------
        canvas_row = QHBoxLayout()
        self.canvas_heatmap = MplCanvas(self, figsize=(6, 4.5))
        self.canvas_loss = MplCanvas(self, figsize=(4, 4.5))
        canvas_row.addWidget(self.canvas_heatmap, 3)
        canvas_row.addWidget(self.canvas_loss, 2)
        layout.addLayout(canvas_row, 1)

        # --- Live training timer -----------------------------------------
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._train_chunk)

        self.refresh()

    # --- Helpers / handlers ---------------------------------------------
    def _step_label(self):
        last = self.trainer.losses[-1] if self.trainer.losses else None
        loss_part = f"loss {last:.3f}" if last is not None else "loss —"
        return f"step {self.trainer.step} / {self.trainer.total_steps}  ·  {loss_part}"

    def _matrix_changed(self, name):
        self.selected_matrix = name
        self.label_desc.setText(MATRIX_DESCRIPTIONS[name])
        self.refresh()

    def _grad_toggled(self, checked):
        self.show_grad = checked
        self.refresh()

    def _step1(self):
        self.trainer.train_step()
        self.refresh()

    def _step10(self):
        for _ in range(10):
            self.trainer.train_step()
        self.refresh()

    def _auto_toggled(self, on):
        if on:
            self.btn_auto.setText("⏸ Pause")
            self.timer.start()
        else:
            self.btn_auto.setText("▶ Auto-train")
            self.timer.stop()

    def _reset(self):
        self.timer.stop()
        self.btn_auto.blockSignals(True)
        self.btn_auto.setChecked(False)
        self.btn_auto.setText("▶ Auto-train")
        self.btn_auto.blockSignals(False)
        # Build a fresh trainer; mutate the existing instance in place so other
        # tabs that hold a reference still see the reset.
        new = Trainer(
            self.trainer.docs,
            (self.trainer.uchars, self.trainer.BOS, self.trainer.vocab_size),
            total_steps=self.trainer.total_steps,
            learning_rate=self.trainer.learning_rate,
        )
        self.trainer.__dict__.update(new.__dict__)
        self.refresh()

    def _train_chunk(self):
        CHUNK = 3
        for _ in range(CHUNK):
            if self.trainer.step >= self.trainer.total_steps:
                self.btn_auto.setChecked(False)
                return
            self.trainer.train_step()
        self.refresh()

    def refresh(self):
        self.label_step.setText(self._step_label())
        self._refresh_heatmap()
        self._refresh_loss()

    def _refresh_heatmap(self):
        self.canvas_heatmap.fig.clear()
        ax = self.canvas_heatmap.fig.add_subplot(111)
        name = self.selected_matrix

        if self.show_grad:
            if self.trainer.last_grad_snapshot is None:
                ax.text(0.5, 0.5, "Take a training step to see gradients.",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=11, color='#666')
                ax.axis('off')
                self.canvas_heatmap.draw_idle()
                return
            arr = np.array(self.trainer.last_grad_snapshot[name])
            vmax = max(abs(arr.min()), abs(arr.max()), 1e-8)
            vmin = -vmax
            title = f"{name} — gradients   (step {self.trainer.step})"
        else:
            arr = matrix_to_array(self.trainer.state_dict, name)
            vmin, vmax = -1.5, 1.5
            title = f"{name} — weights   (step {self.trainer.step})"

        im = ax.imshow(arr, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(title, fontsize=11)
        labels = row_labels_for(name, self.trainer)
        if labels and len(labels) == arr.shape[0]:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
        else:
            ax.set_ylabel("row")
        ax.set_xlabel(
            "column (embedding dim)"
            if name in ('wte', 'wpe', 'lm_head') else "column"
        )
        self.canvas_heatmap.fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        self.canvas_heatmap.draw_idle()

    def _refresh_loss(self):
        self.canvas_loss.fig.clear()
        ax = self.canvas_loss.fig.add_subplot(111)
        if not self.trainer.losses:
            ax.text(0.5, 0.5, "Train a step to begin.",
                    ha='center', va='center', transform=ax.transAxes, color='#666')
            ax.axis('off')
        else:
            ax.plot(self.trainer.losses, color='#2E86AB', linewidth=1.2)
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.set_title(
                f"Training loss   (current: {self.trainer.losses[-1]:.3f})",
                fontsize=11,
            )
            ax.grid(True, alpha=0.3)
        self.canvas_loss.draw_idle()


# =====================================================================
# Tab 3 — Sampling
# =====================================================================
class SamplingTab(QWidget):
    def __init__(self, trainer, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.raw_cache = None
        self.cache_key = None  # (prefix, step) — invalidated when either changes

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Prefix input -------------------------------------------------
        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Prefix:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setMaxLength(BLOCK_SIZE - 1)
        self.prefix_edit.setPlaceholderText(
            "e.g.   emm   (leave empty for the BOS distribution)"
        )
        self.prefix_edit.textChanged.connect(self.refresh)
        prefix_row.addWidget(self.prefix_edit, 1)
        layout.addLayout(prefix_row)

        # --- Sliders -----------------------------------------------------
        sg_box = QGroupBox(
            "Sampling controls (applied in order: temperature → top-k → top-p)"
        )
        sg = QGridLayout(sg_box)

        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(10, 200)
        self.slider_temp.setValue(100)
        self.label_temp_val = QLabel("1.00")
        self.slider_temp.valueChanged.connect(self._temp_changed)
        sg.addWidget(QLabel("Temperature"), 0, 0)
        sg.addWidget(self.slider_temp, 0, 1)
        sg.addWidget(self.label_temp_val, 0, 2)

        self.slider_topk = QSlider(Qt.Horizontal)
        self.slider_topk.setRange(1, self.trainer.vocab_size)
        self.slider_topk.setValue(self.trainer.vocab_size)
        self.label_topk_val = QLabel(str(self.trainer.vocab_size))
        self.slider_topk.valueChanged.connect(self._topk_changed)
        sg.addWidget(QLabel("Top-k"), 1, 0)
        sg.addWidget(self.slider_topk, 1, 1)
        sg.addWidget(self.label_topk_val, 1, 2)

        self.slider_topp = QSlider(Qt.Horizontal)
        self.slider_topp.setRange(5, 100)
        self.slider_topp.setValue(100)
        self.label_topp_val = QLabel("1.00")
        self.slider_topp.valueChanged.connect(self._topp_changed)
        sg.addWidget(QLabel("Top-p"), 2, 0)
        sg.addWidget(self.slider_topp, 2, 1)
        sg.addWidget(self.label_topp_val, 2, 2)

        sg.setColumnStretch(1, 1)
        layout.addWidget(sg_box)

        # --- Bar chart ----------------------------------------------------
        self.canvas = MplCanvas(self, figsize=(11, 4))
        layout.addWidget(self.canvas, 1)

        # --- Diagnostics row ---------------------------------------------
        diag_row = QHBoxLayout()
        self.label_surv = QLabel("Surviving: —")
        self.label_raw_h = QLabel("Raw entropy: —")
        self.label_adj_h = QLabel("Adjusted entropy: —")
        for lab in (self.label_surv, self.label_raw_h, self.label_adj_h):
            lab.setStyleSheet(
                "padding: 4px 14px; background:#f4f4f4; border-radius:4px;"
            )
            diag_row.addWidget(lab)
        diag_row.addStretch(1)
        layout.addLayout(diag_row)

        # --- Generate samples row ----------------------------------------
        sample_row = QHBoxLayout()
        self.btn_sample = QPushButton("Generate 5 samples")
        self.btn_sample.clicked.connect(self._gen_samples)
        sample_row.addWidget(self.btn_sample)
        self.label_samples = QLabel("Click to generate names from the current model.")
        self.label_samples.setStyleSheet(
            "font-family: monospace; padding: 6px 10px; background:#fafafa;"
        )
        self.label_samples.setMinimumHeight(40)
        sample_row.addWidget(self.label_samples, 1)
        layout.addLayout(sample_row)

        self.refresh()

    # --- Slider callbacks (update label + redraw) ------------------------
    def _temp_changed(self, v):
        self.label_temp_val.setText(f"{v / 100:.2f}")
        self.refresh()

    def _topk_changed(self, v):
        self.label_topk_val.setText(str(v))
        self.refresh()

    def _topp_changed(self, v):
        self.label_topp_val.setText(f"{v / 100:.2f}")
        self.refresh()

    def _current_sampling(self):
        T = self.slider_temp.value() / 100
        k = self.slider_topk.value()
        p = self.slider_topp.value() / 100
        return (
            T,
            (None if k >= self.trainer.vocab_size else k),
            (None if p >= 1.0 else p),
        )

    def _gen_samples(self):
        T, k, p = self._current_sampling()
        out = [self.trainer.sample_one(T, k, p) for _ in range(5)]
        self.label_samples.setText(
            "    ".join(s if s else "<empty>" for s in out)
        )

    def refresh(self):
        prefix = self.prefix_edit.text()
        key = (prefix, self.trainer.step)
        if key != self.cache_key:
            self.raw_cache = self.trainer.get_next_token_probs(prefix)
            self.cache_key = key

        T, k, p = self._current_sampling()
        adj = apply_sampling(self.raw_cache, temperature=T, top_k=k, top_p=p)

        self._refresh_chart(self.raw_cache, adj, prefix)
        self._refresh_diagnostics(self.raw_cache, adj)

    def _refresh_chart(self, raw, adj, prefix):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        labels_all = list(self.trainer.uchars) + ['·BOS·']
        order = sorted(range(len(labels_all)), key=lambda i: -raw[i])
        labels_o = [labels_all[i] for i in order]
        raw_o = [raw[i] for i in order]
        adj_o = [adj[i] for i in order]
        x = np.arange(len(labels_o))
        width = 0.4
        ax.bar(x - width / 2, raw_o, width,
               label='raw', color='#9aa0a6', alpha=0.85)
        ax.bar(x + width / 2, adj_o, width,
               label='after temp / top-k / top-p', color='#2E86AB')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_o, fontsize=10)
        ax.set_ylabel("probability")
        title_pref = repr(prefix) if prefix else "<empty>  (just BOS)"
        ax.set_title(
            f"Next-token distribution  ·  prefix = {title_pref}  ·  "
            f"model step {self.trainer.step}",
            fontsize=11,
        )
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        self.canvas.draw_idle()

    def _refresh_diagnostics(self, raw, adj):
        eps = 1e-12
        surv = sum(1 for q in adj if q > 1e-6)
        raw_h = -sum(q * math.log(q + eps) for q in raw)
        adj_h = -sum(q * math.log(q + eps) for q in adj if q > 0)
        self.label_surv.setText(
            f"Surviving tokens: {surv} / {self.trainer.vocab_size}"
        )
        self.label_raw_h.setText(f"Raw entropy: {raw_h:.2f} nats")
        self.label_adj_h.setText(
            f"Adjusted entropy: {adj_h:.2f} nats   (Δ {adj_h - raw_h:+.2f})"
        )


# =====================================================================
# Main window
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("microgpt — a small transformer, visualized")
        self.resize(1280, 820)

        docs = load_dataset()
        tok = build_tokenizer(docs)
        self.trainer = Trainer(docs, tok, total_steps=1000)

        header = QLabel(
            f"{N_LAYER} layer  ·  n_embd={N_EMBD}  ·  {N_HEAD} heads  ·  "
            f"vocab={self.trainer.vocab_size}  ·  block_size={BLOCK_SIZE}  ·  "
            f"params={len(self.trainer.params)}"
        )
        header.setStyleSheet("padding: 4px 10px; color: #444;")

        self.tabs = QTabWidget()
        self.arch_tab = ArchitectureTab(self.trainer)
        self.train_tab = TrainingTab(self.trainer)
        self.sampling_tab = SamplingTab(self.trainer)
        self.tabs.addTab(self.arch_tab, "1 · Architecture")
        self.tabs.addTab(self.train_tab, "2 · Training (live)")
        self.tabs.addTab(self.sampling_tab, "3 · Sampling")
        self.tabs.currentChanged.connect(self._tab_changed)

        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(header)
        v.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

    def _tab_changed(self, idx):
        # Sampling tab caches against trainer.step; refresh it when shown so the
        # bar chart reflects training that happened on the previous tab.
        if self.tabs.widget(idx) is self.sampling_tab:
            self.sampling_tab.refresh()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
