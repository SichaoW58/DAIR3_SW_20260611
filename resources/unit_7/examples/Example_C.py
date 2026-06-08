"""
Example_C.py
Interactive visualization of weight convergence in supervised learning with 
symbolic and numeric steps.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Parameters
x = 1.5
y_desired = 0.5
alpha = 0.1
w_init = 0.8

# Update rule: w_new = w_old + alpha * 2 * (y_desired - w_old * x) * x
w_values = [w_init]
equations = []

w = w_init
for i in range(10):
    y_actual = w * x
    error = y_desired - y_actual
    update = alpha * 2 * error * x
    w_new = w + update

    symbolic = rf"$w_{{{i+1}}} = w_{{{i}}} + \alpha \cdot 2(y_{{desired}} - w_{{{i}}} \cdot x) \cdot x$"
    numeric = rf"$w_{{{i+1}}} = {w:.4f} + {alpha} \cdot 2({y_desired} - {w:.4f} \cdot {x}) \cdot {x} = {w_new:.6f}$"
    equations.append((symbolic, numeric))
    w_values.append(w_new)
    w = w_new

# Setup figure with two subplots
fig, (ax_text, ax_plot) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
fig.canvas.manager.set_window_title("Example_C")
plt.subplots_adjust(bottom=0.3, hspace=0.5)
fig.suptitle(r'$\alpha = 0.1$, $x_0 = 1.5$, $y_{desired} = 0.5$, $w=?$', y=0.98)

# Text box (multi-line)
text_box = ax_text.text(0.01, 0.95, '', transform=ax_text.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment='left')
ax_text.set_axis_off()

# Convergence plot with a line that grows
line_full, = ax_plot.plot([], [], 'bo-', label='w')
cursor_dot, = ax_plot.plot([0], [w_values[0]], 'ro')
ax_plot.set_title("Weight Convergence")
ax_plot.set_xlabel("Step")
ax_plot.set_ylabel("w")
ax_plot.set_xlim(-0.5, len(w_values) - 0.5)
ax_plot.set_xticks(range(len(w_values)))
ax_plot.set_ylim(min(w_values) - 0.05, max(w_values) + 0.05)
ax_plot.grid(True)
ax_plot.legend()

# Slider
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.05])
step_slider = Slider(slider_ax, 'Step', -1, len(equations) - 1, valinit=-1, valstep=1)
step_slider.on_changed(update)

def update(step):
    idx = int(step)
    if idx == -1:
        text_box.set_text('')
        cursor_dot.set_data([], [])
        line_full.set_data([], [])
    else:
        interleaved_lines = []
        for i in range(idx + 1):
            interleaved_lines.append(equations[i][0])
            interleaved_lines.append(equations[i][1])
        full_text = '\n'.join(interleaved_lines)
        text_box.set_text(full_text)
        cursor_dot.set_data([idx + 1], [w_values[idx + 1]])
        line_full.set_data(range(idx + 2), w_values[:idx + 2])
    fig.canvas.draw_idle()


step_slider.on_changed(update)
update(-1)
plt.show()
