"""
Example_B.py
Visual metaphor for gradient descent as a ball descending in a quadratic well
with step-by-step algebra.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the cost function (a simple quadratic well)
def cost_function(x):
    return x**2

def gradient(x):
    return 2 * x

# Gradient descent parameters
x_init = 2.5
alpha = 0.01
steps = [x_init]
equations = []

x = x_init
for i in range(30):
    grad = gradient(x)
    x_new = x - alpha * grad
    symbolic = rf"$x_{{{i+1}}} = x_{{{i}}} - \alpha \cdot \nabla J(x_{{{i}}})$"
    numeric = rf"$x_{{{i+1}}} = {x:.4f} - {alpha} \cdot 2 \cdot {x:.4f} = {x_new:.4f}$"
    equations.append((symbolic, numeric))
    steps.append(x_new)
    x = x_new

# Setup figure
fig, (ax_text, ax_plot) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})
fig.canvas.manager.set_window_title("Example_B")
plt.subplots_adjust(bottom=0.3, hspace=0.5)

x_vals = np.linspace(-3, 3, 400)
y_vals = cost_function(x_vals)

# Plot well
ax_plot.plot(x_vals, y_vals, 'b-', label='Cost function')
ball, = ax_plot.plot([], [], 'ro', markersize=12, label='Ball')
ax_plot.set_title("Gradient Descent: Ball in a Well")
ax_plot.set_xlabel("x")
ax_plot.set_ylabel("Cost")
ax_plot.set_xlim(-3, 3)
ax_plot.set_ylim(0, 10)
ax_plot.legend()

# Text box
text_box = ax_text.text(0.01, 0.95, '', transform=ax_text.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment='left')
ax_text.set_axis_off()

# Slider
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.05])
slider = Slider(slider_ax, 'Step', -1, len(steps) - 2, valinit=0, valstep=1)

def update(val):
    idx = int(slider.val)
    ball.set_data([steps[idx+1]], [cost_function(steps[idx+1])])
    interleaved_lines = []
    for i in range(idx + 1):
        interleaved_lines.append(equations[i][0])
        interleaved_lines.append(equations[i][1])
    text_box.set_text('\n'.join(interleaved_lines))
    fig.canvas.draw_idle()

slider.on_changed(update)
update(-1)
update(-1)
plt.show()
