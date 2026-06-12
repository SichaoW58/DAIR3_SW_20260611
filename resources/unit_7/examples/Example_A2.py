"""
Example_A2.py
Slider-controlled visualization of a symmetric sigmoid function and its 
derivative with tangent lines.

By Juan B. Gutiérrez, Professor of Mathematics 
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Define a flat-start and flat-top sigmoid-like function and its derivative
def func(t):
    return 10 * (np.tanh((t - 5)/3) - np.tanh((t - 15)/3)) / 2 + 5

def derivative(t):
    return (10 / 6) * ((1 - np.tanh((t - 5)/3)**2) / 2 - (1 - np.tanh((t - 15)/3)**2) / 2)

# Time values
t = np.linspace(0, 20, 400)
y = func(t)
dy = derivative(t)

# Set up the figure and axes
fig = plt.figure(figsize=(12, 6))
fig.canvas.manager.set_window_title("Example_A2")
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.subplots_adjust(bottom=0.2)

# Left panel: Function plot
line, = ax1.plot([], [], lw=2)
tangent_line, = ax1.plot([], [], 'r--', lw=1)
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_title("Symmetric Sigmoid Function")
ax1.grid(True)

# Right panel: Derivative plot
deriv_line, = ax2.plot([], [], 'r--', lw=1)
ax2.set_xlim(0, 20)
ax2.set_ylim(min(dy) - 0.5, max(dy) + 0.5)
ax2.set_title("Derivative of Function")
ax2.grid(True)

# Slider axis and widget
ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
frame_slider = Slider(ax_slider, 'Frame', 0, len(t) - 1, valinit=0, valstep=1)

# Update function for slider
def update(val):
    i = int(frame_slider.val)
    t_val = t[:i]
    y_val = y[:i]
    dy_val = dy[:i]

    line.set_data(t_val, y_val)
    deriv_line.set_data(t_val, dy_val)

    if i >= 1:
        t1, t2 = t[i - 1], t[i]
        y1, y2 = y[i - 1], y[i]
        slope = (y2 - y1) / (t2 - t1)
    else:
        slope = dy[i]

    t_curr = t[i]
    y_curr = y[i]
    t_tangent = np.array([0, 20])
    y_tangent = slope * (t_tangent - t_curr) + y_curr
    tangent_line.set_data(t_tangent, y_tangent)

    fig.canvas.draw_idle()

frame_slider.on_changed(update)
update(0)

plt.show()
