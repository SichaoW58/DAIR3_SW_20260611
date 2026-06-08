"""
Example_A1.py
Animated sigmoid-like function with tangent and derivative speedometer 
using polar coordinates.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
fig = plt.figure(figsize=(12, 5))
fig.canvas.manager.set_window_title("Example_A1")
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, polar=True)

# Right panel: Function plot
line, = ax1.plot([], [], lw=2)
tangent_line, = ax1.plot([], [], 'r--', lw=1)
ax1.set_xlim(0, 20)
ax1.set_ylim(0, 20)
ax1.set_title("Symmetric Sigmoid Function")

# Left panel: Speedometer for both positive and negative derivatives
needle, = ax2.plot([], [], lw=3)
ax2.set_ylim(0, 1)
ax2.set_yticklabels([])
ax2.set_xticks(np.linspace(0, 2 * np.pi, 9))
ax2.set_xticklabels(['1', '', '0', '', '-1', '', 'Inf', '', ''])
ax2.set_title("Derivative Speedometer")

# Initialization function
def init():
    line.set_data([], [])
    needle.set_data([], [])
    tangent_line.set_data([], [])
    return line, needle, tangent_line

# Animation function
def animate(i):
    t_val = t[:i]
    y_val = y[:i]
    line.set_data(t_val, y_val)

    max_abs_dy = max(abs(dy))
    angle = (derivative(t[i]) / max_abs_dy) * (np.pi / 2)
    angle = np.pi / 2 - angle  # rotate so zero is vertical
    needle.set_data([0, angle], [0, 1])

    # Compute tangent line using average slope from 2 neighboring points
    if i >= 1:
        t1, t2 = t[i - 1], t[i]
        y1, y2 = y[i - 1], y[i]
        slope = (y2 - y1) / (t2 - t1)
    else:
        slope = dy[i]  # fallback for first frame

    t_curr = t[i]
    y_curr = y[i]
    t_tangent = np.array([0, 20])
    y_tangent = slope * (t_tangent - t_curr) + y_curr
    tangent_line.set_data(t_tangent, y_tangent)

    return line, needle, tangent_line

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t),
                              interval=50, blit=True)

plt.tight_layout()
plt.show()