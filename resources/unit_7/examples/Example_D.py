"""
Example_D.py
High-dimensional matrix learning with real-time visualizations and sliders for 
dimension and iteration control.

By Juan B. Gutiérrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

np.random.seed(2025)

n = 10
m = 1
alpha = 0.01
num_iterations = 200

fig = plt.figure()
fig.canvas.manager.set_window_title("Example_D")

slider_n_ax = plt.axes([0.1, 0.02, 0.35, 0.03])
slider_m_ax = plt.axes([0.55, 0.02, 0.35, 0.03])
slider_iter_ax = plt.axes([0.1, 0.06, 0.8, 0.03])

slider_n = Slider(slider_n_ax, 'n', 1, 50, valinit=n, valstep=1)
slider_m = Slider(slider_m_ax, 'm', 1, 10, valinit=m, valstep=1)
slider_iter = Slider(slider_iter_ax, 'Iteration', 0, num_iterations, valinit=0, valstep=1)

x = np.random.rand(n, m)
y_desired = np.random.rand(n, m)
W = np.random.rand(n, n)

Ws = [W.copy()]
y_actuals = [W @ x]
errors = [np.mean((y_desired - Ws[-1] @ x) ** 2)]

for _ in range(num_iterations):
    y_actual = Ws[-1] @ x
    gradient_W = -2 * (y_desired - y_actual) @ x.T
    W_new = Ws[-1] - alpha * gradient_W
    Ws.append(W_new)
    y_actuals.append(W_new @ x)
    errors.append(np.mean((y_desired - W_new @ x) ** 2))

axes_to_clear = []

def make_axes(n, m):
    global axes_to_clear
    for ax in axes_to_clear:
        ax.remove()
    gs = fig.add_gridspec(2, 5, height_ratios=[15, 1])
    ax_ydesired = fig.add_subplot(gs[0, 0])
    ax_yactual = fig.add_subplot(gs[0, 1])
    ax_matrix = fig.add_subplot(gs[0, 2])
    ax_x = fig.add_subplot(gs[0, 3])
    ax_error = fig.add_subplot(gs[0, 4])
    
    # Hide axes of the first four panels
    ax_ydesired.axis('off')
    ax_yactual.axis('off')
    ax_matrix.axis('off')
    ax_x.axis('off')
    
    axes_to_clear = [ax_ydesired, ax_yactual, ax_matrix, ax_x, ax_error]
    return ax_ydesired, ax_yactual, ax_matrix, ax_x, ax_error


ax_ydesired, ax_yactual, ax_matrix, ax_x, ax_error = make_axes(n, m)

ay_ydesired_display = ax_ydesired.imshow(y_desired, aspect='auto', cmap='Oranges', vmin=0, vmax=1)
y_display = ax_yactual.imshow(y_actuals[0], aspect='auto', cmap='Reds', vmin=0, vmax=1)
matrix_display = ax_matrix.imshow(Ws[0], aspect='auto', cmap='viridis', vmin=0, vmax=1)
x_display = ax_x.imshow(x, aspect='auto', cmap='gray', vmin=0, vmax=1)

ax_ydesired.set_title("y_desired")
ax_yactual.set_title("y_actual")
ax_matrix.set_title("W")
ax_x.set_title("x")
ax_error.set_title("Error over Iterations")

error_line, = ax_error.plot(range(num_iterations + 1), errors, 'r-')
current_iter_line = ax_error.axvline(0, color='black', linestyle='--')
ax_error.set_xlim(0, num_iterations)
ax_error.set_ylim(0, 0.25)
ax_error.set_xlabel("Iteration")
ax_error.set_ylabel("Mean Squared Error")

def recompute():
    global x, y_desired, W, Ws, y_actuals, errors
    global ax_ydesired, ax_yactual, ax_matrix, ax_x, ax_error
    n_val = int(slider_n.val)
    m_val = int(slider_m.val)
    x = np.random.rand(n_val, m_val)
    y_desired = np.random.rand(n_val, m_val)
    W = np.random.rand(n_val, n_val)

    Ws = [W.copy()]
    y_actuals = [W @ x]
    errors = [np.mean((y_desired - Ws[-1] @ x) ** 2)]
    for _ in range(num_iterations):
        y_actual = Ws[-1] @ x
        gradient_W = -2 * (y_desired - y_actual) @ x.T
        W_new = Ws[-1] - alpha * gradient_W
        Ws.append(W_new)
        y_actuals.append(W_new @ x)
        errors.append(np.mean((y_desired - W_new @ x) ** 2))

    ax_ydesired, ax_yactual, ax_matrix, ax_x, ax_error = make_axes(n_val, m_val)

    ax_ydesired.imshow(y_desired, aspect='auto', cmap='Oranges', vmin=0, vmax=1)
    ax_yactual.imshow(y_actuals[0], aspect='auto', cmap='Reds', vmin=0, vmax=1)
    ax_matrix.imshow(Ws[0], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax_x.imshow(x, aspect='auto', cmap='gray', vmin=0, vmax=1)

    ax_ydesired.set_title("y_desired")
    ax_yactual.set_title("y_actual")
    ax_matrix.set_title("W")
    ax_x.set_title("x")
    ax_error.set_title("Error over Iterations")

    ax_error.plot(range(num_iterations + 1), errors, 'r-')
    ax_error.axvline(0, color='black', linestyle='--')
    ax_error.set_xlim(0, num_iterations)
    ax_error.set_ylim(0, 0.25)
    ax_error.set_xlabel("Iteration")
    ax_error.set_ylabel("Mean Squared Error")
    fig.canvas.draw_idle()

def update(val):
    idx = int(slider_iter.val)
    ax_yactual.images[0].set_data(y_actuals[idx])
    ax_matrix.images[0].set_data(Ws[idx])
    ax_error.lines[1].set_xdata([idx, idx])
    fig.canvas.draw_idle()

slider_iter.on_changed(update)
slider_n.on_changed(lambda val: recompute())
slider_m.on_changed(lambda val: recompute())

plt.show()
