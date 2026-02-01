import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_field(X, Y):
    charges = [{'q': 1, 'pos': (-1, 0)}, {'q': -1, 'pos': (1, 0)}]
    Ex, Ey, V = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(X.shape)
    for c in charges:
        dx, dy = X - c['pos'][0], Y - c['pos'][1]
        r = np.sqrt(dx ** 2 + dy ** 2 + 0.1)
        V += c['q'] / r
        Ex += c['q'] * dx / r ** 3
        Ey += c['q'] * dy / r ** 3
    return Ex, Ey, V


def calculate_gauss_flux(center, radius, n_points=100):

    angles = np.linspace(0, 2 * np.pi, n_points)
    px = center[0] + radius * np.cos(angles)
    py = center[1] + radius * np.sin(angles)

    def get_e_at_points(x_p, y_p):
        ex, ey = np.zeros_like(x_p), np.zeros_like(y_p)
        for c in [{'q': 1, 'pos': (-1, 0)}, {'q': -1, 'pos': (1, 0)}]:
            dx, dy = x_p - c['pos'][0], y_p - c['pos'][1]
            r3 = (dx ** 2 + dy ** 2 + 0.01) ** 1.5
            ex += c['q'] * dx / r3
            ey += c['q'] * dy / r3
        return ex, ey

    Ex_p, Ey_p = get_e_at_points(px, py)

    nx = np.cos(angles)
    ny = np.sin(angles)


    dot_product = Ex_p * nx + Ey_p * ny
    dl = (2 * np.pi * radius) / n_points
    flux = np.sum(dot_product) * dl

    return px, py, flux


gauss_x, gauss_y, flux_value = calculate_gauss_flux(center=(-1, 0), radius=0.5)

print(f"Obliczony strumień Gaussa: {flux_value:.4f}")
print(f"Teoretyczny ładunek wewnątrz: {flux_value / (2 * np.pi):.4f}")

x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Ex, Ey, V = get_field(X, Y)

n_particles = 15
px = np.random.uniform(-5, 5, n_particles ** 2)
py = np.random.uniform(-5, 5, n_particles ** 2)

fig, ax = plt.subplots(figsize=(8, 7))

cp = ax.contourf(X, Y, V, levels=40, cmap='RdBu', alpha=0.5)
plt.colorbar(cp, label='Potencjał V')

ax.streamplot(X, Y, Ex, Ey, color='#dddddd', linewidth=0.8)

ax.scatter([-1], [0], color='red', s=150, edgecolors='black', label='+', zorder=5)
ax.scatter([1], [0], color='blue', s=150, edgecolors='black', label='-', zorder=5)

Q = ax.quiver(px, py, np.zeros_like(px), np.zeros_like(py),
              pivot='mid', color='black', scale=25, width=0.005)

ax.set_title("Animowane pole elektryczne dipola")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.legend()



levels = np.linspace(-2, 2, 21)
contours = ax.contour(X, Y, V, levels=levels, colors='white', linewidths=0.5, alpha=0.3)
plt.plot(gauss_x, gauss_y, 'g--', linewidth=2, label=f'Powierzchnia Gaussa (Φ={flux_value:.2f})')

Q = ax.quiver(px, py, np.zeros_like(px), np.zeros_like(py),
              np.zeros_like(px),
              pivot='mid', cmap='inferno', scale=25, width=0.005)

def update(frame):
    global px, py

    def get_e(x_pos, y_pos):
        ex, ey = np.zeros_like(x_pos), np.zeros_like(y_pos)
        for c in [{'q': 1, 'pos': (-1, 0)}, {'q': -1, 'pos': (1, 0)}]:
            dx, dy = x_pos - c['pos'][0], y_pos - c['pos'][1]
            r3 = (dx ** 2 + dy ** 2 + 0.05) ** 1.5
            ex += c['q'] * dx / r3
            ey += c['q'] * dy / r3
        return ex, ey

    ex1, ey1 = get_e(px, py)
    dt = 0.15
    px_mid, py_mid = px + ex1 * dt * 0.5, py + ey1 * dt * 0.5
    ex2, ey2 = get_e(px_mid, py_mid)

    px += ex2 * dt
    py += ey2 * dt

    mag = np.sqrt(ex2 ** 2 + ey2 ** 2) + 1e-9
    color_intensity = np.log1p(mag)

    mask = (np.abs(px) > 5) | (np.abs(py) > 5) | (np.sqrt((px + 1) ** 2 + py ** 2) < 0.15) | (
                np.sqrt((px - 1) ** 2 + py ** 2) < 0.15)
    px[mask] = np.random.uniform(-5, 5, np.sum(mask))
    py[mask] = np.random.uniform(-5, 5, np.sum(mask))

    Q.set_offsets(np.column_stack([px, py]))
    Q.set_UVC(ex2 / mag, ey2 / mag, color_intensity)
    return Q,


print("Generowanie animacji (proszę czekać)...")
ani = FuncAnimation(fig, update, frames=100, interval=40, blit=False)

try:
    ani.save('dipol.gif', writer='pillow')
    print("Sukces! Plik 'dipol.gif' został zapisany.")
except Exception as e:
    print(f"Błąd podczas zapisu: {e}")

plt.show()