# -*- coding: utf-8 -*-

# --- 0. Importaciones ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import splu
import time
import os

# --- 1. Funciones Auxiliares ---

def gaussian_wavepacket(x_nodes, x0, sigma0, k0, hbar=1.0):
    """Crea un paquete de ondas Gaussiano viajero (Eq. 3.34 / 3.35)."""
    a = 1.0 / (4.0 * sigma0**2)
    norm_factor = (2.0 * a / np.pi)**0.25
    spatial_envelope = np.exp(-a * (x_nodes - x0)**2)
    momentum_phase = np.exp(1j * k0 * x_nodes)
    return norm_factor * spatial_envelope * momentum_phase

def get_norm(psi_interior, dx):
    """Calcula la norma (integral de |Psi|^2)."""
    rho = np.abs(psi_interior)**2
    norm = np.sum(rho) * dx
    return norm

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":

    print("Iniciando simulación: 5.2 Pozo Infinito...")

    # Directorio de salida
    OUTPUT_DIR = "5.2"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directorio de salida: '{OUTPUT_DIR}/'")

    # Constantes físicas (adimensionales)
    HBAR = 1.0
    M = 1.0

    # Parámetros del dominio espacial
    L = 180.0
    NX = 3600
    N_INTERIOR = NX - 1
    DX = L / NX
    x_full = np.linspace(0, L, NX + 1)
    x = x_full[1:-1]  # Malla interior

    # Parámetros del dominio temporal
    DT = 5e-3
    T_FINAL = 220.0  # extendido para ver la reflexión completa
    N_STEPS = int(T_FINAL / DT)

    # Parámetros del paquete Gaussiano
    X0 = 45.0
    K0 = 1.0
    SIGMA0 = 5.0
    VG = HBAR * K0 / M

    # Tiempos para snapshots
    snapshot_times = [0.0, 80.0, 160.0, 220.0]
    snapshot_steps = [int(t / DT) for t in snapshot_times]
    snapshots_psi = []

    print(f"Dominio: L={L}, Nx={NX}, dx={DX:.4f}")
    print(f"Tiempo: T={T_FINAL}, Nt={N_STEPS}, dt={DT}")
    print(f"Paquete: x0={X0}, k0={K0}, sigma0={SIGMA0}")
    print(f"Snapshots en t = {snapshot_times}")

    # --- 3. Hamiltoniano Discreto ---
    V_potential = np.zeros(N_INTERIOR)
    diag_T = (HBAR**2 / (M * DX**2)) * np.ones(N_INTERIOR)
    off_diag_T = (-HBAR**2 / (2 * M * DX**2)) * np.ones(N_INTERIOR - 1)

    H_sparse = scipy.sparse.diags(
        [off_diag_T, diag_T + V_potential, off_diag_T],
        [-1, 0, 1],
        shape=(N_INTERIOR, N_INTERIOR),
        format='csc'
    )
    print("Hamiltoniano H_D (V=0 interior) construido.")

    # --- 4. Condición Inicial ---
    psi_0_full = gaussian_wavepacket(x_full, X0, SIGMA0, K0, HBAR)
    psi_0_full[0] = 0.0
    psi_0_full[-1] = 0.0
    psi_0 = psi_0_full[1:-1]

    norm_inicial = np.sum(np.abs(psi_0)**2) * DX
    psi_0 = psi_0 / np.sqrt(norm_inicial)
    print(f"Condición inicial normalizada (Norma = {get_norm(psi_0, DX):.6f})")

    # --- 5. Matrices CN ---
    I = scipy.sparse.eye(N_INTERIOR, dtype=complex, format='csc')
    A_cn = I + 1j * (DT / (2.0 * HBAR)) * H_sparse
    B_cn = I - 1j * (DT / (2.0 * HBAR)) * H_sparse

    print("Factorizando matriz CN...")
    solve_cn = splu(A_cn)
    print("Factorización completada.")

    # --- 6. Simulación ---
    print("Iniciando bucle CN (Pozo Infinito)...")
    start_time = time.time()

    psi_cn = psi_0.copy()
    norm_data = np.zeros(N_STEPS + 1)
    norm_data[0] = get_norm(psi_cn, DX)
    time_axis = np.linspace(0, T_FINAL, N_STEPS + 1)

    if snapshot_steps[0] == 0:
        snapshots_psi.append(psi_0.copy())
        snapshot_steps.pop(0)

    for n in range(N_STEPS):
        rhs_cn = B_cn.dot(psi_cn)
        psi_cn = solve_cn.solve(rhs_cn)
        norm_data[n+1] = get_norm(psi_cn, DX)

        if (n + 1) in snapshot_steps:
            snapshots_psi.append(psi_cn.copy())

        if (n + 1) % (N_STEPS // 10) == 0:
            print(f"  Progreso: {((n+1)/N_STEPS)*100:.0f}%")

    end_time = time.time()
    print(f"Simulación completada en {end_time - start_time:.2f} s.")

    # --- 7. Gráficas ---
    print("Generando gráficas...")
    plt.style.use('seaborn-v0_8-deep')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.family'] = 'serif'

    # --- Gráfica 1: Snapshots de la Evolución ---
fig_path = os.path.join(OUTPUT_DIR, 'TFG_Figura_5_2_Snapshots.png')
fig, axs = plt.subplots(len(snapshot_times), 1, figsize=(8, 10), sharex=True)

for i, t_val in enumerate(snapshot_times):
    rho_full = np.abs(np.concatenate(([0+0j], snapshots_psi[i], [0+0j])))**2
    rho_max = np.max(rho_full)

    axs[i].plot(x_full, rho_full, linewidth=2, color=f'C{i}')
    axs[i].set_title(f'$t = {int(t_val)}$', fontsize=10)     # ← CAMBIO PEDIDO
    axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=10)       # ← CAMBIO PEDIDO
    axs[i].grid(True, linestyle=':', alpha=0.7)
    axs[i].set_ylim(0.0, rho_max * 1.1)
    axs[i].axvline(L, color='k', linestyle='--', linewidth=1.5, alpha=0.5)

axs[-1].set_xlabel('Posición $x$', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(fig_path)
print(f"Figura '{fig_path}' guardada.")

    # --- Gráfica 2: Conservación de la Norma ---
from matplotlib.ticker import FormatStrFormatter

fig_path = os.path.join(OUTPUT_DIR, 'TFG_Figura_5_2_Norma.png')
plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.plot(time_axis, norm_data, label='Norma (Crank–Nicolson)', linewidth=2)

ax.set_title('Conservación de la norma en el pozo infinito', fontsize=14)
ax.set_xlabel('Tiempo $t$', fontsize=12)
ax.set_ylabel(r'Norma total $\int |\Psi|^2 dx$', fontsize=12)

# Rango cercano a 1 y SIN offset en el eje y
ax.set_ylim(0.9998, 1.0002)                       # un poco más ancho para que se vean bien los ticks
ax.ticklabel_format(axis='y', style='plain', useOffset=False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

ax.legend(fontsize=10)
ax.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(fig_path)
print(f"Figura '{fig_path}' guardada.")
