# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import splu
import time
import os

# -------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# -------------------------------------------

print("Iniciando simulación Oscilador Armónico (estado coherente, CN)...")

OUTPUT_DIR = "5.5"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Directorio de salida: '{OUTPUT_DIR}/'")

HBAR  = 1.0
M     = 1.0
OMEGA = 0.1          # frecuencia del oscilador
x_c   = 0.0          # centro del potencial

# Dominio espacial centrado
L = 180.0
NX = 3600
N_INTERIOR = NX - 1
DX = L / NX
x_full = np.linspace(-L/2, L/2, NX + 1)
x = x_full[1:-1]

# Dominio temporal
DT = 5e-3
T_FINAL = 80.0
N_STEPS = int(T_FINAL / DT)

# -------------------------------------------
# 2. ESTADO COHERENTE — PARÁMETROS
# -------------------------------------------

SIGMA0 = np.sqrt(HBAR / (2*M*OMEGA))  # anchura mínima del HO
X0     = -30.0                       # desplazamiento inicial
K0     = 0.7                         # momento inicial (elige el que quieras)

print(f"Paquete coherente: sigma0={SIGMA0:.3f}, X0={X0}, K0={K0}")

# -------------------------------------------
# 3. HAMILTONIANO: T + V_armónico
# -------------------------------------------

V_potential = 0.5 * M * OMEGA**2 * (x - x_c)**2

diag_T     = (HBAR**2 / (M * DX**2)) * np.ones(N_INTERIOR)
off_diag_T = (-HBAR**2 / (2 * M * DX**2)) * np.ones(N_INTERIOR - 1)

H_sparse = scipy.sparse.diags(
    [off_diag_T, diag_T + V_potential, off_diag_T],
    [-1, 0, 1],
    shape=(N_INTERIOR, N_INTERIOR),
    format='csc'
)

print("Hamiltoniano construido correctamente.")

# -------------------------------------------
# 4. ESTADO COHERENTE INICIAL
# -------------------------------------------

def coherent_state(x_nodes, x0, sigma, k0, x_c):
    """
    Estado coherente real del HO:
    gaussiana mínima + fase respecto al centro del potencial.
    """
    return (1.0/(2*np.pi*sigma**2))**0.25 * \
           np.exp(-(x_nodes - x0)**2 / (4*sigma**2)) * \
           np.exp(1j * k0 * (x_nodes - x_c))

psi_0_full = coherent_state(x_full, X0, SIGMA0, K0, x_c)

# Condiciones de contorno tipo pozo infinito
psi_0_full[0]  = 0.0
psi_0_full[-1] = 0.0
psi_0 = psi_0_full[1:-1]

# Normalización
norm0 = np.sum(np.abs(psi_0)**2) * DX
psi_0 = psi_0 / np.sqrt(norm0)

print(f"Norma inicial = {np.sum(np.abs(psi_0)**2)*DX:.6f}")

# -------------------------------------------
# 5. CRANK–NICOLSON
# -------------------------------------------

I    = scipy.sparse.eye(N_INTERIOR, dtype=complex, format='csc')
A_cn = I + 1j*(DT/(2*HBAR))*H_sparse
B_cn = I - 1j*(DT/(2*HBAR))*H_sparse

solve_cn = splu(A_cn)
print("Factorización LU completada.")

def get_observables(psi, x, dx):
    rho = np.abs(psi)**2
    norm = np.sum(rho)*dx
    rho_n = rho/norm
    x_mean = np.sum(x*rho_n)*dx
    x_var  = np.sum((x-x_mean)**2*rho_n)*dx
    return norm, x_mean, np.sqrt(x_var)

# Solución analítica clásica
def x_theory(t):
    p0 = HBAR * K0
    return x_c + (X0 - x_c)*np.cos(OMEGA*t) + (p0/(M*OMEGA))*np.sin(OMEGA*t)

# -------------------------------------------
# 6. SIMULACIÓN
# -------------------------------------------

psi_cn = psi_0.copy()
data = np.zeros((N_STEPS + 1, 4))
time_axis = np.linspace(0, T_FINAL, N_STEPS + 1)

data[0,:] = (0.0,) + get_observables(psi_cn, x, DX)

snapshot_times = [0, 20, 40, 60]
snapshot_steps = [int(s/DT) for s in snapshot_times]
snapshots = []

if 0 in snapshot_steps:
    snapshots.append(psi_cn.copy())

for n in range(N_STEPS):
    rhs = B_cn.dot(psi_cn)
    psi_cn = solve_cn.solve(rhs)

    data[n+1,:] = (time_axis[n+1],) + get_observables(psi_cn, x, DX)

    if (n+1) in snapshot_steps:
        snapshots.append(psi_cn.copy())

# -------------------------------------------
# 7. GRÁFICAS
# -------------------------------------------

plt.style.use("seaborn-v0_8-deep")
plt.rcParams["figure.dpi"] = 140

# --- Snapshots ---
fig, axs = plt.subplots(len(snapshot_times), 1, figsize=(8,9), sharex=True)

V_plot = 0.5*M*OMEGA**2*(x_full-x_c)**2
initial_peak = np.max(np.abs(snapshots[0])**2)
ylim = initial_peak*1.2

for i, t in enumerate(snapshot_times):
    rho = np.abs(np.concatenate(([0], snapshots[i], [0])))**2
    axs[i].plot(x_full, rho, lw=2)
    axs[i].plot(x_full, V_plot/np.max(V_plot)*ylim*0.7, "--", color="gray")
    axs[i].set_ylim(0, ylim)
    axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=10)
    axs[i].set_title(f"t = {t}")
    axs[i].grid(True, linestyle=':', color='0.85')   

axs[-1].set_xlabel("Posición x")
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig(f"{OUTPUT_DIR}/TFG_Figura_5_5_Snapshots.png")

# --- Posición media ---
plt.figure(figsize=(8,5))
plt.plot(time_axis, data[:,2], label="⟨x⟩(t) numérica", lw=2)
plt.plot(time_axis, x_theory(time_axis), "k--", label="⟨x⟩(t) clásica", lw=1.8)
plt.title("Posición media del estado coherente")
plt.xlabel("Tiempo t")
plt.ylabel("Posición media ⟨x⟩(t)")
plt.grid(True, linestyle=':', color='0.85')         
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/TFG_Figura_5_5_PosicionMedia.png")

print("\n¡Simulación completada correctamente! Estado coherente generado y simulación guardada.")