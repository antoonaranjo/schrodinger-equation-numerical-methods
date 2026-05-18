# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve, splu
import time
import os 

# --- 1. Configuración de la Simulación (Parámetros Físicos y Numéricos) ---
print("Iniciando simulación (v2: optimizada)...")

# Directorio de salida
OUTPUT_DIR = "5.1"
# Crear el directorio si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Directorio de salida: '{OUTPUT_DIR}/'")


HBAR = 1.0
M = 1.0

L = 180.0
NX = 3600
N_INTERIOR = NX - 1
DX = L / NX
x_full = np.linspace(0, L, NX + 1)
x = x_full[1:-1]

DT = 5e-3
T_FINAL = 80.0
N_STEPS = int(T_FINAL / DT)

X0 = 45.0
K0 = 1.0
SIGMA0 = 5.0
VG = HBAR * K0 / M

print(f"Dominio: L={L} (Nx={NX}, dx={DX:.4f})")
print(f"Tiempo: T={T_FINAL} (Nt={N_STEPS}, dt={DT})")
print(f"Paquete: x0={X0}, k0={K0}, sigma0={SIGMA0}")

# --- 2. Construcción del Hamiltoniano Discreto ---
V_potential = np.zeros(N_INTERIOR)
diag_T = (HBAR**2 / (M * DX**2)) * np.ones(N_INTERIOR)
off_diag_T = (-HBAR**2 / (2 * M * DX**2)) * np.ones(N_INTERIOR - 1)

H_sparse = scipy.sparse.diags(
    [off_diag_T, diag_T + V_potential, off_diag_T],
    [-1, 0, 1],
    shape=(N_INTERIOR, N_INTERIOR),
    format='csc'
)

print("Hamiltoniano H_D (interior) construido.")

# --- 3. Condición Inicial (Gaussiana Viajera) ---
def gaussian_wavepacket(x_nodes, x0, sigma0, k0):
    a = 1.0 / (4.0 * sigma0**2)
    norm_factor = (2.0 * a / np.pi)**0.25
    spatial_envelope = np.exp(-a * (x_nodes - x0)**2)
    momentum_phase = np.exp(1j * k0 * x_nodes)
    return norm_factor * spatial_envelope * momentum_phase

psi_0_full = gaussian_wavepacket(x_full, X0, SIGMA0, K0)
psi_0_full[0] = 0.0
psi_0_full[-1] = 0.0
psi_0 = psi_0_full[1:-1]

norm_inicial = np.sum(np.abs(psi_0)**2) * DX
psi_0 = psi_0 / np.sqrt(norm_inicial)
print(f"Condición inicial creada y normalizada (Norma = {np.sum(np.abs(psi_0)**2) * DX:.6f})")

# --- 4. Matrices de Evolución ---
I = scipy.sparse.eye(N_INTERIOR, dtype=complex, format='csc')
M_ee = I - 1j * (DT / HBAR) * H_sparse
A_ie = I + 1j * (DT / HBAR) * H_sparse
A_cn = I + 1j * (DT / (2.0 * HBAR)) * H_sparse
B_cn = I - 1j * (DT / (2.0 * HBAR)) * H_sparse
print("Matrices de evolución (EE, IE, CN) construidas.")

# --- 4b. Factorización LU ---
print("Factorizando matrices para IE y CN...")
t_fact_start = time.time()
solve_ie = splu(A_ie)
solve_cn = splu(A_cn)
t_fact_end = time.time()
print(f"Factorización completada en {t_fact_end - t_fact_start:.4f} segundos.")

# --- 5. Observables ---
def get_observables(psi_interior, x_interior, dx):
    rho = np.abs(psi_interior)**2
    norm = np.sum(rho) * dx
    rho_norm = rho / norm if norm > 1e-10 else rho
    x_mean = np.sum(x_interior * rho_norm) * dx
    x_var = np.sum((x_interior - x_mean)**2 * rho_norm) * dx
    sigma_x = np.sqrt(x_var)
    return norm, x_mean, sigma_x

# --- 6. Soluciones Analíticas ---
def x_theory(t):
    return X0 + (HBAR * K0 / M) * t

def sigma_theory(t):
    return np.sqrt(SIGMA0**2 + (HBAR * t / (2 * M * SIGMA0))**2)

# --- 7. Resultados ---
print("Iniciando resultados...")
start_time = time.time()

psi_cn = psi_0.copy()
psi_ee = psi_0.copy()
psi_ie = psi_0.copy()

data_cn = np.zeros((N_STEPS + 1, 4))
data_ee = np.zeros((N_STEPS + 1, 4))
data_ie = np.zeros((N_STEPS + 1, 4))
time_axis = np.linspace(0, T_FINAL, N_STEPS + 1)

data_cn[0,:] = (0.0,) + get_observables(psi_cn, x, DX)
data_ee[0,:] = (0.0,) + get_observables(psi_ee, x, DX)
data_ie[0,:] = (0.0,) + get_observables(psi_ie, x, DX)

for n in range(N_STEPS):
    rhs_cn = B_cn.dot(psi_cn)
    psi_cn = solve_cn.solve(rhs_cn)
    psi_ee = M_ee.dot(psi_ee)
    psi_ie = solve_ie.solve(psi_ie)
    data_cn[n+1,:] = (time_axis[n+1],) + get_observables(psi_cn, x, DX)
    data_ee[n+1,:] = (time_axis[n+1],) + get_observables(psi_ee, x, DX)
    data_ie[n+1,:] = (time_axis[n+1],) + get_observables(psi_ie, x, DX)
    if (n + 1) % (N_STEPS // 10) == 0:
        print(f"  Progreso: {((n+1)/N_STEPS)*100:.0f}%")

end_time = time.time()
print(f"Resultados completados en {end_time - start_time:.2f} segundos.")


# --- 7b. Cálculo de errores de conservación de la norma ---
error_max_cn = np.max(np.abs(data_cn[:,1] - 1))
error_max_ie = np.max(np.abs(data_ie[:,1] - 1))
error_max_ee = np.max(np.abs(data_ee[:,1] - 1))

error_final_cn = abs(data_cn[-1,1] - 1)
error_final_ie = abs(data_ie[-1,1] - 1)
error_final_ee = abs(data_ee[-1,1] - 1)

print("\n--- Errores de conservación de la norma ---")
print(f"Crank–Nicolson → error máximo = {error_max_cn:.2e}, error final = {error_final_cn:.2e}")
print(f"Euler Implícito → error máximo = {error_max_ie:.2e}, error final = {error_final_ie:.2e}")
print(f"Euler Explícito → error máximo = {error_max_ee:.2e}, error final = {error_final_ee:.2e}")
print("-------------------------------------------")

# --- 8. Gráficas ---
print("Generando gráficas...")
plt.style.use('seaborn-v0_8-deep')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'serif'

# --- Gráfica 1: Posición media (SOLO Crank–Nicolson) ---
plt.figure(figsize=(8, 5))
plt.plot(data_cn[:,0], data_cn[:,2], label='$\\langle x \\rangle_{num}$ (Crank–Nicolson)', linewidth=2)
plt.plot(time_axis, x_theory(time_axis), 'k--', label='$\\langle x \\rangle_{ana} = x_0 + v_g t$', linewidth=2)
plt.title('Evolución de la posición media del paquete de ondas', fontsize=14)
plt.xlabel('Tiempo $t$', fontsize=12)
plt.ylabel('Posición media $\\langle x \\rangle(t)$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'TFG_Figura_5_1_Posicion.png')) # <--- MODIFICADO
print(f"Figura '{os.path.join(OUTPUT_DIR, 'TFG_Figura_5_1_Posicion.png')}' guardada.")

# --- Gráfica 2: Anchura ---
plt.figure(figsize=(8, 5))
plt.plot(data_cn[:,0], data_cn[:,3], label='$\\sigma_{x, num}$ (Crank–Nicolson)', linewidth=2)
plt.plot(time_axis, sigma_theory(time_axis), 'k--', label='$\\sigma_{x, ana}(t)$', linewidth=2)
plt.title('Ensanchamiento temporal del paquete de ondas', fontsize=14)
plt.xlabel('Tiempo $t$', fontsize=12)
plt.ylabel('Anchura $\\sigma_x(t)$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'TFG_Figura_5_1_Anchura.png')) # <--- MODIFICADO
print(f"Figura '{os.path.join(OUTPUT_DIR, 'TFG_Figura_5_1_Anchura.png')}' guardada.")

# --- Gráfica 3: Norma ---
plt.figure(figsize=(8, 5))

# CN igual que antes
plt.plot(data_cn[:,0], data_cn[:,1], label='Crank–Nicolson (Unitario)', linewidth=2)

# Euler Explícito — ahora color C2
plt.plot(data_ee[:,0], data_ee[:,1], 
         label='Euler Explícito (Inestable)', linewidth=2, linestyle='--', color='C2')

# Euler Implícito — ahora color C1
plt.plot(data_ie[:,0], data_ie[:,1], 
         label='Euler Implícito (Disipativo)', linewidth=2, linestyle=':', color='C1')

plt.ylim(0, 2.0)

plt.title('Conservación de la norma del paquete de ondas', fontsize=14)
plt.xlabel('Tiempo $t$', fontsize=12)
plt.ylabel('Norma total $\\int |\\Psi|^2 dx$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', color='0.85')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'TFG_Figura_5_7_Norma.png'))
print(f"Figura '{os.path.join(OUTPUT_DIR, 'TFG_Figura_5_7_Norma.png')}' guardada.")

# --- Gráfica 4: Estado final ---
x_plot = x_full
psi_0_plot = np.abs(psi_0_full)**2
rho_cn = np.abs(np.concatenate(([0], psi_cn, [0])))**2
rho_ee = np.abs(np.concatenate(([0], psi_ee, [0])))**2
rho_ie = np.abs(np.concatenate(([0], psi_ie, [0])))**2

plt.figure(figsize=(8, 5))
plt.plot(x_plot, rho_cn, label=f'Crank–Nicolson (Norma={data_cn[-1,1]:.3f})', linewidth=2)
plt.plot(x_plot, rho_ie, label=f'Euler Implícito (Norma={data_ie[-1,1]:.3f})', linewidth=2, linestyle=':')
plt.plot(x_plot, psi_0_plot, 'k-', label='Estado inicial $|\\Psi(x,0)|^2$', linewidth=1, alpha=0.3)
plt.title(f'Comparativa del estado final $|\\Psi(x,T)|^2$', fontsize=14)
plt.xlabel('Posición $x$', fontsize=12)
plt.ylabel('Densidad de probabilidad $|\\Psi|^2$', fontsize=12)
plt.legend(fontsize=10)
max_y = np.max(rho_cn) * 1.5
if np.max(rho_ee) < max_y * 10:
    plt.ylim(0, max_y)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'TFG_Figura_5_7_EstadoFinal.png')) # <--- MODIFICADO
print(f"Figura '{os.path.join(OUTPUT_DIR, 'TFG_Figura_5_7_EstadoFinal.png')}' guardada.")

print("\n¡Proceso completado! Todas las figuras han sido guardadas.")

# --- 9. Errores cuantitativos ---

# 9.1 Errores de norma (todos los esquemas)
err_norm_cn = np.abs(data_cn[:,1] - 1.0)
err_norm_ie = np.abs(data_ie[:,1] - 1.0)
err_norm_ee = np.abs(data_ee[:,1] - 1.0)

err_norm_cn_max, err_norm_cn_final = np.max(err_norm_cn), err_norm_cn[-1]
err_norm_ie_max, err_norm_ie_final = np.max(err_norm_ie), err_norm_ie[-1]
err_norm_ee_max, err_norm_ee_final = np.max(err_norm_ee), err_norm_ee[-1]

# 9.2 Errores de posición media y anchura (CN vs analítica)
x_theo      = x_theory(time_axis)
sigma_theo  = sigma_theory(time_axis)

err_x_cn        = np.abs(data_cn[:,2] - x_theo)
err_sigma_cn    = np.abs(data_cn[:,3] - sigma_theo)

err_x_cn_max,     err_x_cn_final     = np.max(err_x_cn),     err_x_cn[-1]
err_sigma_cn_max, err_sigma_cn_final = np.max(err_sigma_cn), err_sigma_cn[-1]

# 9.3 (Opcional) Error L∞ y L1 en el tiempo para CN (posición y anchura)
# L∞ ya lo hemos llamado "máximo"; L1 en el tiempo:
from numpy import trapz
L1_x_cn     = trapz(err_x_cn,     time_axis) / (time_axis[-1] - time_axis[0])
L1_sigma_cn = trapz(err_sigma_cn, time_axis) / (time_axis[-1] - time_axis[0])

# 9.4 Mostrar resumen
print("\n=== RESUMEN DE ERRORES ===")
print(f"Norma — CN:  max={err_norm_cn_max:.3e}, final={err_norm_cn_final:.3e}")
print(f"Norma — IE:  max={err_norm_ie_max:.3e}, final={err_norm_ie_final:.3e}")
print(f"Norma — EE:  max={err_norm_ee_max:.3e}, final={err_norm_ee_final:.3e}")

print(f"Posición <x>(t) — CN:  max={err_x_cn_max:.3e}, final={err_x_cn_final:.3e}, L1_t={L1_x_cn:.3e}")
print(f"Anchura  σ_x(t) — CN:  max={err_sigma_cn_max:.3e}, final={err_sigma_cn_final:.3e}, L1_t={L1_sigma_cn:.3e}")

# 9.5 Guardar a CSV (para tabla en el TFG o anexos)
import csv
csv_path = os.path.join(OUTPUT_DIR, 'TFG_Errores_Resumen.csv') # <--- MODIFICADO
with open(csv_path,'w',newline='') as f: # <--- MODIFICADO
    w = csv.writer(f)
    w.writerow(["Métrica","Esquema","Error máx","Error final","L1_t"])
    w.writerow(["Norma","Crank-Nicolson", f"{err_norm_cn_max:.6e}", f"{err_norm_cn_final:.6e}", "—"])
    w.writerow(["Norma","Euler Implícito", f"{err_norm_ie_max:.6e}", f"{err_norm_ie_final:.6e}", "—"])
    w.writerow(["Norma","Euler Explícito", f"{err_norm_ee_max:.6e}", f"{err_norm_ee_final:.6e}", "—"])
    w.writerow(["<x>(t)","Crank-Nicolson", f"{err_x_cn_max:.6e}", f"{err_x_cn_final:.6e}", f"{L1_x_cn:.6e}"])
    w.writerow(["σ_x(t)","Crank-Nicolson", f"{err_sigma_cn_max:.6e}", f"{err_sigma_cn_final:.6e}", f"{L1_sigma_cn:.6e}"])

print(f"Archivo '{csv_path}' generado.") # <--- MODIFICADO

# --- 9b. Errores relativos máximos (Crank-Nicolson vs analítica) ---

# Evitar divisiones por cero
x_theo = x_theory(time_axis)
sigma_theo = sigma_theory(time_axis)

# Cálculo de errores relativos instantáneos
rel_err_x_cn = np.abs((data_cn[:,2] - x_theo) / x_theo)
rel_err_sigma_cn = np.abs((data_cn[:,3] - sigma_theo) / sigma_theo)

# Error relativo máximo
E_rel_x_max = np.max(rel_err_x_cn)
E_rel_sigma_max = np.max(rel_err_sigma_cn)

# Mostrar resultados
print("\n=== ERRORES RELATIVOS MÁXIMOS (Crank–Nicolson) ===")
print(f"Posición media <x>(t): E_rel_max = {E_rel_x_max:.3e}")
print(f"Anchura σ_x(t):        E_rel_max = {E_rel_sigma_max:.3e}")