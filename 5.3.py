# -*- coding: utf-8 -*-
"""
TFG - Capítulo 5.3: Pozo Finito de Potencial (Versión 15.3 - Corregida)

Simulación de un paquete de ondas Gaussiano en un estado "casi ligado"
dentro de un pozo de potencial finito.
Se usa Crank-Nicolson.

VERSIÓN 15.3: Corregido el eje Y de la gráfica de Norma Total
para que muestre el 1.0 de forma clara y sin offsets.
"""

# --- 0. Importaciones ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import splu
import time
import os
from matplotlib.ticker import FormatStrFormatter # Importar para formatear ejes

# --- 1. Funciones Auxiliares ---

def gaussian_wavepacket(x_nodes, x0, sigma0, k0, hbar=1.0):
    """Crea un paquete de ondas Gaussiano viajero (Eq. 3.34 / 3.35)."""
    # a = 1 / (4 * sigma_0^2)
    a = 1.0 / (4.0 * sigma0**2)
    
    # Factor de normalización (para integral de -inf a +inf)
    norm_factor = (2.0 * a / np.pi)**0.25
    
    # Envolvente espacial (Gaussiana)
    spatial_envelope = np.exp(-a * (x_nodes - x0)**2)
    
    # Fase de momento (onda plana)
    momentum_phase = np.exp(1j * k0 * x_nodes)
    
    return norm_factor * spatial_envelope * momentum_phase

def get_observables(psi_interior, dx, x_interior, pozo_indices):
    """
    Calcula la norma total y la probabilidad parcial dentro del pozo.
    """
    rho = np.abs(psi_interior)**2
    norm_total = np.sum(rho) * dx
    
    # Calcular la probabilidad solo dentro de los índices del pozo
    prob_in_well = np.sum(rho[pozo_indices]) * dx
    
    return norm_total, prob_in_well

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":

    # --- 2. Configuración de la Simulación ---
    print("Iniciando simulación: 5.3 Pozo Finito...")

    # Directorio de salida para figuras
    OUTPUT_DIR = "5.3" 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directorio de salida: '{OUTPUT_DIR}/'")

    # Constantes físicas (adimensionales)
    HBAR = 1.0
    M = 1.0

    # Parámetros del dominio espacial
    L = 180.0       # Longitud del dominio (0, L)
    NX = 3600       # Número de SUBINTERVALOS (N_x en tu TFG)
    N_INTERIOR = NX - 1 # Puntos interiores (j=1, ..., N_x-1)
    DX = L / NX     # Paso espacial (Delta x)
    
    # Definición de las mallas (CORREGIDO EL ORDEN)
    x_full = np.linspace(0, L, NX + 1) # Malla completa (j=0, ..., N_x)
    x = x_full[1:-1]  # Malla interior (j=1, ..., N_x-1)

    # Parámetros del dominio temporal
    DT = 5e-3
    T_FINAL = 150.0  # Tiempo para ver oscilaciones
    N_STEPS = int(T_FINAL / DT)

    # --- Parámetros del Pozo Finito ---
    V0 = 2.0           # Profundidad del pozo (V(x) = -V0)
    pozo_ancho = 40.0
    pozo_centro = 90.0 # L/2
    pozo_x_L = pozo_centro - pozo_ancho / 2.0 
    pozo_x_R = pozo_centro + pozo_ancho / 2.0 

    # --- Parámetros del Paquete Gaussiano ---
    X0 = 90.0       # Centro inicial (DENTRO del pozo)
    K0 = 1.0        # Momento inicial (para que viaje)
    SIGMA0 = 5.0    # Anchura inicial (sigma_x(0))
    VG = HBAR * K0 / M # Velocidad de grupo teórica
    
    # Tiempos para snapshots (para ver la oscilación)
    snapshot_times = [0.0, 20.0, 40.0, 80.0] 
    snapshot_steps = [int(t / DT) for t in snapshot_times]
    snapshots_psi = []

    print(f"Dominio: L={L}, Nx={NX}, dx={DX:.4f}")
    print(f"Tiempo: T={T_FINAL}, Nt={N_STEPS}, dt={DT}")
    print(f"Pozo: V0={V0} en x=[{pozo_x_L}, {pozo_x_R}]")
    print(f"Paquete: x0={X0}, k0={K0}, sigma0={SIGMA0}")

    # --- 3. Hamiltoniano Discreto ---
    
    # 1. Vector de Potencial V(x)
    V_potential = np.zeros(N_INTERIOR)
    # (Usamos x, el vector de puntos interiores)
    pozo_indices = np.where((x >= pozo_x_L) & (x <= pozo_x_R))
    V_potential[pozo_indices] = -V0
    
    # 2. Término Cinético (T)
    diag_T = (HBAR**2 / (M * DX**2)) * np.ones(N_INTERIOR)
    off_diag_T = (-HBAR**2 / (2 * M * DX**2)) * np.ones(N_INTERIOR - 1)

    # 3. Hamiltoniano (H = T + V)
    H_sparse = scipy.sparse.diags(
        [off_diag_T, diag_T + V_potential, off_diag_T],
        [-1, 0, 1],
        shape=(N_INTERIOR, N_INTERIOR),
        format='csc'
    )
    print("Hamiltoniano H_D (con pozo finito) construido.")

    # --- 4. Condición Inicial ---
    psi_0_full = gaussian_wavepacket(x_full, X0, SIGMA0, K0, HBAR)
    psi_0_full[0] = 0.0
    psi_0_full[-1] = 0.0
    psi_0 = psi_0_full[1:-1]

    norm_inicial = np.sum(np.abs(psi_0)**2) * DX
    psi_0 = psi_0 / np.sqrt(norm_inicial)
    norma_t0, p_pozo_t0 = get_observables(psi_0, DX, x, pozo_indices)
    print(f"Condición inicial normalizada (Norma = {norma_t0:.6f})")

    # --- 5. Matrices CN ---
    I = scipy.sparse.eye(N_INTERIOR, dtype=complex, format='csc')
    A_cn = I + 1j * (DT / (2.0 * HBAR)) * H_sparse
    B_cn = I - 1j * (DT / (2.0 * HBAR)) * H_sparse

    print("Factorizando matriz CN...")
    solve_cn = splu(A_cn)
    print("Factorización completada.")

    # --- 6. Simulación ---
    print("Iniciando bucle CN (Pozo Finito)...")
    start_time = time.time()

    psi_cn = psi_0.copy()
    norm_data = np.zeros(N_STEPS + 1)
    p_pozo_data = np.zeros(N_STEPS + 1)
    
    norm_data[0] = norma_t0
    p_pozo_data[0] = p_pozo_t0
    
    time_axis = np.linspace(0, T_FINAL, N_STEPS + 1)

    if snapshot_steps[0] == 0:
        snapshots_psi.append(psi_0.copy())
        snapshot_steps.pop(0)

    for n in range(N_STEPS):
        rhs_cn = B_cn.dot(psi_cn)
        psi_cn = solve_cn.solve(rhs_cn)
        
        norm_t, p_pozo_t = get_observables(psi_cn, DX, x, pozo_indices)
        norm_data[n+1] = norm_t
        p_pozo_data[n+1] = p_pozo_t

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

    # --- Gráfica 1: Snapshots de la Oscilación ---
    fig_path = os.path.join(OUTPUT_DIR, 'TFG_Figura_5_3_Snapshots.png')
    fig, axs = plt.subplots(len(snapshot_times), 1, figsize=(8, 10), sharex=True)

    # Encontrar la altura máxima inicial para fijar el eje Y
    rho_inicial_full = np.abs(np.concatenate(([0+0j], snapshots_psi[0], [0+0j])))**2
    ylim_max = np.max(rho_inicial_full) * 1.1 # Usar la altura inicial + 10%

    for i, t in enumerate(snapshot_times):
        rho_full = np.abs(np.concatenate(([0+0j], snapshots_psi[i], [0+0j])))**2
        
        # 1. DIBUJAR EL POZO (fondo gris)
        axs[i].fill_between(x_full, 0, ylim_max, 
                            where=(x_full >= pozo_x_L) & (x_full <= pozo_x_R), 
                            color='gray', alpha=0.2, label='Pozo ($V_0$)')
        
        # 2. DIBUJAR LA PARTÍCULA (encima)
        axs[i].plot(x_full, rho_full, linewidth=2, color=f'C{i}')
        
        axs[i].set_title(f'$t = {int(t)}$', fontsize=10) # <--- CAMBIADO
        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=10) # <--- CAMBIADO
        axs[i].grid(True, linestyle=':', alpha=0.7)
        axs[i].set_ylim(0.0, ylim_max) # Eje Y fijo
        axs[i].set_xlim(0, L)

    axs[-1].set_xlabel('Posición $x$', fontsize=12) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path)
    print(f"Figura '{fig_path}' guardada.")

    # --- Gráfica 2: Fuga de Probabilidad (Probabilidad Parcial) ---
    fig_path = os.path.join(OUTPUT_DIR, 'TFG_Figura_5_3_FugaProbabilidad.png')
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, p_pozo_data, label='Probabilidad en pozo ($P_{pozo}$)', linewidth=2)
    
    plt.title('Probabilidad dentro del pozo finito', fontsize=14)
    plt.xlabel('Tiempo t', fontsize=12) 
    plt.ylabel('Probabilidad en pozo $P_{pozo}(t)$', fontsize=12)
    
    # Zoom a la "zona de acción" para ver la fuga y los temblores
    min_prob = np.min(p_pozo_data)
    plt.ylim(min_prob * 0.98, 1.001) 
    
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Figura '{fig_path}' guardada.")

    # --- Gráfica 3: Conservación de la Norma TOTAL ---
    fig_path = os.path.join(OUTPUT_DIR, 'TFG_Figura_5_3_Norma.png')
    plt.figure(figsize=(8, 5))
    ax = plt.gca() # Obtener el eje actual
    ax.plot(time_axis, norm_data, label='Norma Total (Crank-Nicolson)', linewidth=2)
    plt.title('Conservación de la norma total en el pozo finito', fontsize=14)
    plt.xlabel('Tiempo', fontsize=12) 
    plt.ylabel('Norma total $\int |\Psi|^2 dx$', fontsize=12)
    
    # --- CAMBIO SOLICITADO ---
    # Centrar el 1.0 en el eje Y y evitar el "offset"
    ax.set_ylim(0.9, 1.1) 
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # --- FIN DEL CAMBIO ---
    
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Figura '{fig_path}' guardada.")

    print("\n¡Proceso completado!")