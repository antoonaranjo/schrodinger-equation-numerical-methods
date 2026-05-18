# -*- coding: utf-8 -*-
"""
TFG - Capítulo 5.4: Barrera Rectangular de Potencial (Versión 7.0 - k0=2.0 Unificado y Corregido)

Simulación de un paquete de ondas Gaussiano incidiendo sobre una barrera,
cubriendo los dos regímenes físicos:
1. Dispersión sobre la barrera (E > V0)
2. Efecto Túnel (E < V0)

Ambos escenarios usan k0=2.0 para una velocidad constante y se ajustan las V0
para asegurar los regímenes E > V0 y E < V0.
"""

# --- 0. Importaciones ---
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import splu
import time
import os
from matplotlib.ticker import FormatStrFormatter

# --- 1. Funciones Auxiliares ---

def gaussian_wavepacket(x_nodes, x0, sigma0, k0, hbar=1.0):
    """Crea un paquete de ondas Gaussiano viajero (Eq. 3.34 / 3.35)."""
    a = 1.0 / (4.0 * sigma0**2)
    norm_factor = (2.0 * a / np.pi)**0.25
    spatial_envelope = np.exp(-(x_nodes - x0)**2 / (4.0 * sigma0**2))
    momentum_phase = np.exp(1j * k0 * x_nodes)
    return norm_factor * spatial_envelope * momentum_phase

def get_observables(psi_interior, dx, x_interior, barrera_x_L, barrera_x_R):
    """
    Calcula la norma total, y las probabilidades parciales de
    Reflexión (izquierda) y Transmisión (derecha).
    """
    rho = np.abs(psi_interior)**2
    norm_total = np.sum(rho) * dx
    
    # Índices para las regiones de la izquierda y derecha
    indices_izq = np.where(x_interior < barrera_x_L)
    indices_der = np.where(x_interior > barrera_x_R)
    
    # Calcular probabilidades parciales
    prob_izquierda = np.sum(rho[indices_izq]) * dx
    prob_derecha = np.sum(rho[indices_der]) * dx
    
    # La probabilidad restante (1 - R - T) está en la barrera.
    
    return norm_total, prob_izquierda, prob_derecha

def create_hamiltonian(N_INTERIOR, DX, V_potential, HBAR=1.0, M=1.0):
    """Construye la matriz Hamiltoniana dispersa (T + V)."""
    
    # Término Cinético (T)
    diag_T = (HBAR**2 / (M * DX**2)) * np.ones(N_INTERIOR)
    off_diag_T = (-HBAR**2 / (2 * M * DX**2)) * np.ones(N_INTERIOR - 1)

    # Hamiltoniano (T + V)
    H_sparse = scipy.sparse.diags(
        [off_diag_T, diag_T + V_potential, off_diag_T],
        [-1, 0, 1],
        shape=(N_INTERIOR, N_INTERIOR),
        format='csc'
    )
    return H_sparse

def generate_plots(output_dir, scenario_name, time_axis, norm_data, p_izq_data, p_der_data, 
                   snapshots_psi, snapshot_times, x_full, ylim_max, 
                   barrera_x_L, barrera_x_R, L, T_FINAL):
    """
    Genera y guarda el conjunto de 3 gráficas para una simulación.
    """
    # Texto de escenario con notación LaTeX: $E<V_0$ o $E>V_0$
    titulo_math = scenario_name.replace("E_menor_V0", r'$E<V_0$') \
                               .replace("E_mayor_V0", r'$E>V_0$')
    
    # --- Gráfica 1: Snapshots de la Evolución ---
    fig_path = os.path.join(output_dir, f'TFG_Figura_5_4_{scenario_name}_Snapshots.png')
    fig, axs = plt.subplots(len(snapshot_times), 1, figsize=(9, 11), sharex=True) 


    # El ylim_max es ahora el 70% de la altura máxima para mejor visualización
    initial_peak = np.max(np.abs(snapshots_psi[0])**2)
    ylim_snapshot = initial_peak / 0.7 

    for i, t in enumerate(snapshot_times):
        rho_full = np.abs(np.concatenate(([0+0j], snapshots_psi[i], [0+0j])))**2
        
        # 1. DIBUJAR LA BARRERA (fondo gris)
        axs[i].fill_between(x_full, 0, ylim_snapshot, 
                            where=(x_full >= barrera_x_L) & (x_full <= barrera_x_R), 
                            color='gray', alpha=0.3, label='Barrera ($V_0$)')
        
        # 2. DIBUJAR LA PARTÍCULA (encima)
        axs[i].plot(x_full, rho_full, linewidth=2, color=f'C{i}')
        
        axs[i].set_title(f'$t = {t:.0f}$', fontsize=10)
        axs[i].set_ylabel(r'$|\Psi(x,t)|^2$', fontsize=10)
        axs[i].grid(True, linestyle=':', alpha=0.7)
        axs[i].set_ylim(0.0, ylim_snapshot) 
        axs[i].set_xlim(0, L)

    axs[-1].set_xlabel('Posición $x$', fontsize=12) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_path)
    print(f"  > Figura '{fig_path}' guardada.")

    # --- Gráfica 2: Coeficientes R y T ---
    fig_path = os.path.join(output_dir, f'TFG_Figura_5_4_{scenario_name}_Probabilidad.png')
    plt.figure(figsize=(8, 5))
    
    # R (Reflejada) y T (Transmitida)
    plt.plot(time_axis, p_izq_data, label='Prob. Reflejada $R(t)$', linewidth=2, linestyle='--')
    plt.plot(time_axis, p_der_data, label='Prob. Transmitida $T(t)$', linewidth=2, linestyle=':')
    
    # Probabilidad en la Barrera 
    P_fuera = p_izq_data + p_der_data
    P_en_barrera_temporal = norm_data - P_fuera
    P_en_barrera_temporal[P_en_barrera_temporal < 0] = 0 # Limpiar ruido de precisión de máquina
    
    plt.plot(time_axis, P_en_barrera_temporal, label='Prob. en Barrera', linewidth=1, color='red', alpha=0.6)
    
    plt.title(f'Coeficientes de reflexión y transmisión ({titulo_math})', fontsize=14)
    plt.xlabel('Tiempo t', fontsize=12) 
    plt.ylabel('Probabilidad parcial', fontsize=12)
    plt.ylim(0.0, 1.0) 
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"  > Figura '{fig_path}' guardada.")

    # --- Gráfica 3: Conservación de la Norma TOTAL ---
    fig_path = os.path.join(output_dir, f'TFG_Figura_5_4_{scenario_name}_Norma.png')
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(time_axis, norm_data, label='Norma total (Crank–Nicolson)', linewidth=2)
    plt.title(f'Conservación de la norma total ({titulo_math})', fontsize=14)
    plt.xlabel('Tiempo t', fontsize=12) 
    plt.ylabel(r'Norma total $\int |\Psi|^2 dx$', fontsize=12)
    
    # Zoom centrado en 1.0
    ax.set_ylim(0.999, 1.001) 
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"  > Figura '{fig_path}' guardada.")


def run_simulation(k0, v0, scenario_name, L, NX, DT, T_FINAL, X0, SIGMA0, HBAR, M, OUTPUT_DIR):
    """
    Ejecuta una simulación completa de CN para un k0 y V0 dados.
    """
    print(f"\n--- Iniciando Escenario: {scenario_name} ---")
    
    # Cálculo de energía media (solo para el print de información)
    E_cin_avg = (k0**2 / 2.0) + (1.0 / (8.0 * SIGMA0**2))
    
    print(f"(k0={k0}, V0={v0}) -> E_avg={E_cin_avg:.3f}")

    # --- Configuración ---
    N_INTERIOR = NX - 1
    DX = L / NX
    x_full = np.linspace(0, L, NX + 1)
    x = x_full[1:-1]
    N_STEPS = int(T_FINAL / DT)
    
    # Parámetros de la Barrera
    barrera_ancho = 20.0
    barrera_centro = 90.0
    barrera_x_L = barrera_centro - barrera_ancho / 2.0 # x=80
    barrera_x_R = barrera_centro + barrera_ancho / 2.0 # x=100
    
    # Tiempos para snapshots
    snapshot_times = [0.0, 20.0, 30.0, 40.0] 
    T_FINAL_ACTUAL = 60.0
    
    snapshot_steps = [int(t / DT) for t in snapshot_times]
    snapshots_psi = []
    
    # Ajustamos el número de pasos a T_FINAL_ACTUAL
    N_STEPS_ACTUAL = int(T_FINAL_ACTUAL / DT)

    # --- Hamiltoniano ---
    V_potential = np.zeros(N_INTERIOR)
    barrera_indices = np.where((x >= barrera_x_L) & (x <= barrera_x_R))
    V_potential[barrera_indices] = v0
    
    H_sparse = create_hamiltonian(N_INTERIOR, DX, V_potential, HBAR, M)

    # --- Condición Inicial ---
    psi_0_full = gaussian_wavepacket(x_full, X0, SIGMA0, k0, HBAR)
    psi_0_full[0] = 0.0
    psi_0_full[-1] = 0.0
    psi_0 = psi_0_full[1:-1]
    
    norm_inicial = np.sum(np.abs(psi_0)**2) * DX
    psi_0 = psi_0 / np.sqrt(norm_inicial)
    
    norma_t0, p_izq_t0, p_der_t0 = get_observables(psi_0, DX, x, barrera_x_L, barrera_x_R)
    print(f"Condición inicial normalizada (Norma = {norma_t0:.6f})")

    # --- Matrices CN y Factorización ---
    I = scipy.sparse.eye(N_INTERIOR, dtype=complex, format='csc')
    A_cn = I + 1j * (DT / (2.0 * HBAR)) * H_sparse
    B_cn = I - 1j * (DT / (2.0 * HBAR)) * H_sparse
    solve_cn = splu(A_cn)
    print("Factorización de CN completada.")

    # --- Simulación ---
    start_time = time.time()
    psi_cn = psi_0.copy()
    
    norm_data = np.zeros(N_STEPS_ACTUAL + 1)
    p_izq_data = np.zeros(N_STEPS_ACTUAL + 1)
    p_der_data = np.zeros(N_STEPS_ACTUAL + 1)
    
    norm_data[0] = norma_t0
    p_izq_data[0] = p_izq_t0
    p_der_data[0] = p_der_t0
    
    time_axis = np.linspace(0, T_FINAL_ACTUAL, N_STEPS_ACTUAL + 1)

    if snapshot_steps[0] == 0:
        snapshots_psi.append(psi_0.copy())

    for n in range(N_STEPS_ACTUAL):
        rhs_cn = B_cn.dot(psi_cn)
        psi_cn = solve_cn.solve(rhs_cn)
        
        norm_t, p_izq, p_der = get_observables(psi_cn, DX, x, barrera_x_L, barrera_x_R)
        norm_data[n+1] = norm_t
        p_izq_data[n+1] = p_izq
        p_der_data[n+1] = p_der

        if (n + 1) in snapshot_steps:
            snapshots_psi.append(psi_cn.copy())

    print(f"Simulación '{scenario_name}' completada en {time.time() - start_time:.2f} s.")

    # --- NUEVO: cálculo e impresión de R+T+P_barrera al final ---
    R_final = p_izq_data[-1]
    T_final = p_der_data[-1]
    P_barrera_final = norm_data[-1] - (R_final + T_final)
    prob_total = R_final + T_final + P_barrera_final

    if scenario_name == "E_menor_V0":
        etiqueta_escenario = "E<V0"
    elif scenario_name == "E_mayor_V0":
        etiqueta_escenario = "E>V0"
    else:
        etiqueta_escenario = scenario_name

    print(f"Probabilidad total para {etiqueta_escenario}: R+T+P_barrera = {prob_total:.6f}")
    print(f"  -> R_final = {R_final:.6f}, T_final = {T_final:.6f}, P_barrera_final = {P_barrera_final:.6f}")

    # --- Gráficas ---
    print(f"Generando gráficas para '{scenario_name}'...")
    ylim_max = np.max(np.abs(psi_0)**2) * 1.1
    
    generate_plots(OUTPUT_DIR, scenario_name, time_axis, norm_data, p_izq_data, p_der_data, 
                   snapshots_psi, snapshot_times, x_full, ylim_max, 
                   barrera_x_L, barrera_x_R, L, T_FINAL_ACTUAL)

# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":

    OUTPUT_DIR = "5.4"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Simulaciones de la Barrera Rectangular (Cap. 5.4)")
    print(f"Los resultados se guardarán en: '{OUTPUT_DIR}/'")

    HBAR = 1.0
    M = 1.0

    L = 180.0
    NX = 3600
    DX = L / NX
    DT = 5e-3

    X0 = 45.0
    SIGMA0 = 5.0
    
    V0_ALTA = 2.2  # Efecto túnel (E<V0)
    V0_BAJA = 0.2  # Dispersión (E>V0)

    K0_UNIFICADO = 2.0
    
    # Escenario 1: E > V0
    run_simulation(
        k0=K0_UNIFICADO, 
        v0=V0_BAJA, 
        scenario_name="E_mayor_V0",
        L=L, NX=NX, DT=DT, T_FINAL=90.0, 
        X0=X0, SIGMA0=SIGMA0, 
        HBAR=HBAR, M=M, OUTPUT_DIR=OUTPUT_DIR
    )

    # Escenario 2: E < V0 (túnel)
    run_simulation(
        k0=K0_UNIFICADO, 
        v0=V0_ALTA, 
        scenario_name="E_menor_V0",
        L=L, NX=NX, DT=DT, T_FINAL=90.0, 
        X0=X0, SIGMA0=SIGMA0, 
        HBAR=HBAR, M=M, OUTPUT_DIR=OUTPUT_DIR
    )

    print("\n¡Proceso completado! Todas las 6 figuras han sido guardadas.")