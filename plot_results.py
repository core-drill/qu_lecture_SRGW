import numpy as np
from matplotlib import pyplot as plt
import json
import os
from scipy.special import ellipk

l_ring = 26700
def plot_rotational_symmetry_results(output_folder, output_path):

    storage_comments = ["0_degree", "90_degree", "180_degree", "270_degree"]
    colors = ['m', 'r', 'g', 'black']
    line_styles = ['--', '-', '-.', ':']
    
    plt.figure(figsize=(10, 6))
    
    for i in range(len(storage_comments)):
        comment = storage_comments[i]
        color = colors[i]
        line_style = line_styles[i]
        data = np.loadtxt(f"{output_folder}_{comment}/gravitational_force_distribution.csv", delimiter=',', skiprows=1)
        s_cells = data[:, 0]
        f_parallel = data[:, 1]
        
        plt.plot(s_cells, f_parallel, label=f'{comment}', color=color, linestyle=line_style)
    
    plt.xlabel('s (m)')
    plt.ylabel('Gravitational Force F_parallel (N)')
    plt.title('Gravitational Force Distribution for Different Positions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "rotational_symmetry_F_comparison.png"), dpi=300)
    plt.close()

    data_time = np.loadtxt(f"{output_folder}_{storage_comments[0]}/object_trajectory.csv", delimiter=',', skiprows=1)
    time = data_time[:, 0]
    for i in range(len(storage_comments)):
        comment = storage_comments[i]
        color = colors[i]
        line_style = line_styles[i]
        data = np.loadtxt(f"{output_folder}_{comment}/uranium_plus_FODO_6D_history.csv", delimiter=',', skiprows=1)
        delta_l = data[:, 4]
        plt.plot(time, delta_l, label=f'{comment}', color=color, linestyle=line_style)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement Δl (m)')
    plt.title('Object Trajectory Displacement for Different Positions')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "rotational_symmetry_trajectory_comparison.png"), dpi=300)
    plt.close()

    r = l_ring / (2 * np.pi)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    plt.figure(figsize=(8, 8))
    plt.plot(x_circle, y_circle, 'k--', label='Storage Ring')
    positions = [
        (5e3, 0),
        (0, 5e3),
        (-5e3, 0),
        (0, -5e3)
    ]
    for i in range(len(storage_comments)):
        x_pos, y_pos = positions[i]
        plt.scatter(x_pos, y_pos, label=f'Position {storage_comments[i]}')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Object Positions in Storage Ring')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "rotational_symmetry_positions.png"), dpi=300)
    plt.close()


def plot_y_symmetry_results(output_folder, output_path):
    storage_comments = ["plus_vy", "minus_vy"]
    colors = ['b', 'r']
    line_styles = ['-', ':']
    
    plt.figure(figsize=(10, 6))
    
    for i in range(len(storage_comments)):
        comment = storage_comments[i]
        color = colors[i]
        line_style = line_styles[i]
        data = np.loadtxt(f"{output_folder}_{comment}/gravitational_force_distribution.csv", delimiter=',', skiprows=1)
        s_cells = data[:, 0]
        f_parallel = data[:, 1]
        
        plt.plot(s_cells, f_parallel, label=f'{comment}', color=color, linestyle=line_style)
    
    plt.xlabel('s (m)')
    plt.ylabel('Gravitational Force F_parallel (N)')
    plt.title('Gravitational Force Distribution for Different Velocities')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "y_symmetry_F_comparison.png"), dpi=300)
    plt.close()


    data_time = np.loadtxt(f"{output_folder}_{storage_comments[0]}/object_trajectory.csv", delimiter=',', skiprows=1)
    time = data_time[:, 0]
    for i in range(len(storage_comments)):
        comment = storage_comments[i]
        color = colors[i]
        line_style = line_styles[i]
        data = np.loadtxt(f"{output_folder}_{comment}/uranium_plus_FODO_6D_history.csv", delimiter=',', skiprows=1)
        delta_l = data[:, 4]
        plt.plot(time, delta_l, label=f'{comment}', color=color, linestyle=line_style)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement Δl (m)')
    plt.title('Object Trajectory Displacement for Different Velocities')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "y_symmetry_trajectory_comparison.png"), dpi=300)
    plt.close()

    r = l_ring / (2 * np.pi)
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)
    
    plus_vy_t_pos = np.loadtxt(f"{output_folder}_{storage_comments[0]}/object_trajectory.csv", delimiter=',', skiprows=1)
    minus_vy_t_pos = np.loadtxt(f"{output_folder}_{storage_comments[1]}/object_trajectory.csv", delimiter=',', skiprows=1)
    plus_vy_x = plus_vy_t_pos[:, 1]
    plus_vy_y = plus_vy_t_pos[:, 2]
    minus_vy_x = minus_vy_t_pos[:, 1]
    minus_vy_y = minus_vy_t_pos[:, 2]
    plt.figure(figsize=(8, 8))
    plt.plot(x_circle, y_circle, 'k--', label='Storage Ring')
    plt.plot(plus_vy_x, plus_vy_y, label='Trajectory plus_vy', color='b')
    plt.plot(minus_vy_x, minus_vy_y, label='Trajectory minus_vy', color='r')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Object Trajectories for y-symmetry Velocities')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, "y_symmetry_trajectories.png"), dpi=300)
    plt.close()


def delta_T_ana_linear_1e10round(x0, y0):
    k_round = 1e4
    config = json.load(open('FODO_config.json'))
    scan_config = json.load(open('scan_config.json'))

    beta = config['beta']
    gamma = 1/np.sqrt(1-beta**2)
    c = 299792458.0  # m/s
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M = 1e10  # kg 
    l_ring = 26700.0
    R = l_ring/2/np.pi
    H = 0
    x = x0 
    y = y0 
    z = 10000
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    m =-(4*R*r*np.sin(theta)/(r**2+R**2-2*r*R*np.sin(theta)))
    K = ellipk(m)
    delta_T_round = -(R*G*M/(beta*c)**3)*((2*np.pi/(np.sqrt(R**2+r**2-2*R*r*np.cos(phi)*np.sin(theta)))) - (4*K/np.sqrt(r**2+R**2-2*r*R*np.sin(theta))) )

    delta_T_ana_total = delta_T_round * k_round

    return delta_T_ana_total

def delta_T_vs_xy_1e10round(x_range, y_range):
    delta_T_matrix = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            delta_T_matrix[i, j] = delta_T_ana_linear_1e10round(x, y)
    return delta_T_matrix

def plot_delta_T_heatmap(output_path):
    x_range = np.linspace(-2e4, 2e4, 100)
    y_range = np.linspace(-2e4, 2e4, 100)
    delta_T_matrix = delta_T_vs_xy_1e10round(x_range, y_range)
    from matplotlib.colors import TwoSlopeNorm

    norm = TwoSlopeNorm(
    vmin=delta_T_matrix.min(),   
    vcenter=0.0,       
    vmax=delta_T_matrix.max()
)

    plt.figure(figsize=(8, 6))
    plt.imshow(delta_T_matrix, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], origin='lower', cmap='coolwarm', norm=norm)
    cbar = plt.colorbar(label='ΔT (s)')
    ticks = np.linspace(delta_T_matrix.min(), delta_T_matrix.max(), 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('ΔT Heatmap for 1e30 kg Object (k = 1e4, z=10000 m)')
    plt.savefig(os.path.join(output_path, "delta_T_heatmap_xy2e4_z_10000.png"), dpi=300)
    plt.close()

def plot_delta_T_heatmap_abs_log(output_path):
    x_range = np.linspace(-1e5, 1e5, 100)
    y_range = np.linspace(-1e5, 1e5, 100)
    delta_T_matrix = delta_T_vs_xy_1e10round(x_range, y_range)
    abs_delta_T_matrix = np.abs(delta_T_matrix)
    log_delta_T_matrix = np.log10(abs_delta_T_matrix + 1e-20)  # Adding a small value to avoid log(0)

    plt.figure(figsize=(8, 6))
    plt.imshow(log_delta_T_matrix, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], origin='lower', cmap='tab10')
    cbar = plt.colorbar(label='log10(|ΔT|) (s)')
    # ticks = np.linspace(np.min(log_delta_T_matrix), np.max(log_delta_T_matrix), 7)
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Logarithmic |ΔT| Heatmap for 1e10 kg Object (k = 1e4, z=10 m)')
    plt.savefig(os.path.join(output_path, "delta_T_heatmap_xy1e5_z_10_log.png"), dpi=300)
    plt.close()




def delta_T_ana_linear_1e10round_xz(x0, z0):
    k_round = 1e4
    config = json.load(open('FODO_config.json'))
    scan_config = json.load(open('scan_config.json'))

    beta = config['beta']
    gamma = 1/np.sqrt(1-beta**2)
    c = 299792458.0  # m/s
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    M = 1e30  # kg 
    l_ring = 26700.0
    R = l_ring/2/np.pi
    H = 0
    x = x0 
    y = 0 
    z = z0
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    m =-(4*R*r*np.sin(theta)/(r**2+R**2-2*r*R*np.sin(theta)))
    K = ellipk(m)
    delta_T_round = -(R*G*M/(beta*c)**3)*((2*np.pi/(np.sqrt(R**2+r**2-2*R*r*np.cos(phi)*np.sin(theta)))) - (4*K/np.sqrt(r**2+R**2-2*r*R*np.sin(theta))) )

    delta_T_ana_total = delta_T_round * k_round

    return delta_T_ana_total

def delta_T_vs_xz_1e10round(x_range, z_range):
    delta_T_matrix = np.zeros((len(z_range), len(x_range)))
    for i, z in enumerate(z_range):
        for j, x in enumerate(x_range):
            delta_T_matrix[i, j] = delta_T_ana_linear_1e10round_xz(x, z)
    return delta_T_matrix

def plot_delta_T_heatmap_xz(output_path):
    x_range = np.linspace(-1e4, 1e4, 100)
    z_range = np.linspace(-1e4, 1e4, 100)
    delta_T_matrix = delta_T_vs_xz_1e10round(x_range, z_range)
    from matplotlib.colors import TwoSlopeNorm

    norm = TwoSlopeNorm(
    vmin=delta_T_matrix.min(),   
    vcenter=0.0,       
    vmax=delta_T_matrix.max()
)

    plt.figure(figsize=(8, 6))
    plt.imshow(delta_T_matrix, extent=[x_range[0], x_range[-1], z_range[0], z_range[-1]], origin='lower', cmap='seismic', norm=norm)
    cbar = plt.colorbar(label='ΔT (s)')
    ticks = np.linspace(delta_T_matrix.min(), delta_T_matrix.max(), 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.title('ΔT Heatmap for 1e10 kg Object (k = 1e4, y=0 m)')
    plt.savefig(os.path.join(output_path, "delta_T_heatmap_xz1e4_y_0.png"), dpi=300)
    plt.close()


    
if __name__ == "__main__":
    output_path_rotational_symmetry = 'output_results/rotational_symmetry/'
    if not os.path.exists(output_path_rotational_symmetry):
        os.makedirs(output_path_rotational_symmetry)
    plot_rotational_symmetry_results('D:\\study\\qu_seminar\\code\\qu_lecture_SRGW\\output\\rotational_symmetry\\1.000e+09kg_stationary',output_path_rotational_symmetry)

    output_path_y_symmetry = 'output_results/y_symmetry/'
    if not os.path.exists(output_path_y_symmetry):
        os.makedirs(output_path_y_symmetry)
    plot_y_symmetry_results('D:\\study\\qu_seminar\\code\\qu_lecture_SRGW\\output\\y_symmetry\\1.000e+09kg_linear',output_path_y_symmetry)

    output_path_delta_T = 'output_results/delta_T_heatmap1e30/'
    if not os.path.exists(output_path_delta_T):
        os.makedirs(output_path_delta_T)
    plot_delta_T_heatmap(output_path_delta_T)
    # plot_delta_T_heatmap_xz(output_path_delta_T)
    # plot_delta_T_heatmap_abs_log(output_path_delta_T)