import numpy as np
from matplotlib import pyplot as plt
import iminuit
from iminuit import minimize
from iminuit import Minuit, describe
from iminuit.cost import LeastSquares
import json
from scipy.special import ellipk
import time


config = json.load(open('FODO_config.json'))
scan_config = json.load(open('scan_config.json'))

beta = config['beta']
gamma = 1/np.sqrt(1-beta**2)
print(gamma)
c = 299792458.0  # m/s
G = 6.67430e-11  # m^3 kg^-1 s^-2
M = scan_config['object_mass_kg']  # kg 
l_ring = 26700.0
R = l_ring/2/np.pi
H = 0
times = config['times']
k_round = scan_config["simulation_parameters"]['change_per_times']
T_round = l_ring / (beta * c)  # s
t = [i * k_round * T_round for i in range(int(times/k_round))]


data = np.loadtxt('output/uranium_plus_FODO_6D_history.csv', delimiter=',', skiprows=1)
data_l = data[:, 4]

plt.figure(figsize=(8, 6))
plt.plot(t, data_l)
plt.xlabel('Time (s)')
plt.ylabel("delta L (m)")
plt.title("heavyobject effect to particle's longitudinal displacement with time")
plt.grid()
plt.savefig('output/uranium_plus_l_vs_t.png', dpi = 400)
plt.close()



slip_factor = 0.89718 
def delta_T_ana_linear(x0, y0, z0, vx, vy, vz, t):
    t = np.array(t)
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    m =-(4*R*r*np.sin(theta)/(r**2+R**2-2*r*R*np.sin(theta)))
    K = ellipk(m)
    delta_T_round = -(R*G*M/(beta*c)**3)*((2*np.pi/(np.sqrt(R**2+r**2-2*R*r*np.cos(phi)*np.sin(theta)))) - (4*K/np.sqrt(r**2+R**2-2*r*R*np.sin(theta))) )
    delta_T_ana_k_round = delta_T_round * k_round

    delta_T_ana_total = np.cumsum(delta_T_ana_k_round)

    return delta_T_ana_total

x0_set = scan_config['linear_motion']['initial_position'][0]
y0_set = scan_config['linear_motion']['initial_position'][1]
z0_set = scan_config['linear_motion']['initial_position'][2]
vx_set = scan_config['linear_motion']['velocity_vector'][0]
vy_set = scan_config['linear_motion']['velocity_vector'][1]
vz_set = scan_config['linear_motion']['velocity_vector'][2]
T_ana = delta_T_ana_linear(x0_set, y0_set, z0_set, vx_set, vy_set, vz_set, t)
T_ana = T_ana 
delta_T_num =  data_l / (beta * c) / (slip_factor * gamma**-1) 

plt.figure(figsize=(8, 6))
plt.plot(t, T_ana, label=r'Analytical Result: $\Delta T_{ana}$', color='orange')
plt.plot(t, delta_T_num, label=r'Numerical Result: $\gamma*\Delta T_{num}/\eta, \Delta T_{num} = x_5/\beta c$', linestyle='dashed', color='blue')
plt.xlabel('Time (s)')
plt.ylabel("delta T (s)")
plt.title("Comparison of Analytical and Numerical Results")
plt.legend()
plt.grid()
plt.savefig('output/uranium_plus_l_vs_t_comparison.png', dpi = 400)
plt.close()







fit_factor = 1e25
def delta_T_ana_x(t, x0, vx):
    t = np.array(t)
    x = x0 + vx * t
    y = y0_set + vy_set * t
    z = z0_set + vz_set * t
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    m =-(4*R*r*np.sin(theta)/(r**2+R**2-2*r*R*np.sin(theta)))
    K = ellipk(m)
    delta_T_round = -(R*G*M/(beta*c)**3)*((2*np.pi/(np.sqrt(R**2+r**2-2*R*r*np.cos(phi)*np.sin(theta)))) - (4*K/np.sqrt(r**2+R**2-2*r*R*np.sin(theta))) )
    delta_T_ana_k_round = delta_T_round * k_round

    delta_T_ana_total = np.cumsum(delta_T_ana_k_round)

    return delta_T_ana_total * fit_factor


least_squares = LeastSquares(t, delta_T_num*fit_factor, 1, delta_T_ana_x)
x0_fit=x0_set*1.1
vx_fit=vx_set*1.1
m = Minuit(least_squares, x0=x0_fit, vx=vx_fit)
m.migrad()
m.hesse()
print(m.values)
T_ana_fit = delta_T_ana_x(t, m.values['x0'], m.values['vx'])
plt.figure(figsize=(8, 6))
plt.scatter(t, T_ana_fit/fit_factor, label='Fitted Result', s=10, color='black')
plt.plot(t, delta_T_num, label='Numerical Result', linestyle='dashed', color='blue')
plt.xlabel('Time (s)')
plt.ylabel("delta T (s)")
plt.title("Fitted Analytical vs Numerical Results")
plt.legend()
plt.text(0.6*max(t), (min(delta_T_num)+max(delta_T_num))/2, f"fit results:\n"
         f"x0={m.values['x0']:.2f} m\n"
            f"vx={m.values['vx']:.2f} m/s")
plt.grid()
plt.savefig('output/x_fit.png', dpi = 400)
plt.close()


def delta_T_ana_xy3(t, x0, y0, vx):
    t = np.array(t)
    x = x0 + vx * t
    y = y0 + vy_set * t
    z = z0_set + vz_set * t
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    m =-(4*R*r*np.sin(theta)/(r**2+R**2-2*r*R*np.sin(theta)))
    K = ellipk(m)
    delta_T_round = -(R*G*M/(beta*c)**3)*((2*np.pi/(np.sqrt(R**2+r**2-2*R*r*np.cos(phi)*np.sin(theta)))) - (4*K/np.sqrt(r**2+R**2-2*r*R*np.sin(theta))) )
    delta_T_ana_k_round = delta_T_round * k_round

    delta_T_ana_total = np.cumsum(delta_T_ana_k_round)

    return delta_T_ana_total * fit_factor

def seg_sum_xy3(x0, y0, vx):
    T_ana_fit = delta_T_ana_xy3(t, x0, y0, vx)
    residuals = (T_ana_fit - delta_T_num * fit_factor)
    seg1 = np.sum(residuals[0:70]**2)/70
    seg2 = np.sum(residuals[70:140]**2)/70
    seg3 = np.sum(residuals[140:]**2)/60
    return seg1 + seg2 + seg3

x0_fit=x0_set*1.1
y0_fit=y0_set*1.1
vx_fit=vx_set*1.1
m = Minuit(seg_sum_xy3, x0=x0_fit, y0=y0_fit, vx=vx_fit)
m.migrad()
m.hesse()
print(m.values)
T_ana_fit = delta_T_ana_xy3(t, m.values['x0'], m.values['y0'], m.values['vx'])
plt.figure(figsize=(8, 6))
plt.scatter(t, T_ana_fit/fit_factor, label='Fitted Result', s=10, color='black')
plt.plot(t, delta_T_num, label='Numerical Result', linestyle='dashed', color='blue')
plt.xlabel('Time (s)')
plt.ylabel("delta T (s)")
plt.title("Fitted Analytical vs Numerical Results")
plt.legend()
plt.text(0.6*max(t), (min(delta_T_num)+max(delta_T_num))/2, f"fit results:\n"
         f"x0={m.values['x0']:.2f} m\n"
            f"y0={m.values['y0']:.2f} m\n"
            f"vx={m.values['vx']:.2f} m/s")
plt.grid()
plt.savefig('output/xy3_fit.png', dpi = 400)
plt.close()
    

def delta_T_ana_xy4(t, x0, y0, z0, vx):
    t = np.array(t)
    x = x0 + vx * t
    y = y0 + vy_set * t
    z = z0 + vz_set * t
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    m =-(4*R*r*np.sin(theta)/(r**2+R**2-2*r*R*np.sin(theta)))
    K = ellipk(m)
    delta_T_round = -(R*G*M/(beta*c)**3)*((2*np.pi/(np.sqrt(R**2+r**2-2*R*r*np.cos(phi)*np.sin(theta)))) - (4*K/np.sqrt(r**2+R**2-2*r*R*np.sin(theta))) )
    delta_T_ana_k_round = delta_T_round * k_round

    delta_T_ana_total = np.cumsum(delta_T_ana_k_round)

    return delta_T_ana_total * fit_factor

def seg_sum_xy4(x0, y0, z0, vx):
    T_ana_fit = delta_T_ana_xy4(t, x0, y0, z0, vx)
    residuals = (T_ana_fit - delta_T_num * fit_factor)
    seg1 = np.sum(residuals[0:50]**2)/50
    seg2 = np.sum(residuals[50:100]**2)/50
    seg3 = np.sum(residuals[100:150]**2)/50
    seg4 = np.sum(residuals[150:]**2)/50
    return seg1 + seg2 + seg3 + seg4

x0_fit=x0_set*1.1
y0_fit=y0_set*1.1
z0_fit=z0_set*1.1
vx_fit=vx_set*1.1
m = Minuit(seg_sum_xy4, x0=x0_fit, y0=y0_fit, z0=z0_fit, vx=vx_fit)
m.migrad()
m.hesse()
print(m.values)
T_ana_fit = delta_T_ana_xy4(t, m.values['x0'], m.values['y0'], m.values['z0'], m.values['vx'])
plt.figure(figsize=(8, 6))
plt.scatter(t, T_ana_fit/fit_factor, label='Fitted Result', s=10, color='black')
plt.plot(t, delta_T_num, label='Numerical Result', linestyle='dashed', color='blue')
plt.xlabel('Time (s)')
plt.ylabel("delta T (s)")
plt.title("Fitted Analytical vs Numerical Results")
plt.legend()
plt.text(0.6*max(t), (min(delta_T_num)+max(delta_T_num))/2, f"fit results:\n"
         f"x0={m.values['x0']:.2f} m\n"
            f"y0={m.values['y0']:.2f} m\n"
            f"z0={m.values['z0']:.2f} m\n"
            f"vx={m.values['vx']:.2f} m/s")
plt.grid()
plt.savefig('output/xyz4_fit.png', dpi = 400)
plt.close()


## xy3 fit from julia
# x0_julia = 82.07785643027873
# y0_julia = -445.7176632846839
# vx_julia = -11.424423788999508
# T_ana_fit = delta_T_ana_xy3(t, x0_julia, y0_julia, vx_julia)
# plt.figure(figsize=(8, 6))
# plt.scatter(t, T_ana_fit/fit_factor, label='Fitted Result', s=10, color='black')
# plt.plot(t, delta_T_num, label='Numerical Result', linestyle='dashed', color='blue')
# plt.xlabel('Time (s)')
# plt.ylabel("delta T (s)")
# plt.title("Fitted Analytical vs Numerical Results (from Julia)")
# plt.legend()
# plt.text(0.6*max(t), (min(delta_T_num)+max(delta_T_num))/2, f"fit results:\n"
#          f"x0={x0_julia:.2f} m\n"
#             f"y0={y0_julia:.2f} m\n"
#             f"vx={vx_julia:.2f} m/s")
# plt.grid()
# plt.savefig('output/xy3_julia_fit.png', dpi = 400)
# plt.close()

    