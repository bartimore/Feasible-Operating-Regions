import pandapower.plotting as plot
import numpy as np
import matplotlib.pyplot as plt
import pandapower as pp
from grid_cuts import create_network_n
import pickle

l = 100 * 1.0e-3  # km
r_ohm_per_km = 0.14 / l
x_ohm_per_km = 0.04 / l
c_nf_per_km = 0.32 * 1.0e3
v_base = 230.94
v_min = 0.95 * v_base

def run_single_pf(net, ps, qs):
    # Set load
    net.load['p_mw'] = ps * 1.0e-6
    net.load['q_mvar'] = qs * 1.0e-6

    # Do power flow
    pp.runpp_3ph(net)


    # Get results
    busses = net.res_bus_3ph
    busses['vm_a_phn'] = busses['vm_a_pu'] * net.bus['vn_kv'] * 1000 / np.sqrt(3)
    busses['vm_b_phn'] = busses['vm_b_pu'] * net.bus['vn_kv'] * 1000 / np.sqrt(3)
    busses['vm_c_phn'] = busses['vm_c_pu'] * net.bus['vn_kv'] * 1000 / np.sqrt(3)


    lines = net.res_line_3ph[['i_a_from_ka', 'i_b_from_ka', 'i_c_from_ka', 'i_n_from_ka', 'loading_percent',
                              'p_a_from_mw', 'q_a_from_mvar', 'p_b_from_mw', 'q_b_from_mvar',
                              'p_c_from_mw', 'q_c_from_mvar']]


    results = {'v_bus_lv': busses['vm_a_phn'].iloc[0],
               'v_bus_load': busses['vm_a_phn'].iloc[-1],
               'p_line_1': 3 * lines['p_a_from_mw'].iloc[0] * 1.0e6,
               'p_line_2': 3 * lines['p_a_from_mw'].iloc[1] * 1.0e6,
               'p_trafo': 3 * lines['p_a_from_mw'].iloc[0] * 1.0e6,
               'q_line_1': 3 * lines['q_a_from_mvar'].iloc[0] * 1.0e6,
               'q_line_2': 3 * lines['q_a_from_mvar'].iloc[1] * 1.0e6,
               'q_trafo': 3 * lines['q_a_from_mvar'].iloc[0] * 1.0e6,
               'loading_percent_line_1': lines['loading_percent'].iloc[0],
               'loading_percent_line_2': lines['loading_percent'].iloc[0]}
    return results

def get_n_loads_cuts(net, p0s, q0s):
    # TODO: split for p and q. look at (40, 20, 20) and (20, 10, 10)
    # TODO: 2 loads on p and 3 on q
    n_loads = 1
    v_load = 0.0
    while v_load < v_min:
        ps = np.zeros(len(p0s))
        qs = np.zeros(len(q0s))
        ps[:n_loads] = p0s[:n_loads]/3
        qs[:n_loads] = q0s[:n_loads]/3
        results = run_single_pf(net, ps, qs)
        v_load = results['v_bus_load']
        n_loads += 1
    print(f"Number of loads used for cuts: {n_loads}")
    return n_loads


def calculate_cut(net, p0s, q0s, n_loads_p, n_loads_q):
    # Assume p and q in standard units
    # Do the calculation over 1 phase
    p0s_ph = p0s[:n_loads_p] / 3
    q0s_ph = q0s[:n_loads_q] / 3

    n_cables_p = n_loads_p
    n_cables_q = n_loads_q
    r = l * r_ohm_per_km
    x = l * x_ohm_per_km
    R = n_cables_p * r
    X = n_cables_q * x

    print(f'slope loss: {-R / X}, {n_loads_p}, {n_loads_q}')
    slope = - R/X
    intercept = (v_base**2 - v_min**2)/(2 * X)
    intercept += 1/X * sum(sum(r * p0s_ph[:n]) for n in range(1, len(p0s_ph)))
    intercept += 1/X * sum(sum(x * q0s_ph[:n]) for n in range(1, len(q0s_ph)))
    return slope, 3 * intercept


def calculate_cut_loss(net, p0s, q0s, n_loads_p, n_loads_q):
    print(f"Calculate cut for: {n_loads_p, n_loads_q}")
    # Assume p and q in standard units
    # Do the calculation over 1 phase
    p0s_ph = p0s[:n_loads_p] / 3
    q0s_ph = q0s[:n_loads_q] / 3

    n_cables_p = n_loads_p
    n_cables_q = n_loads_q
    r = l * r_ohm_per_km
    x = l * x_ohm_per_km
    R = n_cables_p * r
    X = n_cables_q * x

    print(f'slope loss: {-R/X}, {n_loads_p}, {n_loads_q}')
    slope = - R/X
    intercept = (v_base**2 - v_min**2)/(2 * X)
    intercept += 1/X * sum(sum(r * p0s_ph[:n])for n in range(1, n_loads_p))
    intercept += 1/X * sum(sum(x * q0s_ph[:n])for n in range(1, n_loads_q))

    # Add loss terms
    # Calculate first order approximation of the loss terms
    # Calculate line flows based on the loads
    P0s_ph = np.cumsum(p0s_ph[::-1])[::-1]
    Q0s_ph = np.cumsum(q0s_ph[::-1])[::-1]

    # Make sure that the arrays are of equal length
    n_loads_max = max(n_loads_p, n_loads_q)
    if len(P0s_ph) < n_loads_max:
        P0s_ph_copy = P0s_ph.copy()
        P0s_ph = np.zeros(n_loads_q)
        P0s_ph[:n_loads_p] = P0s_ph_copy
    if len(Q0s_ph) < n_loads_max:
        Q0s_ph_copy = Q0s_ph.copy()
        Q0s_ph = np.zeros(n_loads_p)
        Q0s_ph[:n_loads_q] = Q0s_ph_copy

    print(P0s_ph, Q0s_ph)
    # print("check lengths or correct?", 1.0/0)
    print("flows")
    print(P0s_ph * 3, Q0s_ph * 3)
    currents = (P0s_ph**2 + Q0s_ph**2)/v_base**2
    losses_P = r * currents
    losses_Q = x * currents

    print(losses_P * 3, losses_Q * 3)
    # We only add terms if there was P0 and Q0 through the branches
    # This causes a larger error in the cases where only 1 load causes the voltage drop already
    intercept += 1 / X * sum(sum(r * losses_P[:n + 1]) for n in range(1, n_loads_p))  # + 1 is more consistent with derivation
    intercept += 1 / X * sum(sum(x * losses_Q[:n + 1]) for n in range(1, n_loads_q))  # + 1 is more consistent with derivation
    return slope, 3 * intercept


# User input
ps_max = np.array([40, 20, 20]) * 1.0e3
qs_max = np.array([20, 10, 10]) * 1.0e3

assert len(ps_max) == len(qs_max), 'ps and qs should be of the same length'
n_loads = len(ps_max)
net = create_network_n(r_ohm_per_km, x_ohm_per_km, c_nf_per_km, l, n_loads)

p_max = np.sum(ps_max)
q_max = np.sum(qs_max)
num = 201

p_tot_values = np.linspace(0, p_max, num)
q_tot_values = np.linspace(0, q_max, num)

ratios = np.linspace(0, 1.0, num)

'''
p_trafo = []
q_trafo = []

count = 0
for p_tot in p_tot_values:
    # Bucket fill P
    ps = np.zeros(len(ps_max))
    for i in range(len(ps)):
        ps[i] = min(p_tot, ps_max[i])
        p_tot -= ps[i]

    for q_tot in q_tot_values:
        # Bucket fill Q
        qs = np.zeros(len(qs_max))
        for i in range(len(qs)):
            qs[i] = min(q_tot, qs_max[i])
            q_tot -= qs[i]

        results = run_single_pf(net, ps, qs)

        v_load = results['v_bus_load']
        count += 1
        # For debugging
        # print(ps, qs)
        # print(results['p_trafo'], results['q_trafo'])
        # print(v_load)
        print(f'run: {count}/{num * num}')
        if v_load >= v_min:
            p_trafo.append(results['p_trafo'])
            q_trafo.append(results['q_trafo'])

with open('p_trafo_200.pkl', 'wb') as file:
    pickle.dump(np.array(p_trafo), file)

with open('q_trafo_200.pkl', 'wb') as file:
    pickle.dump(np.array(q_trafo), file)
    
'''

# '''
with open('p_trafo_200.pkl', 'rb') as file_p:
    p_trafo = pickle.load(file_p)

with open('q_trafo_200.pkl', 'rb') as file_q:
    q_trafo = pickle.load(file_q)
# '''



plt.scatter(np.array(p_trafo)/1.0e3, np.array(q_trafo)/1.0e3)
max_total_p = np.sum(ps_max)
p_space_cut = np.linspace(0.0 * max_total_p/1.0e3, 1.1 * max_total_p/1.0e3, num=10)

for n_loads_p in range(1, n_loads + 1):
    for n_loads_q in range(1, n_loads + 1):
        print(f"Cuts for loop: {n_loads_p, n_loads_q}")
        slope, intercept = calculate_cut(net, ps_max, qs_max, n_loads_p, n_loads_q)

        print(f'cut: {slope, intercept}')
        # plt.plot(p_space_cut, p_space_cut * slope + intercept/1.0e3, label=f'({n_loads_p}, {n_loads_q})')
        slope, intercept = calculate_cut_loss(net, ps_max, qs_max, n_loads_p, n_loads_q)
        print(f'cut: {slope, intercept}')
        to_show = [(1, 3), (2, 2), (2, 1)]
        if (n_loads_p, n_loads_q) in to_show:
            plt.plot(p_space_cut, p_space_cut * slope + intercept/1.0e3, label=f'({n_loads_p}, {n_loads_q})')
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

print(run_single_pf(net, np.array([40e3, 20e3, 20e3]), np.array([20e3, 0, 0])))

'''
kijk naar eerste node is 400^2
daarna hebben we termen
drop = sqrt(- 2 r P^0 - 2 r P^1 + (r^2 + x^2) (P^0 ^2 + Q^0 ^2)/v^2 0)



'''