import pandapower as pp
from radialgraph import Node, Root, Edge, RadialGraph
import numpy as np
import matplotlib.pyplot as plt


def create_graph() -> RadialGraph:
    # Create a radial graph
    graph = RadialGraph()

    # Create nodes
    root = Root("Root")
    node1 = Node("Node 1", 40, 20)  # p, q in kW
    node2 = Node("Node 2", 30, 15)
    node3 = Node("Node 3", 20, 10)
    node4 = Node("Node 4", 50, 25)
    node5 = Node("Node 5", 20, 10)
    node6 = Node("Node 6", 30, 5)
    node7 = Node("Node 7", 10, 2)

    # Add nodes to the graph
    graph.add_node(root)
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    graph.add_node(node5)
    graph.add_node(node6)
    graph.add_node(node7)

    # Add edges to the graph
    graph.add_edge(root, node1, 1, 1)  # r, x in units
    graph.add_edge(node1, node2, 1, 1)
    graph.add_edge(node1, node3, 2, 2)
    graph.add_edge(node1, node4, 1, 1)
    graph.add_edge(node2, node5, 1, 1)
    graph.add_edge(node3, node6, 1, 1)
    graph.add_edge(node4, node7, 1, 1)
    return graph


def create_pp_network_from_graph(radial_graph: RadialGraph):
    net = pp.create_empty_network()
    vn_kv_mv = 10.25  # kV
    vn_kv_lv = 0.4  # kV

    # Create nodes and loads
    nodes_graph = radial_graph.nodes
    busses = dict()
    # Create root node
    root = nodes_graph[0]
    bus_root = pp.create_bus(net, name=root.name, vn_kv=vn_kv_lv, type='n')
    busses[root.name] = bus_root
    # Create other nodes
    for n in nodes_graph[1:]:
        bus = pp.create_bus(net, name=n.name, vn_kv=vn_kv_lv, type='n')
        pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, type='wye', name="Load " + str(n.name))
        busses[n.name] = bus

    # Create external grid
    # Params
    s_sc_max_mva = 1000
    rx_max = 0.1
    x0x_max = 1.0
    r0x0_max = 0.1

    pp.create_ext_grid(net, bus=bus_root, vm_pu=1.0, va_degree=0.0, slack_weight=1.0,
                       s_sc_max_mva=s_sc_max_mva, in_service=True, name='Grid Connection',
                       rx_max=rx_max, x0x_max=x0x_max, r0x0_max=r0x0_max)

    # Create branches
    edges_graph = radial_graph.edges

    # Params
    r_unit = 0.14  # Ohm
    x_unit = 0.04  # Ohm
    l = 100 * 1.0e-3  # km

    r0_ohm_per_km = 1.5 * 1.64  # Ohm/km
    x0_ohm_per_km = 1.5 * 0.15   # Ohm/km
    c0_nf_per_km = 1.5 * 0.32 * 1.0e3  # nF/km
    max_i_ka = 100 * 110 * 1.0e-3  # kA

    for i, e in enumerate(edges_graph):
        from_bus = busses[e.node1.name]
        to_bus = busses[e.node2.name]
        r_ohm_per_km = e.resistance * r_unit / l
        x_ohm_per_km = e.reactance * x_unit / l
        c_nf_per_km = 1.5 * 0.32 * 1.0e3  # nF/km

        pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, length_km=l, r_ohm_per_km=r_ohm_per_km,
                                       x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km, max_i_ka=max_i_ka,
                                       name="Line" + str(i), r0_ohm_per_km=r0_ohm_per_km, x0_ohm_per_km=x0_ohm_per_km,
                                       c0_nf_per_km=c0_nf_per_km)
    return net


def run_single_pf(net, active_powers: list, reactive_powers: list) -> dict:
    # Set load
    net.load['p_mw'] = [1.0e-3 * p for p in active_powers]
    net.load['q_mvar'] = [1.0e-3 * q for q in reactive_powers]

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

    results = {'v_buses': busses['vm_a_phn'],
               # 'p_line_1': 3 * lines['p_a_from_mw'].iloc[0] * 1.0e6,
               # 'p_line_2': 3 * lines['p_a_from_mw'].iloc[1] * 1.0e6,
               'p_trafo': 3 * lines['p_a_from_mw'].iloc[0] * 1.0e3,  # in kW
               # 'q_line_1': 3 * lines['q_a_from_mvar'].iloc[0] * 1.0e6,
               # 'q_line_2': 3 * lines['q_a_from_mvar'].iloc[1] * 1.0e6,
               'q_trafo': 3 * lines['q_a_from_mvar'].iloc[0] * 1.0e3}  # in kW

    return results


if __name__ == "__main__":
    graph = create_graph()
    net = create_pp_network_from_graph(graph)
    info = graph.get_active_bucket_fill_info()
    print(info)

    # Calculate numerator slope and numerator p_term intercept
    # We do this for every iteration of the bucket-fill algorithm
    for i in range(1, len(info) + 1):  # number of iterations <= number of nodes
        iteration_info = info['iteration_' + str(i)]
        fill_factors = iteration_info['fill_factors']
        to_fill_nodes = iteration_info['to_fill_nodes']

        destination_node = to_fill_nodes[0]  # the exact node does not matter as the voltage is equal
        print(f"destination: {destination_node.name}")
        numerator_slope = 0.0
        # Loop 1 for outer sum over the path
        for edge in destination_node.get_edge_path():
            r = edge.resistance
            print(edge.node2.name)
            tree = edge.node2.get_nodes_in_tree()
            print(f"Tree: {[n.name for n in tree]}")
            sum_fill_factors_in_tree = 0.0
            # Loop 2 for inner sum over the nodes in the tree
            for n in tree:
                if n in to_fill_nodes:
                    print(f"f taken into acount of: {n.name}")
                    sum_fill_factors_in_tree += fill_factors[to_fill_nodes.index(n)]
            print(f"new sum: {sum_fill_factors_in_tree}")  # for every edge it prints
            numerator_slope += r * sum_fill_factors_in_tree
        print(f"numerator_slope: {numerator_slope}")

    print(1.0/0)

    for node in graph.nodes:
        print(node.name, [n.name for n in node.get_nodes_in_tree()])
    print(1.0 / 0)

