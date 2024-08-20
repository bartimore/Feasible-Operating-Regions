from radialgraph import RadialGraph
import numpy as np
import pandapower as pp


class LoadFlowCalculator:
    def __init__(self, radial_graph: RadialGraph):
        # Set relevant grid representation for power flows
        radial_graph.set_connectivity_attributes()
        radial_graph.set_pp_representation()
        self.radial_graph = radial_graph

    def do_lossless_loadflow(self, p_loads: list, q_loads: list, v2_root: float) -> dict:
        # Apply basic checks
        assert len(p_loads) == len(self.radial_graph.nodes) - 1, "Provide loads for all loads except root node"
        assert len(q_loads) == len(self.radial_graph.nodes) - 1, "Provide loads for all loads except root node"

        # Calculate active power in edges
        P_edges = self._calculate_lossless_edge_flows(p_loads)

        # Calculate reactive power in edges
        Q_edges = self._calculate_lossless_edge_flows(q_loads)

        # Calculate squared voltages in nodes
        v2_nodes = self._calculate_v2s(P_edges, Q_edges, v2_root)
        v_nodes = [np.sqrt(v2) for v2 in v2_nodes]

        return {'P_edges': P_edges, 'Q_edges': Q_edges, 'v_nodes': v_nodes}

    def do_pp_loadflow(self, p_loads: list, q_loads: list, v2_root: float) -> dict:
        # Apply basic checks
        assert len(p_loads) == len(self.radial_graph.nodes) - 1, "Provide loads for all loads except root node"
        assert len(q_loads) == len(self.radial_graph.nodes) - 1, "Provide loads for all loads except root node"

        net = self.radial_graph.pp_representation

        # Set loads and voltage
        net.load['p_mw'] = [1.0e-3 * p for p in p_loads]
        net.load['q_mvar'] = [1.0e-3 * q for q in q_loads]
        net.bus['vn_kv'] = np.sqrt(v2_root) * np.sqrt(3) / 1000.0

        # Do power flow
        pp.runpp(net, check_connectivity=True)

        # Get results
        buses = net.res_bus
        lines = net.res_line[['p_from_mw', 'q_from_mvar']]

        P_edges = list(lines['p_from_mw'].values * 1.0e3)
        Q_edges = list(lines['q_from_mvar'].values * 1.0e3)
        v_nodes = list((buses['vm_pu'] * net.bus['vn_kv'] * 1.0e3 / np.sqrt(3)).values)

        return {'P_edges': P_edges, 'Q_edges': Q_edges, 'v_nodes': v_nodes}

    def _calculate_lossless_edge_flows(self, loads: list) -> list:
        # Set load vector based on connectivity and solve the linear system
        # Linear system: connectivity matrix P = load vector
        assert len(loads) == np.sum(self.radial_graph.load_connectivity_vector), "Load nr does not match connectivity"
        load_vector = np.zeros(len(self.radial_graph.nodes) - 1)
        load_indices = np.where(self.radial_graph.load_connectivity_vector == 1)
        load_vector[load_indices] = loads

        edge_flows = np.linalg.solve(self.radial_graph.connectivity_matrix, load_vector).tolist()
        return edge_flows

    def _calculate_v2s(self, P_edges: list, Q_edges: list, v2_root: float) -> list:
        # lindistflow: v2_n+1 = v2_n - 2RP - 2XQ
        # We again abuse the connectivity matrix as follows:
        # C^T v2 = drops along edges, this system represents an equation for every edge
        r_edges = [e.resistance for e in self.radial_graph.edges]
        x_edges = [e.reactance for e in self.radial_graph.edges]
        P_edges_W = [1000.0/3 * P for P in P_edges]  # base v is 230 so calculate with 1 phase
        Q_edges_W = [1000.0/3 * Q for Q in Q_edges]  # base v is 230 so calculate with 1 phase

        edge_v2_drops = [-2 * r_edges[i] * P_edges_W[i] - 2 * x_edges[i] * Q_edges_W[i]
                         for i in range(len(self.radial_graph.edges))]
        node_v2_drops = np.linalg.solve(self.radial_graph.connectivity_matrix.T, edge_v2_drops)
        v2_nodes = (np.array(node_v2_drops) + v2_root).tolist()
        return v2_nodes


