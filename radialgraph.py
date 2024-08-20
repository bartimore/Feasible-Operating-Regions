from __future__ import annotations
import numpy as np
from itertools import combinations
from typing import Optional
import pandapower as pp


class Node:
    def __init__(self, name: str, p_max: float, q_max: float):
        # Apply checks
        assert p_max >= 0, "Please don't use only positive values (loads) for p_max"
        assert q_max >= 0, "Please don't use only positive values (loads) for q_max"

        self.name = name
        self.p_max = p_max
        self.q_max = q_max
        self.behind_neighbors = []
        self.edge_path = None  # The list of edges from the root node to this node

    def add_behind_neighbor(self, neighbor: Node):
        self.behind_neighbors.append(neighbor)

    def get_behind_neighbors(self):
        return self.behind_neighbors

    def add_edge_path(self, path: list):
        self.edge_path = path

    def get_edge_path(self) -> list:
        return self.edge_path

    def get_nodes_in_tree(self) -> set:
        nodes_in_tree = self.depth_first_search(self)
        return nodes_in_tree

    def depth_first_search(self, start_node, nodes_visited=None) -> set:
        # Basic iterative implementation of depth first search algorithm
        # This will be used to search the graph to obtain nodes after other nodes
        if nodes_visited is None:
            nodes_visited = set()
        nodes_visited.add(start_node)
        for neighbor in start_node.get_behind_neighbors():
            if neighbor not in nodes_visited:
                self.depth_first_search(neighbor, nodes_visited)
        return nodes_visited


class Root(Node):
    def __init__(self, name="Root", p_max=0.0, q_max=0.0):
        epsilon = 1.0e-6
        assert name == 'Root', "Please don't change the name of the root node"
        assert p_max <= epsilon, "Please don't add a load to the root node"
        assert q_max <= epsilon, "Please don't add a load to the root node"
        super().__init__(name, p_max, q_max)
        self.add_edge_path([])


class Edge:
    def __init__(self, node1, node2, resistance, reactance):
        self.node1 = node1
        self.node2 = node2
        self.resistance = resistance
        self.reactance = reactance


class RadialGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.pp_representation: Optional[dict] = None
        self.connectivity_matrix: Optional[np.array] = None
        self.load_connectivity_vector: Optional[np.array] = None

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, node1: Node, node2: Node, resistance: float, reactance: float):
        edge = Edge(node1, node2, resistance, reactance)
        self.edges.append(edge)
        node1.add_behind_neighbor(node2)
        assert node1.edge_path is not None, "node1 should have a edge_path value"
        node2.add_edge_path(node1.edge_path + [edge])

    def set_pp_representation(self):
        """
        Create pandapower network object based on the RadialGraph topology and properties
        More parameters are necesssary than the RadialGraph provides, so typical values are choses
        load and voltage properties are used as dummy variables, as these will be specified later
        We define:
        - buses
        - loads
        - external grid
        - lines
        """
        net = pp.create_empty_network()

        # Create buses and loads
        name_nodes_map = dict()  # Use to map the buses to the lines
        # Create root node
        root = [n for n in self.nodes if n.name == 'Root'][0]
        bus_root = pp.create_bus(net, name=root.name, vn_kv=0.0, type='n')
        name_nodes_map[root.name] = bus_root
        # Create other nodes
        for n in self.nodes:
            if n.name != 'Root':
                bus = pp.create_bus(net, name=n.name, vn_kv=0.0, type='n')
                pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, type='wye', name="Load " + str(n.name))
                name_nodes_map[n.name] = bus

        # Create external grid
        # Params
        s_sc_max_mva = 1000  # Set huge value
        rx_max = 0.1
        x0x_max = 1.0
        r0x0_max = 0.1

        pp.create_ext_grid(net, bus=bus_root, vm_pu=1.0, va_degree=0.0, slack_weight=1.0,
                           s_sc_max_mva=s_sc_max_mva, in_service=True, name='Grid Connection',
                           rx_max=rx_max, x0x_max=x0x_max, r0x0_max=r0x0_max)

        # Create lines
        # Params
        max_i_ka = 100  # kA
        max_loading_percent = 100.0

        for i, e in enumerate(self.edges):
            from_bus = name_nodes_map[e.node1.name]
            to_bus = name_nodes_map[e.node2.name]
            r_ohm_per_km = e.resistance
            x_ohm_per_km = e.reactance
            c_nf_per_km = 0.0  # nF/km

            pp.create_line_from_parameters(net, from_bus=from_bus, to_bus=to_bus, length_km=1.0,
                                           r_ohm_per_km=r_ohm_per_km,
                                           x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km, max_i_ka=max_i_ka,
                                           name="Line" + str(i), max_loading_percent=max_loading_percent,
                                           )

        self.pp_representation = net

    def set_connectivity_attributes(self):
        # Calculate the connectivity vector and matrix necessary for the lossless powerflow
        self._set_connectivity_matrix()
        self._set_load_connectivity_vector()

    def _set_connectivity_matrix(self):
        # construct connectivity matrix: n_nodes x n_edges, (n, e) 1 if line e is coming into node n
        # (n, e) is -1 if line is going out of bus n
        # We make it square afterwards by removing the balance equation at the root node
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        connectivity_matrix = np.zeros((n_nodes, n_edges))
        node_name_index_map = {node.name: index for index, node in enumerate(self.nodes)}
        print(node_name_index_map)

        for e_index, e in enumerate(self.edges):
            from_node = e.node1
            to_node = e.node2

            connectivity_matrix[node_name_index_map[from_node.name], e_index] = -1
            connectivity_matrix[node_name_index_map[to_node.name], e_index] = 1

        # Remove the row for the root node
        connectivity_matrix = np.delete(connectivity_matrix, node_name_index_map["Root"], axis=0)
        self.connectivity_matrix = connectivity_matrix

    def _set_load_connectivity_vector(self):
        # Just like in the connectivity matrix we don't consider the root node
        # TODO:  generalize to settings where a node can also have no load
        self.load_connectivity_vector = np.array([1 for load in range(len(self.nodes) - 1)])

    def apply_active_bucket_filling(self, total_active_power: float) -> dict:
        print("apply bucket filling active power")
        info = self.get_active_bucket_filling_info(total_active_power)
        loads = info[list(info)[-1]]['loads_end']  # Get the loads end from the last iteration
        return loads

    def apply_reactive_bucket_filling(self, total_reactive_power: float) -> dict:
        print("apply bucket filling reactive power")
        info = self.get_reactive_bucket_filling_info(total_reactive_power)
        loads = info[list(info)[-1]]['loads_end']  # Get the loads end from the last iteration
        return loads

    def _get_bucket_filling_info(self, load_type: str, edge_prop_type: str, load_to_fill: float):
        # TODO: Remove commented out print statements used for debugging
        information_dict = dict()
        # Note: load_type is either 'p' or 'q', edge_prop_type is either 'resistance' or 'reactance'
        max_load_type = load_type + '_max'  # either p_max or q_max

        # Do some input tests
        assert load_to_fill <= sum([getattr(n, max_load_type) for n in self.nodes]), "provided load_to_fill is too big"

        # Allocate output: how much load to put at every node
        loads = {n.name: 0.0 for n in self.nodes}
        remaining_bucket_sizes = {n.name: getattr(n, max_load_type) for n in self.nodes}

        # Initialize: start at the root node
        root = [n for n in self.nodes if n.name == 'Root'][0]
        nodes_to_fill = root.behind_neighbors.copy()  # Make copy to not alter the state of the root node
        total_load_already_filled = 0.0

        iteration = 1

        # Make sure the information dict contains information if no real iteration is necessary
        if load_to_fill < 1.0e-5:
            iteration_information = dict()
            iteration_information['loads_start'] = loads.copy()
            iteration_information['loads_end'] = loads.copy()
            information_dict['iteration_' + str(iteration)] = iteration_information

        while round(load_to_fill, 5) > 0.0:  # round is for preventing small residual load due to floating point errors
            # print(f"load_to_fill: {load_to_fill}")
            # print(f"nodes_to_fill: {[n.name for n in nodes_to_fill]}")
            # Set up dict to store outputs
            iteration_information = dict()

            # Construct load factors and determine which loads are filled to max capacity
            # For the load factors we calculate the resistance/reactance to the node from the last common node
            edge_paths = self._calculate_edge_paths_from_last_common_node(nodes_to_fill)
            edge_prop_paths = [sum(getattr(e, edge_prop_type) for e in edge_path) for edge_path in edge_paths]
            fill_factors = self._calculate_fill_factors(edge_prop_paths)

            # Find the loads that are first the be full given the fill factors
            total_load_full = [remaining_bucket_sizes[n.name] / fill_factors[i] for i, n in enumerate(nodes_to_fill)]
            # print(f'Total load full: {total_load_full}')
            min_total_load_full = min(total_load_full)
            nodes_max_fill = [n for i, n in enumerate(nodes_to_fill) if total_load_full[i] == min_total_load_full]

            # Constrain the total amount of filling by the amount that is left to fill
            total_fill = min(min_total_load_full, load_to_fill)
            fills = [factor * total_fill for factor in fill_factors]
            # print(f'Total fill: {total_fill}')
            # print(f'fills: {fills}')

            # Update information
            iteration_information['to_fill_nodes'] = nodes_to_fill.copy()
            iteration_information['to_fill_nodes_names'] = [n.name for n in nodes_to_fill]

            # Update the loads
            # print(f'Old remaining_bucket_sizes: {remaining_bucket_sizes}')
            iteration_information['loads_start'] = loads.copy()
            for i, n in enumerate(nodes_to_fill):
                loads[n.name] += fills[i]
                remaining_bucket_sizes[n.name] -= fills[i]
            iteration_information['loads_end'] = loads.copy()
            # print(f'New remaining_bucket_sizes: {remaining_bucket_sizes}')

            # Check whether we have leftover load to fill. If so, update nodes_to_fill
            if load_to_fill > total_fill:
                # print("NOT FULL THIS ROUND", load_to_fill, total_fill)
                # print(f'Old nodes to fill: {[n.name for n in nodes_to_fill]}')
                # print(f'Nodes full: {[n.name for n in nodes_max_fill]}')
                # Add new nodes and remove full nodes
                for n in nodes_max_fill:
                    nodes_to_fill += n.behind_neighbors
                    nodes_to_fill.remove(n)
                # print(f'New nodes to fill: {[n.name for n in nodes_to_fill]}')
                # print(f'New remaining_bucket_sizes: {remaining_bucket_sizes}')
            else:
                print(f"Bucket fill finished after {iteration} iterations")

            # Update load_to_fill
            # print(f"Old load to fill: {load_to_fill}")
            load_to_fill -= total_fill
            # print(f"New load to fill: {load_to_fill}")
            iteration_information['fill_factors'] = fill_factors
            iteration_information['total_load_filled_start'] = total_load_already_filled
            total_load_already_filled += total_fill
            information_dict['iteration_' + str(iteration)] = iteration_information
            iteration += 1

        return information_dict

    def get_active_bucket_filling_info(self, total_active_power: float) -> dict:
        # _apply_bucket_filling is set up flexibly to be applied on both active/reactive power
        info = self._get_bucket_filling_info('p', 'resistance', total_active_power)
        return info

    def get_reactive_bucket_filling_info(self, total_reactive_power: float) -> dict:
        # _apply_bucket_filling is set up flexibly to be applied on both active/reactive power
        info = self._get_bucket_filling_info('q', 'reactance', total_reactive_power)
        return info

    @staticmethod
    def _calculate_edge_paths_from_last_common_node(nodes):
        # Split the case with only 1 node, because we have a common node otherwise
        if len(nodes) == 1:
            # Take only the last resistance if there is only 1 node left
            edge_paths = [[nodes[0].edge_path[-1]]]  # list (for every edge) of edge_paths (being lists)
        else:
            # Remove all the common edges
            edge_paths = [set(n.edge_path) for n in nodes]
            common_edges = set.intersection(*edge_paths)
            edge_paths = [edge_path.difference(common_edges) for edge_path in edge_paths]
        return edge_paths

    @staticmethod
    def _calculate_fill_factors(edge_property_paths: list) -> list:
        """
        If we have r1, r2, r3 we calculate
        for the first factor r2 r3/(r1 r2 + r1 r3 + r2 r3), etc.
        """
        factors = []
        if len(edge_property_paths) == 1:
            factors.append(1.0)
        else:
            for i in range(len(edge_property_paths)):
                numerator = np.prod(edge_property_paths[:i] + edge_property_paths[i + 1:])
                denominator = sum([np.prod(comb) for comb in combinations(edge_property_paths, len(edge_property_paths) - 1)])
                factor = numerator / denominator
                factors.append(factor)

        return factors
