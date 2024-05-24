import numpy as np
from itertools import combinations


class Node:
    def __init__(self, name, p_max, q_max):
        self.name = name
        self.p_max = p_max
        self.q_max = q_max
        self.behind_neighbors = []
        self.edge_path = None  # The list of edges from the root node to this node

    def add_behind_neighbor(self, neighbor):
        self.behind_neighbors.append(neighbor)

    def get_behind_neighbors(self):
        return self.behind_neighbors

    def add_edge_path(self, path):
        self.edge_path = path

    def get_edge_path(self):
        return self.edge_path

    def get_nodes_in_tree(self):
        nodes_in_tree = self.depth_first_search(self)
        return nodes_in_tree

    def depth_first_search(self, start_node, nodes_visited=None):
        if nodes_visited is None:
            nodes_visited = set()
        nodes_visited.add(start_node)
        for neighbor in start_node.get_behind_neighbors():
            if neighbor not in nodes_visited:
                self.depth_first_search(neighbor, nodes_visited)
        return nodes_visited


class Root(Node):
    def __init__(self, name, p_max=0.0, q_max=0.0):
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

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, node1, node2, resistance, reactance):
        edge = Edge(node1, node2, resistance, reactance)
        self.edges.append(edge)
        node1.add_behind_neighbor(node2)
        assert node1.edge_path is not None, "node1 should have a edge_path value"
        node2.add_edge_path(node1.edge_path + [edge])

    def apply_active_bucket_filling(self, total_active_power: float) -> dict:
        # _apply_bucket_filling is set up flexibly to be applied on both active/reactive power
        loads = self._apply_bucket_filling('p', 'resistance', total_active_power)
        return loads

    def apply_reactive_bucket_filling(self, total_reactive_power: float) -> dict:
        # _apply_bucket_filling is set up flexibly to be applied on both active/reactive power
        loads = self._apply_bucket_filling('q', 'reactance', total_reactive_power)
        return loads

    def _apply_bucket_filling(self, load_type: str, edge_prop_type: str, load_to_fill: float):
        # This is where the magic happens
        # Note: load_type is either 'p' or 'q', edge_prop_type is either 'resistance' or 'reactance'
        max_load_type = load_type + '_max'  # either p_max or q_max

        # Do some input tests
        assert max_load_type in ['p_max', 'q_max'], 'either provide p or q as load type'
        assert edge_prop_type in ['resistance', 'reactance'], 'either provide resistance or reactance for edge_prop_type'
        assert load_to_fill <= sum([getattr(n, max_load_type) for n in self.nodes]), "provided load_to_fill is too big"

        # Allocate output: how much load to put at every node
        loads = {n.name: 0.0 for n in self.nodes}

        # Initialize: start at the root node
        root = self.nodes[0]
        nodes_to_fill = root.behind_neighbors

        while round(load_to_fill, 5) > 0.0:  # round is for preventing small residual load due to floating point errors
            # Construct load factors and determine which loads are filled to max capacity
            # For the load factors we calculate the resistance/reactance to the node from the last common node
            edge_paths = self._calculate_edge_paths_from_last_common_node(nodes_to_fill)
            edge_prop_paths = [sum(getattr(e, edge_prop_type) for e in edge_path) for edge_path in edge_paths]
            print('edge_prop_paths: ', edge_prop_paths)
            load_maxs = [getattr(n, max_load_type) for n in nodes_to_fill]
            fill_factors = self._calculate_fill_factors(edge_prop_paths)

            # Find the loads that are first the be full given the fill factors
            total_load_full = [p_max / factor for factor, p_max in zip(fill_factors, load_maxs)]
            print(f'Total load full: {total_load_full}')
            min_total_load_full = min(total_load_full)
            nodes_max_fill = [nodes_to_fill[i] for i, x in enumerate(total_load_full) if x == min_total_load_full]

            # Constrain the total amount of filling by the amount that is left to fill
            total_fill = min(min_total_load_full, load_to_fill)
            fills = [factor * total_fill for factor in fill_factors]
            print(f'Total fill: {total_fill}')
            print(f'fills: {fills}')

            # Update the max loads, nodes to fill and the total load to fill
            # Check whether we have leftover load to fill
            if load_to_fill > total_fill:
                print("NOT FULL THIS ROUND")
                # Update p_max based on fills
                print(f'Old p_max: {[getattr(n, max_load_type) for n in nodes_to_fill]}')
                for i, n in enumerate(nodes_to_fill):
                    new_max_load = getattr(n, max_load_type) - fills[i]
                    setattr(n, max_load_type, new_max_load)
                    loads[n.name] += fills[i]

                print(f'Old nodes to fill: {[n.name for n in nodes_to_fill]}')
                print(f'Nodes full: {[n.name for n in nodes_max_fill]}')
                # Add new nodes and remove full nodes
                # TODO: this changes the neighbors attribute of the root node? Why?
                for n in nodes_max_fill:
                    nodes_to_fill += n.behind_neighbors
                    nodes_to_fill.remove(n)
                print(f'New nodes to fill: {[n.name for n in nodes_to_fill]}')
                print(f'New p_max: {[getattr(n, max_load_type) for n in nodes_to_fill]}')

            else:
                print("FULL THIS ROUND")
                for i, n in enumerate(nodes_to_fill):
                    loads[n.name] += fills[i]

            # Update load_to_fill
            print(f"Old load to fill: {load_to_fill}")
            load_to_fill -= total_fill
            print(f"New load to fill: {load_to_fill}")

        print(f"Loads: {loads}")
        print(f"sum: {sum(loads.values())}, load_to_fill: {load_to_fill}")
        return loads

    def get_active_bucket_fill_info(self) -> dict:
        # _apply_bucket_filling is set up flexibly to be applied on both active/reactive power
        info = self._get_bucket_fill_info('p', 'resistance')
        return info

    def get_reactive_bucket_fill_info(self) -> dict:
        # _apply_bucket_filling is set up flexibly to be applied on both active/reactive power
        info = self._get_bucket_fill_info('q', 'reactance')
        return info

    def _get_bucket_fill_info(self, load_type: str, edge_prop_type: str) -> dict:
        information_dict = dict()

        # Note: load_type is either 'p' or 'q', edge_prop_type is either 'resistance' or 'reactance'
        max_load_type = load_type + '_max'  # either p_max or q_max

        # Do some input tests
        assert max_load_type in ['p_max', 'q_max'], 'either provide p or q as load type'
        assert edge_prop_type in ['resistance', 'reactance'], 'either provide resistance or reactance for edge_prop_type'
        load_to_fill = sum([getattr(n, max_load_type) for n in self.nodes])

        # Allocate output: how much load to put at every node
        loads = {n.name: 0.0 for n in self.nodes}

        # Initialize: start at the root node
        root = self.nodes[0]
        nodes_to_fill = root.behind_neighbors
        nodes_already_filled = []
        total_load_already_filled = 0.0

        iteration = 1
        while round(load_to_fill, 5) > 0.0:  # round is for preventing small residual load due to floating point errors
            # Set up dict to store outputs
            iteration_information = dict()

            # Construct load factors and determine which loads are filled to max capacity
            # For the load factors we calculate the resistance/reactance to the node from the last common node
            edge_paths = self._calculate_edge_paths_from_last_common_node(nodes_to_fill)
            edge_prop_paths = [sum(getattr(e, edge_prop_type) for e in edge_path) for edge_path in edge_paths]
            print('edge_prop_paths: ', edge_prop_paths)
            load_maxs = [getattr(n, max_load_type) for n in nodes_to_fill]
            fill_factors = self._calculate_fill_factors(edge_prop_paths)

            # Find the loads that are first the be full given the fill factors
            total_load_full = [p_max / factor for factor, p_max in zip(fill_factors, load_maxs)]
            print(f'Total load full: {total_load_full}')
            min_total_load_full = min(total_load_full)
            nodes_max_fill = [nodes_to_fill[i] for i, x in enumerate(total_load_full) if x == min_total_load_full]

            # Constrain the total amount of filling by the amount that is left to fill
            total_fill = min(min_total_load_full, load_to_fill)
            fills = [factor * total_fill for factor in fill_factors]
            print(f'Total fill: {total_fill}')
            print(f'fills: {fills}')

            # Update
            iteration_information['to_fill_nodes'] = nodes_to_fill.copy()
            iteration_information['to_fill_nodes_names'] = [n.name for n in nodes_to_fill]

            # Update the max loads, nodes to fill and the total load to fill
            # Check whether we have leftover load to fill
            if load_to_fill > total_fill:
                print("NOT FULL THIS ROUND")
                # Update p_max based on fills
                print(f'Old p_max: {[getattr(n, max_load_type) for n in nodes_to_fill]}')
                for i, n in enumerate(nodes_to_fill):
                    new_max_load = getattr(n, max_load_type) - fills[i]
                    setattr(n, max_load_type, new_max_load)
                    loads[n.name] += fills[i]

                print(f'Old nodes to fill: {[n.name for n in nodes_to_fill]}')
                print(f'Nodes full: {[n.name for n in nodes_max_fill]}')
                # Add new nodes and remove full nodes
                # TODO: this changes the neighbors attribute of the root node? Why?
                for n in nodes_max_fill:
                    nodes_to_fill += n.behind_neighbors
                    nodes_to_fill.remove(n)
                print(f'New nodes to fill: {[n.name for n in nodes_to_fill]}')
                print(f'New p_max: {[getattr(n, max_load_type) for n in nodes_to_fill]}')

            else:
                print("FULL THIS ROUND")
                for i, n in enumerate(nodes_to_fill):
                    loads[n.name] += fills[i]

            # Update load_to_fill
            print(f"Old load to fill: {load_to_fill}")
            load_to_fill -= total_fill
            print(f"New load to fill: {load_to_fill}")

            # Store last information
            iteration_information['fill_factors'] = fill_factors
            iteration_information['total_load_filled_start'] = total_load_already_filled
            iteration_information['full_nodes_start'] = nodes_already_filled.copy()  # First store as we need them
            iteration_information['full_nodes_start_name'] = [n.name for n in nodes_already_filled.copy()]
            total_load_already_filled += total_fill
            nodes_already_filled += nodes_max_fill  # at the start of the iteration
            information_dict['iteration_' + str(iteration)] = iteration_information
            iteration += 1

        print(f"Loads: {loads}")
        print(f"sum: {sum(loads.values())}, load_to_fill: {load_to_fill}")
        return information_dict

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
        print('length new edge_paths: ', [len(edge_path) for edge_path in edge_paths])
        return edge_paths

    @staticmethod
    def _calculate_fill_factors(edge_property_paths: list, check=False) -> list:
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

                # print some values to check the results
                if check:
                    print(f'edge_property_paths: {edge_property_paths}')
                    print(f'numerator terms: {edge_property_paths[:i] + edge_property_paths[i + 1:]}')
                    print(f'denominator terms: {[comb for comb in combinations(edge_property_paths, len(edge_property_paths) - 1)]}')

        print(f"factors: {factors}")
        return factors
