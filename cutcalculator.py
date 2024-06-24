from radialgraph import RadialGraph
from loadflowcalculator import LoadFlowCalculator
from typing import Tuple, List, Dict


class CutCalculator:
    def __init__(self, radial_graph: RadialGraph, load_flow_calculator: LoadFlowCalculator):
        self.radial_graph = radial_graph
        self.load_flow_calculator = load_flow_calculator

    def get_cut_slopes_intercepts(self, v_properties: dict, include_losses: bool, loss_parameter: float = 0.5) -> Tuple[Dict]:
        # TODO: remove redundant print statements
        """
        This function calculates the slopes and the intercepts of the linear voltage cuts
        The expressions for these values can be found in the paper

        For every iteration (i, j) of the bucket fill algorithm in active, reactive power we get a cut

        Loosely spoken, we need to calculate:
        - slope:
            * numerator, depending on the fill factors of active power
            * denominator, depending on the fill factors of the reactive power
        - intercept:
            * a voltage term independent of the bucket fill results
            * an active power term depending on the loads at te begining of the iteration
            * an active power term depending on the fill factors of the loads that are being filled that iteration
            * (an active power term depending on the losses)
            * a reactive power term depending on the loads at te begining of the iteration
            * a reactive power term depending on the fill factors of the loads that are being filled that iteration
            * (a reactive power term depending on the losses)

        The loss parameter is a parameter within [0, 1], which expresses for the estimation of the branch flows for
        the loss calculation how much of the weight is on the branch flows of the beginning/end of the bucket fill
        iteration.

        Example:
            loss_parameter = 0: losses are calculated based on the branch flows at the beginning of the iteration
            loss_parameter = 1: losses are calculated based on the branch flows at the end of the iteration
            loss_parameter = 0.5: branch flows are the weighted sum of the flows at the beginning/end of the iteration
        """
        # First calculate the voltage term
        v_term_intercept = self._calculate_voltage_intercept_term(v_properties)

        # Calculate the bucket filling information for p and q
        total_p_load = sum([n.p_max for n in self.radial_graph.nodes])
        total_q_load = sum([n.q_max for n in self.radial_graph.nodes])
        active_info = self.radial_graph.get_active_bucket_filling_info(total_p_load)
        reactive_info = self.radial_graph.get_reactive_bucket_filling_info(total_q_load)

        print("Bucket fill information:")
        print("Active power: ", active_info)
        print("Reactive power: ", reactive_info)

        # Get active power terms (except for the losses)
        # The p terms are independent of the q flows and vice versa (this does not hold for the loss terms)
        active_power_term_tuple = self._calculate_active_power_terms(active_info)
        reactive_power_term_tuple = self._calculate_reactive_power_terms(reactive_info)
        numerators_slope, p_start_terms_intercept, p_fill_terms_intercept = active_power_term_tuple
        denominators_slope, q_start_terms_intercept, q_fill_terms_intercept = reactive_power_term_tuple

        # Calculate slopes and intercepts
        slopes = dict()
        intercepts = dict()
        for i in range(len(active_info)):
            for j in range(len(reactive_info)):
                slopes[(i + 1, j + 1)] = - numerators_slope[i]/denominators_slope[j]
                intercepts[(i + 1, j + 1)] = 3/(denominators_slope[j]) * (
                        v_term_intercept + 1.0e3/3 * (
                        p_start_terms_intercept[i] - p_fill_terms_intercept[i] + q_start_terms_intercept[j]
                        - q_fill_terms_intercept[j])
                )

        # Add losses if necessary
        if include_losses:
            assert loss_parameter >= 0, "Please provide a loss parameter >= 0"
            assert loss_parameter <= 1, "Please provide a loss parameter <= 1"
            p_loss_term_dict, q_loss_term_dict = self._calculate_loss_terms(active_info,
                                                                            reactive_info,
                                                                            v_properties,
                                                                            loss_parameter)
            for cut_id in intercepts:
                j = cut_id[1]
                denominator_slope = denominators_slope[j - 1]
                intercepts[cut_id] += 3/denominator_slope * (1000.0/3) ** 2 * (p_loss_term_dict[cut_id] +
                                                                               q_loss_term_dict[cut_id])

        return slopes, intercepts

    @staticmethod
    def _calculate_voltage_intercept_term(v_properties: dict) -> float:
        return (v_properties['v_base'] ** 2 - v_properties['v_min'] ** 2) / 2

    def _calculate_active_power_terms(self, active_bucket_filling_information: dict) -> Tuple[List[float]]:
        print("Calculate P terms")
        return self._calculate_power_terms(active_bucket_filling_information, 'resistance')

    def _calculate_reactive_power_terms(self, reactive_bucket_filling_information: dict) -> Tuple[List[float]]:
        print("Calculate Q terms")
        return self._calculate_power_terms(reactive_bucket_filling_information, 'reactance')

    def _calculate_power_terms(self, bucket_filling_information: dict, edge_prop_type: str) -> Tuple[List[float]]:
        # TODO: remove redundant print statements
        slope_terms = []
        start_terms_intercept = []
        fill_terms_intercept = []
        partitioning = []

        for i in range(1, len(bucket_filling_information) + 1):  # number of iterations <= number of nodes
            print(f"Iteration: {i}")
            print(bucket_filling_information)
            iteration_info = bucket_filling_information['iteration_' + str(i)]
            to_fill_nodes = iteration_info['to_fill_nodes']
            loads_start = iteration_info['loads_start']
            fill_factors = iteration_info['fill_factors']
            total_load_filled_start = iteration_info['total_load_filled_start']

            partitioning.append(total_load_filled_start)

            destination_node = to_fill_nodes[0]  # the exact node does not matter as the voltage is equal
            print(f"destination: {destination_node.name}")
            print(f"fill factors: {fill_factors}")
            print(f"loads_start: {loads_start}")
            print(f"total_load_filled_start: {total_load_filled_start}")

            slope_term = 0.0
            start_term_intercept = 0.0
            fill_term_intercept = 0.0

            # Loop 1 for outer sum over the path
            for edge in destination_node.get_edge_path():
                z = getattr(edge, edge_prop_type)  # either r or x

                # Take the tree of the second node, as the power through the edge is defined as P - sum over
                # all the loads not behind that branch (not in the tree of node 2)
                print(edge.node2.name)
                tree = edge.node2.get_nodes_in_tree()
                not_tree = [n for n in self.radial_graph.nodes if n.name not in [m.name for m in tree]]
                # not_tree = list(set(graph.nodes).difference(set(tree)))
                print(f"Tree: {[n.name for n in tree]}")
                print(f"Nodes: {[n.name for n in self.radial_graph.nodes]}")
                print(f"Not Tree: {[n.name for n in not_tree]}")

                print(f"Nodes: {[n.p_max for n in self.radial_graph.nodes]}")
                print(f"Not Tree: {[n.p_max for n in not_tree]}")

                # Calculate sum over trees
                sum_fill_factors_in_tree = 0.0
                # Loop 2 for inner sum over the nodes in the tree
                for n in tree:
                    if n.name in [m.name for m in to_fill_nodes]:
                        print(f"f taken into acount of: {n.name, fill_factors[to_fill_nodes.index(n)]}")
                        sum_fill_factors_in_tree += fill_factors[to_fill_nodes.index(n)]

                sum_start_not_tree = 0.0
                # Loop 2 for inner sum over the nodes in the not tree
                for n in not_tree:
                    sum_start_not_tree += loads_start[n.name]

                sum_fill_factors_in_not_tree = 1.0 - sum_fill_factors_in_tree

                # Calculate terms
                slope_term += z * sum_fill_factors_in_tree
                # Note the P_prev does not change along the path
                start_term_intercept += z * sum_start_not_tree
                fill_term_intercept += z * sum_fill_factors_in_not_tree * total_load_filled_start
            print(f"slope_term: {slope_term}")
            print(f"start_term_intercept: {start_term_intercept}")
            print(f"fill_term_intercept: {fill_term_intercept}")

            slope_terms.append(slope_term)
            start_terms_intercept.append(start_term_intercept)
            fill_terms_intercept.append(fill_term_intercept)

        return slope_terms, start_terms_intercept, fill_terms_intercept

    def _calculate_loss_terms(self,
                             active_bucket_filling_information: dict,
                             reactive_bucket_filling_information: dict,
                             v_properties: dict,
                             loss_parameter: float) -> Tuple[Dict[Tuple, float]]:

        # TODO: remove redundant print statements
        active_power_loss_term_dict = dict()
        reactive_power_loss_term_dict = dict()

        # Define help map to map calculated losses to the node behind that edge
        edge_nr_node2_name_map = {e_nr: e.node2.name for e_nr, e in enumerate(self.radial_graph.edges)}

        # Calculate loss terms with nested for loop, as the losses in P depend on both P and Q etc.
        print("Calculate Loss terms:")
        for i in range(1, len(active_bucket_filling_information) + 1):  # number of iterations <= number of node
            print(f"Iteration p: {i}")
            active_iteration_info = active_bucket_filling_information['iteration_' + str(i)]
            active_destination_node = active_iteration_info['to_fill_nodes'][0]
            active_loads_start = active_iteration_info['loads_start']
            active_loads_end = active_iteration_info['loads_end']
            loss_parameter = 0.5
            active_loads = {n_name: (loss_parameter * active_loads_start[n_name] + (1 - loss_parameter) * active_loads_end[n_name]) for n_name in active_loads_start}
            # active_loads = active_iteration_info['loads_start']
            loads_p = list(
                {n_name: active_loads[n_name] for n_name in active_loads if n_name != 'Root'}.values())
            for j in range(1, len(reactive_bucket_filling_information) + 1):  # number of iterations <= number of nodes
                print(f"Iteration q: {j}")
                reactive_iteration_info = reactive_bucket_filling_information['iteration_' + str(j)]
                reactive_destination_node = reactive_iteration_info['to_fill_nodes'][0]
                reactive_loads_start = reactive_iteration_info['loads_start']
                reactive_loads_end = reactive_iteration_info['loads_end']
                reactive_loads = {n_name: (loss_parameter * active_loads_start[n_name] + (1 - loss_parameter) * active_loads_end[n_name]) for n_name in
                                reactive_loads_start}
                # reactive_loads = reactive_iteration_info['loads_start']
                loads_q = list({n_name: reactive_loads[n_name] for n_name in reactive_loads if
                                n_name != 'Root'}.values())

                print(f"active_destination_node: {active_destination_node.name}")
                print(f"reactive_destination_node: {reactive_destination_node.name}")

                # Calculate powerflow and losses based on loads from previous iteration
                print("check loads:", loads_p, loads_q)
                v_base = v_properties['v_base']
                lf_results = self.load_flow_calculator.do_lossless_loadflow(loads_p, loads_q, v_base**2)
                edge_flow_losses_per_ohm = [(P ** 2 + Q ** 2) / v_base ** 2 for P, Q in
                                            zip(lf_results['P_edges'], lf_results['Q_edges'])]
                assert len(edge_flow_losses_per_ohm) == len(self.radial_graph.edges), "not the same number of losses for every edge"
                node2_flow_losses_per_ohm_map = {edge_nr_node2_name_map[e_nr]: loss for e_nr, loss in
                                                 enumerate(edge_flow_losses_per_ohm)}

                print("Losses per ohm:")
                print(lf_results)
                print(node2_flow_losses_per_ohm_map)

                # Loop 1 for outer sum over the path
                p_loss_term_intercept = 0.0
                q_loss_term_intercept = 0.0

                for edge in active_destination_node.get_edge_path():
                    r = edge.resistance
                    tree = edge.node2.get_nodes_in_tree()
                    not_tree = [n for n in self.radial_graph.nodes if n.name not in [m.name for m in tree]]
                    # not_tree = list(set(graph.nodes).difference(set(tree)))
                    print(f"Tree: {[n.name for n in tree]}")
                    print(f"Nodes: {[n.name for n in self.radial_graph.nodes]}")
                    print(f"Not Tree: {[n.name for n in not_tree]}")

                    print(f"Nodes: {[n.p_max for n in self.radial_graph.nodes]}")
                    print(f"Not Tree: {[n.p_max for n in not_tree]}")
                    # Loop 2 for inner sum over the nodes in the not tree
                    for n in not_tree:
                        if n.name != 'Root':
                            p_loss_term_intercept += r ** 2 * node2_flow_losses_per_ohm_map[
                                n.name]  # rP = r^2 P_per_ohm

                for edge in reactive_destination_node.get_edge_path():
                    x = edge.reactance
                    tree = edge.node2.get_nodes_in_tree()
                    not_tree = [n for n in self.radial_graph.nodes if n.name not in [m.name for m in tree]]
                    # Loop 2 for inner sum over the nodes in the not tree
                    for n in not_tree:
                        if n.name != 'Root':
                            q_loss_term_intercept += x ** 2 * node2_flow_losses_per_ohm_map[
                                n.name]  # xQ = x^2 Q_per_ohm

                print(f"p_loss_term_intercept: {p_loss_term_intercept}")
                print(f"q_loss_term_intercept: {q_loss_term_intercept}")

                active_power_loss_term_dict[(i, j)] = p_loss_term_intercept
                reactive_power_loss_term_dict[(i, j)] = q_loss_term_intercept

        return active_power_loss_term_dict, reactive_power_loss_term_dict


