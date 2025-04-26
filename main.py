from create_graphs import create_linear_graph, create_simple_branch_graph, create_graph, create_IEEE33bus_gen, create_IEEE33bus_load
from loadflowcalculator import LoadFlowCalculator
from cutcalculator import CutCalculator
import numpy as np
import warnings
import matplotlib.pyplot as plt

def get_P_from_lf_results(lf_results: dict) -> float:
    return max(list(lf_results['P_edges']), key=abs)

def get_Q_from_lf_results(lf_results: dict) -> float:
    return max(list(lf_results['Q_edges']), key=abs)

if __name__ == "__main__":
    # Create a radial graph

    all_generation = True  # Either all loads or all generation!
    # graph = create_IEEE33bus_load()
    if all_generation:
        graph = create_IEEE33bus_gen()
    else:
        graph = create_IEEE33bus_load()

    # Create loadflow calculator
    load_flow_calculator = LoadFlowCalculator(graph)

    # Create cut calculator
    cut_calculator = CutCalculator(graph, load_flow_calculator)

    # Set voltage parameters
    v_base = 12.66 * 1.0e3 / np.sqrt(3)  # 12.66 * 1.0e3
    v_min = 0.95 * v_base
    v_max = 1.05 * v_base
    v_properties = {'v_base': v_base, 'v_min': v_min, 'v_max': v_max}

    '''
    Calculate the sampled FOR
    '''
    # Create total active and reactive load linspaces

    n_points = 21
    max_active_power = sum([n.p_max for n in graph.nodes])
    max_reactive_power = sum([n.q_max for n in graph.nodes])

    p_totals = np.linspace(min(max_active_power, 0.0), max(0.0, max_active_power), n_points)
    q_totals = np.linspace(min(max_reactive_power, 0.0), max(0.0, max_reactive_power), n_points)

    # p_totals = np.linspace(0.0, max(0.0, max_active_power), n_points)
    # q_totals = np.linspace(min(max_reactive_power, 0.0), max(0.0, max_active_power), n_points)
    # '''

    # Allocate result lists
    Ps_transformer = []
    Qs_transformer = []

    Ps_transformer_no_loss = []
    Qs_transformer_no_loss = []

    count = 0  # keep track of how many load flow calculations we have done
    for i, p_total in enumerate(p_totals[::-1]):
        for q_total in q_totals[::-1]:
            # Calculate individual loads by means of bucket filling
            loads_p_dict = graph.apply_active_bucket_filling(p_total)
            loads_p = [load for node_name, load in loads_p_dict.items() if node_name != 'Root']

            loads_q_dict = graph.apply_reactive_bucket_filling(q_total)
            loads_q = [load for node_name, load in loads_q_dict.items() if node_name != 'Root']
            print("loads p, q: ", loads_p, loads_q)

            # Run load flows
            try:
                lf_lossless_results = load_flow_calculator.do_lossless_loadflow(loads_p, loads_q, v_base ** 2)
                lf_pp_results = load_flow_calculator.do_pp_loadflow(loads_p, loads_q, v_base ** 2)

                # Check if voltage constraint is violated and safe P and Q through the interconnection
                # Pandapower loadflow:
                v_nodes = lf_pp_results['v_nodes']
                if min(v_nodes) > v_min and max(v_nodes) < v_max:
                    Ps_transformer.append(get_P_from_lf_results(lf_pp_results))
                    Qs_transformer.append(get_Q_from_lf_results(lf_pp_results))

                # Lossless loadflow:
                v_nodes = lf_lossless_results['v_nodes']
                if min(v_nodes) > v_min and max(v_nodes) < v_max:
                    print(min(v_nodes), max(v_nodes))
                    print(v_min, v_max)
                    Ps_transformer_no_loss.append(get_P_from_lf_results(lf_lossless_results))
                    Qs_transformer_no_loss.append(get_Q_from_lf_results(lf_lossless_results))

            except:
                # If the pandapower load flow did not converge we skip this point
                warnings.warn("Pandapower loadflow did not converge")

            print(f'Progress sampling: {round(100 * (count + 1)/n_points ** 2, 2)}%')
            count += 1

    # '''
    '''
    Calculate voltage cuts
    
    The loss parameter is a parameter within [0, 1], which expresses for the estimation of the branch flows for
    the loss calculation how much of the weight is on the branch flows of the beginning/end of the bucket fill 
    iteration.
    
    Example:
        loss_parameter = 0: losses are calculated based on the branch flows at the end beginning of the iteration
        loss_parameter = 1: losses are calculated based on the branch flows at the beginning of the iteration
        loss_parameter = 0.5: branch flows are the average of the flows at the beginning/end of the iteration
    '''

    # '''
    slopes, intercepts = cut_calculator.get_cut_slopes_intercepts(v_properties, include_losses=False)
    slopes_losses, intercepts_losses = cut_calculator.get_cut_slopes_intercepts(v_properties,
                                                                                include_losses=True,
                                                                                loss_parameter=1.0 * all_generation)
    # '''

    '''
    Plotting
    '''
    # Cuts
    # '''
    cut_ids = slopes.keys()
    cut_ids_to_show = cut_ids
    cut_ids_losses_to_show = cut_ids
    # You can use these lists to select only a subset of the cuts to show
    # cut_ids_to_show = [[(1, 3), (2, 2), (2, 1)]]
    # cut_ids_losses_to_show = [[(1, 3), (2, 2), (2, 1)]]
    for cut_id in cut_ids:
        slope = slopes[cut_id]  # Var/W
        slope_loss = slopes_losses[cut_id]  # Var/W

        intercept = intercepts[cut_id]  # in Var
        intercept_loss = intercepts_losses[cut_id]  # in Var

        q_values = intercept + slope * 1000 * p_totals  # p_totals is in kW
        q_values_losses = intercept_loss + slope * 1000 * p_totals
        # plt.plot(p_totals, q_values/1000.0, 'r', label=str(cut_id), alpha=0.5)
        plt.plot(p_totals, q_values_losses / 1000.0, 'b', label=str(cut_id), alpha=0.1)

    # Samples
    plt.scatter(Ps_transformer, Qs_transformer, color='b', label='Full Distflow')
    plt.scatter(Ps_transformer_no_loss, Qs_transformer_no_loss, color='r', label='Linear Distflow')



    plt.legend()
    # '''
    plt.ylim(min(0, 1.1 * max_reactive_power), max(0, 1.1 * max_reactive_power))
    plt.show()
