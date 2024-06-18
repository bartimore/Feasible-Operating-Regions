from radialgraph import Node, Root, RadialGraph


def create_graph() -> RadialGraph:
    # Create a radial graph
    graph = RadialGraph()

    # Create nodes
    root = Root("Root")
    node1 = Node("Node 1", 40, 20)  # p, q in kW, 3 phases (balanced load is assumed)
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
    r_unit = 0.14  # Ohm
    x_unit = 0.04  # Ohm
    graph.add_edge(root, node1, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node1, node2, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node1, node3, 2 * r_unit, 2 * x_unit)
    graph.add_edge(node1, node4, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node2, node5, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node3, node6, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node4, node7, 1 * r_unit, 1 * x_unit)

    return graph


def create_simple_branch_graph() -> RadialGraph:
    # Create a radial graph
    graph = RadialGraph()

    # Create nodes
    root = Root("Root")
    node1 = Node("Node 1", 40, 20)  # p, q in kW, 3 phases (balanced load is assumed)
    node2 = Node("Node 2", 30, 15)
    node3 = Node("Node 3", 20, 10)
    node4 = Node("Node 4", 50, 25)

    # Add nodes to the graph
    graph.add_node(root)
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    # Add edges to the graph
    r_unit = 0.14 / 2  # Ohm
    x_unit = 0.04 / 2  # Ohm
    graph.add_edge(root, node1, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node1, node2, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node1, node3, 2 * r_unit, 2 * x_unit)
    graph.add_edge(node1, node4, 1 * r_unit, 1 * x_unit)

    return graph


def create_linear_graph() -> RadialGraph:
    # Create a radial graph
    graph = RadialGraph()

    # Create nodes
    root = Root("Root")
    node1 = Node("Node 1", 40, 20)  # p, q in kW, 3 phases (balanced load is assumed)
    node2 = Node("Node 2", 20, 10)
    node3 = Node("Node 3", 20, 10)

    # Add nodes to the graph
    graph.add_node(root)
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)

    # Add edges to the graph
    r_unit = 0.14  # Ohm
    x_unit = 0.04  # Ohm
    graph.add_edge(root, node1, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node1, node2, 1 * r_unit, 1 * x_unit)
    graph.add_edge(node2, node3, 1 * r_unit, 1 * x_unit)

    return graph