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

def create_IEEE33bus() -> RadialGraph:
    # Create a radial graph
    graph = RadialGraph()
    # Create nodes
    root = Root("Root")
    node1 = Node("Node 1", 100, 60) # p, q in kW, 3 phases (balanced load is assumed)
    node2 = Node("Node 2", 90, 40)
    node3 = Node("Node 3", 120, 80)
    node4 = Node("Node 4", 60, 30) # p, q in kW, 3 phases (balanced load is assumed)
    node5 = Node("Node 5", 60, 20)
    node6 = Node("Node 6", 200, 100)
    node7 = Node("Node 7", 200, 100) # p, q in kW, 3 phases (balanced load is assumed)
    node8 = Node("Node 8", 100, 50)
    node9 = Node("Node 9", 60, 20)
    node10 = Node("Node 10", 45, 30) # p, q in kW, 3 phases (balanced load is assumed)
    node11 = Node("Node 11", 60, 35)
    node12 = Node("Node 12", 60, 35)
    node13 = Node("Node 13", 120, 80) # p, q in kW, 3 phases (balanced load is assumed)
    node14 = Node("Node 14", 70, 30)
    node15 = Node("Node 15", 60, 20)
    node16 = Node("Node 16", 90, 30) # p, q in kW, 3 phases (balanced load is assumed)
    node17 = Node("Node 17", 100, 50)
    node18 = Node("Node 18", 90, 40)
    node19 = Node("Node 19", 90, 40) # p, q in kW, 3 phases (balanced load is assumed)
    node20 = Node("Node 20", 90, 40)
    node21 = Node("Node 21", 90, 40)
    node22 = Node("Node 22", 90, 50) # p, q in kW, 3 phases (balanced load is assumed)
    node23 = Node("Node 23", 420, 200)
    node24 = Node("Node 24", 420, 200)
    node25 = Node("Node 25", 60, 25)
    node26 = Node("Node 26", 60, 25) # p, q in kW, 3 phases (balanced load is assumed)
    node27 = Node("Node 27", 60, 20)
    node28 = Node("Node 28", 120, 70)
    node29 = Node("Node 29", 200, 600) # p, q in kW, 3 phases (balanced load is assumed)
    node30 = Node("Node 30", 150, 70)
    node31 = Node("Node 31", 210, 100)
    node32 = Node("Node 32", 60, 40) # p, q in kW, 3 phases (balanced load is assumed)
    # Add nodes to the graph
    graph.add_node(root)
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)
    graph.add_node(node5)
    graph.add_node(node6)
    graph.add_node(node7)
    graph.add_node(node8)
    graph.add_node(node9)
    graph.add_node(node10)
    graph.add_node(node11)
    graph.add_node(node12)
    graph.add_node(node13)
    graph.add_node(node14)
    graph.add_node(node15)
    graph.add_node(node16)
    graph.add_node(node17)
    graph.add_node(node18)
    graph.add_node(node19)
    graph.add_node(node20)
    graph.add_node(node21)
    graph.add_node(node22)
    graph.add_node(node23)
    graph.add_node(node24)
    graph.add_node(node25)
    graph.add_node(node26)
    graph.add_node(node27)
    graph.add_node(node28)
    graph.add_node(node29)
    graph.add_node(node30)
    graph.add_node(node31)
    graph.add_node(node32)
    # Add edges to the graph
    graph.add_edge(root, node1, 0.0922, 0.0470)
    graph.add_edge(node1, node2, 0.4930, 0.2511)
    graph.add_edge(node2, node3, 0.3660, 0.1864)
    graph.add_edge(node3, node4, 0.3811, 0.1941)
    graph.add_edge(node4, node5, 0.8190, 0.7070)
    graph.add_edge(node5, node6, 0.1872, 0.6188)
    graph.add_edge(node6, node7, 0.7114, 0.2351)
    graph.add_edge(node7, node8, 1.0300, 0.7400)
    graph.add_edge(node8, node9, 1.0440, 0.7400)
    graph.add_edge(node9, node10, 0.1966, 0.0650)
    graph.add_edge(node10, node11, 0.3744, 0.1238)
    graph.add_edge(node11, node12, 1.4680, 1.1550)
    graph.add_edge(node12, node13, 0.5416, 0.7129)
    graph.add_edge(node13, node14, 0.5910, 0.5260)
    graph.add_edge(node14, node15, 0.7463, 0.5450)
    graph.add_edge(node15, node16, 1.2890, 1.7210)
    graph.add_edge(node16, node17, 0.7320, 0.5740)
    graph.add_edge(node1, node18, 0.1640, 0.1565)
    graph.add_edge(node18, node19, 1.5042, 1.3554)
    graph.add_edge(node19, node20, 0.4095, 0.4784)
    graph.add_edge(node20, node21, 0.7089, 0.9373)
    graph.add_edge(node2, node22, 0.4512, 0.3083)
    graph.add_edge(node22, node23, 0.8980, 0.7091)
    #graph.add_edge(node24, node28, 0.8960, 0.7011)
    graph.add_edge(node23, node24, 0.8960, 0.7011)
    graph.add_edge(node5, node25, 0.2030, 0.1034)
    graph.add_edge(node25, node26, 0.2842, 0.1447)
    graph.add_edge(node26, node27, 1.0590, 0.9337)
    graph.add_edge(node27, node28, 0.8042, 0.7006)
    graph.add_edge(node28, node29, 0.5075, 0.2585)
    graph.add_edge(node29, node30, 0.9744, 0.9630)
    graph.add_edge(node30, node31, 0.3105, 0.3619)
    graph.add_edge(node31, node32, 0.3410, 0.5302)
    #graph.add_edge(node20, node7, 2.0000, 2.0000)
    #graph.add_edge(node8, node14, 2.0000, 2.0000)
    #graph.add_edge(node11, node21, 2.0000, 2.0000)
    #graph.add_edge(node17, node32, 0.5000, 0.5000)
    return graph