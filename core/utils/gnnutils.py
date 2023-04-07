
from torch_geometric.data import Data


def decompose_graph(graph):

    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key == "x":
            x = graph.x
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)


def copy_geometric_data(graph):
    graph_info = {k: graph[k] for k in graph.keys}
    ret = Data(**graph_info)

    return ret
