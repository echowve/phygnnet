import torch
import torch.nn as nn

from torch_scatter import scatter_add
from core.utils.gnnutils import decompose_graph


class EdgeBlock(nn.Module):

    def __init__(self, sender_func=None, receiver_func=None, edge_func=None):

        super(EdgeBlock, self).__init__()

        self.sender_func = sender_func
        self.receiver_func = receiver_func
        self.edge_func = edge_func

    def forward(self, graph):

        node_attr, edge_index, edge_attr, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        v = self.sender_func(senders_attr)
        v = self.receiver_func(receivers_attr) + v
        v = self.edge_func(edge_attr) + v

        graph.edge_attr = v
        return graph


class NodeBlock(nn.Module):

    def __init__(self, custom_func):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr
        nodes_to_collect = []

        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(
            edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)
        graph.x = x
        return graph
