import torch.nn as nn
from .blocks import EdgeBlock, NodeBlock
from core.utils.gnnutils import decompose_graph, copy_geometric_data


def build_mlp(in_size, hidden_size, out_size, lay_norm=False):

    module = nn.Sequential(
        nn.Linear(in_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, out_size)
        )

    if lay_norm:
        return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):

    def __init__(self,
                 edge_input_size=128,
                 node_input_size=128,
                 hidden_size=64):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph_input):

        graph = copy_geometric_data(graph_input)
        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        graph.x = node_
        graph.edge_attr = edge_
        return graph


class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()

        sender_func = build_mlp(hidden_size, hidden_size, hidden_size)
        receiver_func = build_mlp(hidden_size, hidden_size, hidden_size)
        edge_func = build_mlp(hidden_size, hidden_size, hidden_size)

        nb_custom_func = build_mlp(2*hidden_size, hidden_size, hidden_size)
        self.eb_module = EdgeBlock(
            sender_func=sender_func,
            receiver_func=receiver_func,
            edge_func=edge_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):

        node_attr, edge_attr = graph.x.clone(), graph.edge_attr.clone()
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        graph.x += node_attr
        graph.edge_attr += edge_attr

        return graph


class Decoder(nn.Module):

    def __init__(self, hidden_size=64, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num,
                 node_input_size,
                 edge_input_size,
                 hidden_size=128,
                 ndim=2):

        super(EncoderProcesserDecoder, self).__init__()
        self.message_passing_num = message_passing_num
        self.encoder = Encoder(edge_input_size=edge_input_size,
                               node_input_size=node_input_size,
                               hidden_size=hidden_size)
        processer_list = []
        for i in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(hidden_size=hidden_size, output_size=ndim)

    def forward(self, graph_input):
        graph = self.encoder(graph_input)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
