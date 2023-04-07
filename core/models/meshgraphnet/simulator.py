from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from core.utils.gnnutils import copy_geometric_data
import os


class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size,
                 ndim, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.ndim = ndim
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num,
                                             node_input_size=node_input_size,
                                             edge_input_size=edge_input_size,
                                             ndim=ndim)

    def forward(self, graph: Data):

        graph_last = copy_geometric_data(graph)
        node_type = torch.squeeze(graph.node_type).clone()
        one_hot = torch.nn.functional.one_hot(node_type, 2)
        graph.x = torch.cat([graph.x, one_hot], dim=-1)
        predicted_tmp = self.model(graph)
        v = predicted_tmp[:, :self.ndim] + graph_last.x[:, :self.ndim]

        return v

    def save_model(self, optimizer=None):
        path = os.path.dirname(self.model_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer_dict = {}
        optimizer_dict.update({'optimizer': optimizer.state_dict()})

        to_save_dict = {'model': self.state_dict()}
        to_save_dict.update(optimizer_dict)

        torch.save(to_save_dict, self.model_dir)

    def load_model(self, model_dir=None, optimizer=None):

        if model_dir is None:
            model_dir = self.model_dir

        tmp = torch.load(model_dir, map_location='cpu')
        # print(tmp)
        dicts = tmp['model']
        self.load_state_dict(dicts, strict=True)

        if optimizer is None:
            return
        optimizer.load_state_dict(tmp['optimizer'])
