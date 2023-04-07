import torch
from torch_geometric.data import Data
from collections import defaultdict


class SolveGradientsLST(object):
    '''
    gradients weights
    '''

    def __init__(self):
        self.w_dict = defaultdict(list)
        self.index_dict = defaultdict(list)
        self.node_index_dict = defaultdict(list)

    def solve_single_varible(self, graph: Data, u):

        pos = graph.pos
        edges = graph.edge_index

        u_differ = u[edges[0]] - u[edges[1]]
        dydx = pos[edges[0]] - pos[edges[1]]

        dudxdy = torch.empty_like(pos)

        if len(self.w_dict) == 0:

            for node_index in range(pos.shape[0]):
                u_edge_index = torch.where(edges[1] == node_index)[0]
                u_ = u_differ[u_edge_index]
                number_u_ = u_.shape[0]
                A = dydx[edges[1] == node_index].detach()
                W = torch.inverse(A.t().mm(A)).mm(A.t())
                self.w_dict[number_u_].append(W)
                self.index_dict[number_u_].append(u_edge_index)
                self.node_index_dict[number_u_].append(node_index)
                dudxdy[node_index] = W.mm(u_).squeeze()

        else:

            for number_u_ in self.w_dict.keys():

                W = torch.stack(self.w_dict[number_u_])
                U = u_differ[torch.cat(
                    self.index_dict[number_u_], dim=-1)].reshape(
                        (W.shape[0], -1, 1)
                        )

                index = self.node_index_dict[number_u_]
                value = W.bmm(U).reshape((-1, pos.shape[1]))
                dudxdy[index] = value

        return dudxdy

    def __call__(self, data, predicted):

        if isinstance(data, torch.Tensor):
            data = Data(pos=data)
        ndim = predicted.shape[1]

        gradients = []
        for i in range(ndim):
            gradients.append(self.solve_single_varible(
                data, predicted[:, i:i+1]))

        return gradients
