from .gradients import SolveGradientsLST as grad
from .laplician import SolveWeightLST2d
import torch
from torch_geometric.utils import get_laplacian


__all__ = ['grad', 'laplician']


class laplician():
    def __init__(self) -> None:
        self.L_matrix_dict = {}
        self.solver2d = SolveWeightLST2d()

    def __call__(self, graph, values):

        L = self.L_matrix_dict.get(graph.label, None)
        if L is None:

            weights_t = self.solver2d(data=graph)
            index = torch.stack(
                [graph.edge_index[1], graph.edge_index[0]], dim=0)
            edges, weight = get_laplacian(index, weights_t)
            L = torch.sparse.FloatTensor(edges, weight)
            self.L_matrix_dict.setdefault(graph.label, L)

        return -L.mm(values)
