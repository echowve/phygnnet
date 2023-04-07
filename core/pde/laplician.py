from torch_geometric.data import Data
import torch
from collections import defaultdict
from tqdm import tqdm


class SolveWeightLST2d(object):
    '''
    laplician weightsh
    '''

    def __init__(self):

        def func(pos):

            x = pos[:, 0:1]
            y = pos[:, 1:2]
            v = torch.cat([x, y, x*y, x*x, y*y], dim=-1)
            return v

        def laplican_func(pos):

            v = torch.zeros((pos.shape[0], 5),
                            dtype=pos.dtype, device=pos.device)
            v[:, 3] = 2
            v[:, 4] = 2
            return v

        self.func = func
        self.laplican_func = laplican_func

    def __call__(self, data: Data):

        pos = data.pos
        edges = data.edge_index

        number_nodes = pos.shape[0]
        weights = torch.zeros_like(edges[1], dtype=torch.float)

        lap = self.laplican_func(pos)
        diff_ = self.func(pos[edges[1]] - pos[edges[0]])

        all_A_dict = defaultdict(list)
        all_B_dict = defaultdict(list)
        index_dict = defaultdict(list)

        for i in tqdm(range(number_nodes)):

            diff = diff_[edges[1] == i]
            laplician_value = lap[i:i+1]

            A = diff.t()
            neibor = A.shape[1]
            B = laplician_value.t()

            # all_B.append(B)
            all_A_dict[neibor].append(A)
            all_B_dict[neibor].append(B)
            index_dict[neibor].append(i)

        for n in all_A_dict.keys():
            A = torch.stack(all_A_dict[n], dim=0)
            B = torch.stack(all_B_dict[n], dim=0)
            index = index_dict[n]
            X = torch.linalg.lstsq(A, B).solution
            for i, w in enumerate(X):
                receiver = index[i]
                w = w.squeeze()
                weights[edges[1] == receiver] = w

        weights = weights.detach()
        return weights
