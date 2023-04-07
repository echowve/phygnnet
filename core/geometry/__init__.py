from fenics import Point
from mshr import generate_mesh, Circle
import numpy as np
from enum import IntEnum
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch


class NodeType(IntEnum):
    inner = 0
    boundary = 1


def get_node_type_circle(pos):
    max_x = np.max(pos[:, 0])
    min_x = np.min(pos[:, 0])
    radius = (max_x - min_x)/2

    dist = np.linalg.norm(pos, axis=-1)
    on_boundary = np.isclose(dist, radius, atol=1e-4)

    node_type = np.ones((pos.shape[0], 1))
    node_type[on_boundary] = NodeType.boundary
    node_type[np.logical_not(on_boundary)] = NodeType.inner

    return np.squeeze(node_type)


class CircleMesh():

    node_type_ref = NodeType

    def __init__(self) -> None:
        self.transform = T.Compose([
            T.FaceToEdge(remove_faces=False),
            T.Cartesian(norm=False),
            T.Distance(norm=False)
            ])
        domain = Circle(Point(0.5, 0.5), 0.5)
        self.mesh = generate_mesh(domain, 70)
        self.pos = self.mesh.coordinates().astype(np.float32)
        self.faces = self.mesh.cells().astype(np.int64).T
        self.node_type = get_node_type_circle(self.pos).astype(np.int64)
        print("Node numbers: %d" % self.pos.shape[0])

    def getGraphData(self):
        graph = Data(pos=torch.as_tensor(self.pos),
                     face=torch.as_tensor(self.faces))
        graph = self.transform(graph)
        graph.num_nodes = graph.pos.shape[0]
        graph.node_type = torch.as_tensor(self.node_type)
        graph.label = 0
        return graph
