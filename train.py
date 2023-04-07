
import torch
from torch.utils.tensorboard import SummaryWriter

from core.models import msgPassing
from core.geometry import CircleMesh
from core.utils.tools import parse_config, modelTrainer
from functions import BurgesFunc as Func
device = torch.device(0)

delta_t = 1e-3
out_ndim = 2
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name

func_main = Func(delta_t=delta_t)

bc = func_main.boundary_condition
ic = func_main.init_condition

model = msgPassing(message_passing_num=1, node_input_size=2+out_ndim,
                   edge_input_size=3, ndim=out_ndim, model_dir=ckptpath)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

mesh = CircleMesh()
graph = mesh.getGraphData().to(device)


train_config = parse_config()
writer = SummaryWriter('runs/%s' % Func.func_name)


setattr(train_config, 'pde', func_main.pde)
setattr(train_config, 'delta_t', delta_t)
setattr(train_config, 'ic', ic)
setattr(train_config, 'bc', bc)
setattr(train_config, 'graph', graph)
setattr(train_config, 'model', model)
setattr(train_config, 'optimizer', optimizer)
setattr(train_config, 'train_steps', 10)  # 训练的time step
setattr(train_config, 'epchoes', 50000)
setattr(train_config, 'NodeTypesRef', CircleMesh.node_type_ref)
setattr(train_config, 'writer', writer)


modelTrainer(train_config)
