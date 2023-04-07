
import torch
from core.utils.tools import parse_config, rollout_error
from core.utils.tools import render_results, modelTester
from core.models import msgPassing
from core.geometry import CircleMesh
from functions import BurgesFunc as Func

out_ndim = 2
delta_t = 1e-3
ckptpath = 'checkpoint/simulator_%s.pth' % Func.func_name
device = torch.device(0)
test_steps = 500

func_main = Func(delta_t=delta_t)
bc = func_main.boundary_condition
ic = func_main.init_condition

mesh = CircleMesh()
graph = mesh.getGraphData()


model = msgPassing(message_passing_num=1, node_input_size=2+out_ndim,
                   edge_input_size=3, ndim=out_ndim, model_dir=ckptpath)
model.load_model()
model.to(device)
model.eval()

test_config = parse_config()
setattr(test_config, 'delta_t', delta_t)
setattr(test_config, 'ic', ic)
setattr(test_config, 'bc', bc)
setattr(test_config, 'model', model)
setattr(test_config, 'test_steps', test_steps)
setattr(test_config, 'NodeTypesRef', CircleMesh.node_type_ref)
setattr(test_config, 'graph', graph)
setattr(test_config, 'device', device)


predicted_results = modelTester(test_config)
real_results = []
for step in range(1, test_config.test_steps + 1):
    t = step * delta_t
    v1 = func_main.exact_solution(graph.pos, t)
    real_results.append(v1)
real_results = torch.stack(real_results, dim=0).cpu().numpy()


rollout_error(predicted_results, real_results)
render_results(predicted_results, real_results, graph)
