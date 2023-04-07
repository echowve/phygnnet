import torch
from core.pde import laplician, grad


class BurgesFunc():

    R = 80
    func_name = 'burgers'

    def __init__(self, delta_t) -> None:
        self.delta_t = delta_t
        self.laplicianop = laplician()
        self.gradop = grad()

    @classmethod
    def init_condition(cls, pos):
        x = pos[:, 0:1]
        y = pos[:, 1:2]

        item = cls.R*(-4*x + 4*y)/32

        u = 3/4 - 1/(4*(1 + torch.exp(item)))
        v = 3/4 + 1/(4*(1 + torch.exp(item)))
        return torch.cat((u, v), dim=-1)

    @classmethod
    def boundary_condition(cls, pos, t):
        x = pos[:, 0:1]
        y = pos[:, 1:2]

        item = cls.R*(-t - 4*x + 4*y)/32
        u = 3/4 - 1/(4*(1 + torch.exp(item)))
        v = 3/4 + 1/(4*(1 + torch.exp(item)))

        return torch.cat((u, v), dim=-1)

    # 解析解
    def exact_solution(cls, pos, t):
        return cls.boundary_condition(pos, t)

    def pde(self, graph, values_last, values_this, **argv):

        u_this = values_this[:, 0:1]
        v_this = values_this[:, 1:2]

        u_last = values_last[:, 0:1]
        v_last = values_last[:, 1:2]

        dudt = (u_this - u_last)/self.delta_t
        dvdt = (v_this - v_last)/self.delta_t

        gradvalue = self.gradop(graph, values_this)
        gradu = gradvalue[0]
        gradv = gradvalue[1]

        lapvalue = self.laplicianop(graph, values_this)
        lapu = lapvalue[:, 0:1]
        lapv = lapvalue[:, 1:2]

        lossu = dudt + (gradu*values_this).sum(dim=-1,
                                               keepdim=True) - 1/self.R*lapu
        lossv = dvdt + (gradv*values_this).sum(dim=-1,
                                               keepdim=True) - 1/self.R*lapv

        return torch.cat([lossu, lossv], axis=1)
