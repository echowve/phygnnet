import json
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Config:
    def __init__(self) -> None:
        pass

    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value


def parse_config(file='config.json'):
    configs = Config()
    if not os.path.exists(file):
        return configs
    with open(file, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            config = Config()
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    config.setattr(k1, v1)
            else:
                raise TypeError
            configs[k] = config
    return configs[k]


def modelTrainer(config):

    delta_t = config.delta_t
    model = config.model
    graph = config.graph

    best_loss = np.inf
    for epcho in range(1, config.epchoes+1):
        graph.x = config.ic(graph.pos)

        begin_time = 0
        total_steps_loss = 0

        on_boundary = torch.squeeze(
            graph.node_type == config.NodeTypesRef.boundary)
        config.optimizer.zero_grad()

        losses = {}
        for step in range(1, config.train_steps+1):

            this_time = begin_time + delta_t*step

            value_last = graph.x.detach().clone()
            boundary_value = config.bc(graph.pos, this_time)
            graph.x[on_boundary] = boundary_value
            predicted = model(graph)
            # hard boundary
            predicted[on_boundary] = boundary_value[on_boundary]

            pde_loss = config.pde(graph, value_last, predicted)
            pde_loss[on_boundary] = 0
            loss = torch.norm(pde_loss)/pde_loss.numel()

            loss.backward()
            graph.x = predicted.detach()

            losses.update({"step%d" % step: loss.detach()})
            total_steps_loss += loss.item()/config.train_steps

        config.writer.add_scalars("loss", losses, epcho)
        config.writer.add_scalar("total_steps_loss", total_steps_loss, epcho)
        config.writer.flush()
        config.optimizer.step()

        if total_steps_loss < best_loss:
            best_loss = total_steps_loss
            model.save_model(config.optimizer)
            print('mode saved at loss: %.4e' % best_loss)

    print('Train complete! Model saved to %s' % config.model.model_dir)


@torch.no_grad()
def modelTester(config):

    delta_t = config.delta_t
    model = config.model.to(config.device)
    config.graph = config.graph.to(config.device)

    test_steps = config.test_steps

    config.graph.x = config.ic(config.graph.pos)

    begin_time = 0
    test_results = []
    on_boundary = torch.squeeze(
        config.graph.node_type == config.NodeTypesRef.boundary)

    def predictor(model, graph, step):
        this_time = begin_time + delta_t * step

        boundary_value = config.bc(graph.pos, this_time)
        graph.x[on_boundary] = boundary_value
        predicted = model(graph)
        predicted[on_boundary] = boundary_value[on_boundary]

        return predicted

    for step in tqdm(range(1, test_steps + 1)):
        v = predictor(model, config.graph, step)
        config.graph.x = v.detach()
        v = v.clone().cpu().numpy()
        test_results.append(v)

    v = np.stack(test_results, axis=0)
    return v


def rollout_error(predicteds, targets):

    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1),
                   axis=0)/np.arange(1, number_len + 1))

    for show_step in range(0, 1000000):
        if show_step in [1, 100, 200, 300, 400, 500] and show_step < number_len:
            print(
                'testing rmse  @ step %d loss: %.2e' %
                (show_step, loss[show_step]))
        else:
            continue

    return loss


def render_results(predicteds, exacts, graph):
    test_begin_step = 0
    if predicteds is None:
        return
    total_test_steps = predicteds.shape[0]
    pos = graph.pos.cpu().numpy()

    u_max = np.max(predicteds[:, :, 0])
    u_min = np.min(predicteds[:, :, 0])

    os.makedirs('images', exist_ok=True)
    x = pos[:, 0]
    y = pos[:, 1]

    for index_ in tqdm(range(total_test_steps)):
        if index_ % 20 != 0:
            continue
        predicted = predicteds[index_]
        real = exacts[index_]

        data_index = 0

        c = predicted[:, data_index:data_index+1]
        cr = real[:, data_index:data_index+1]

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        b = axes.scatter(x, y, c=cr,  vmin=u_min, vmax=u_max, cmap="plasma")
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%du_true.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        b = axes.scatter(x, y, c=c,  vmin=u_min, vmax=u_max, cmap="plasma")
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%du_predict.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()

        diff = c - cr
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))

        b = axes.scatter(x, y, c=diff, vmin=-0.01, vmax=0.01, cmap='bwr')
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%du_diff.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()

        data_index = 1
        v_max = np.max(predicted[:,  data_index])
        v_min = np.min(predicted[:,  data_index])

        c = predicted[:, data_index:data_index+1]
        cr = real[:, data_index:data_index+1]

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        b = axes.scatter(x, y, c=cr,  vmin=v_min, vmax=v_max, cmap="plasma")
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%dv_true.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        b = axes.scatter(x, y, c=c,  vmin=v_min, vmax=v_max, cmap="plasma")
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%dv_predict.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()

        diff = c - cr
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))

        b = axes.scatter(x, y, c=diff, vmin=-0.01, vmax=0.01, cmap='bwr')
        fig.colorbar(b, ax=axes)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig('images/result%dv_diff.png' %
                    (test_begin_step+index_), bbox_inches='tight')
        plt.close()
