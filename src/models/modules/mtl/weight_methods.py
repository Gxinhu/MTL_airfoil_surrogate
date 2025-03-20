from abc import abstractmethod
from typing import List, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares


EPS = 1e-8  # for numerical stability


class WeightMethods:
    def __init__(
        self, method: str, n_tasks: int, device: torch.device, **kwargs
    ):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __ceil__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm=1.0):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class FAMO(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 1e-5,
        w_lr: float = 0.025,
        task_weights: Union[List[float], torch.Tensor] = None,
        max_norm: float = 1.0,
    ):
        super().__init__(n_tasks, device=device)
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor(
            [0.0] * n_tasks, device=device, requires_grad=True
        )
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses, **kwargs):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z, "logits": self.w.detach().clone()}

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - (
            curr_loss - self.min_losses + 1e-8
        ).log()
        with torch.enable_grad():
            d = torch.autograd.grad(
                F.softmax(self.w, -1), self.w, grad_outputs=delta.detach()
            )[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()


class STCH(WeightMethod):
    r"""STCH.

    This method is proposed in `Smooth Tchebycheff Scalarization for Multi-Objective Optimization (ICML 2024) <https://openreview.net/forum?id=m4dO5L6eCp>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Xi-L/STCH/tree/main/STCH_MTL>`_.

    """

    def __init__(self, device, mu=1, n_tasks=2):
        super().__init__(n_tasks, device=device)
        self.device = device
        self.mu = mu
        self.task_num = n_tasks
        self.init_param()

    def init_param(self):
        self.step = 0
        self.nadir_vector = None

        self.average_loss = 0.0
        self.average_loss_count = 0
        self.warmup_epoch = 4

    def get_weighted_loss(self, losses, **kwargs):
        self.step += 1
        mu = self.mu
        warmup_epoch = self.warmup_epoch

        batch_weight = np.ones(len(losses))
        self.epoch = kwargs["epoch"]

        if self.epoch < warmup_epoch:
            loss = torch.mul(
                torch.log(losses + 1e-20),
                torch.ones_like(losses).to(self.device),
            ).sum()
            return loss, batch_weight
        elif self.epoch == warmup_epoch:
            loss = torch.mul(
                torch.log(losses + 1e-20),
                torch.ones_like(losses).to(self.device),
            ).sum()
            self.average_loss += losses.detach()
            self.average_loss_count += 1

            return loss, batch_weight
        else:
            if self.nadir_vector is None:
                self.nadir_vector = self.average_loss / self.average_loss_count
                print(self.nadir_vector)

            losses = torch.log(losses / self.nadir_vector + 1e-20)
            max_term = torch.max(losses.data).detach()
            reg_losses = losses - max_term

            loss = (
                mu
                * torch.log(torch.sum(torch.exp(reg_losses / mu)))
                * self.task_num
            )
            return loss, batch_weight


class FairGrad(WeightMethod):
    def __init__(self, n_tasks, device: torch.device, alpha=1.0, max_norm=1.0):
        super().__init__(n_tasks, device=device)
        self.alpha = alpha
        self.max_norm = max_norm

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks - 1:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.fairgrad(grads, alpha=self.alpha)
        self.overwrite_grad(shared_parameters, g, grad_dims)
        return GTG, w_cpu

    def fairgrad(self, grads, alpha=1.0):
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]

        x_start = np.ones(self.n_tasks) / self.n_tasks
        A = GG.data.cpu().numpy()

        def objfn(x):
            # return np.power(np.dot(A, x), alpha) - 1 / x
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        g = (grads * ww.view(1, -1)).sum(1)
        return g, GG.data.cpu().numpy(), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
        self,
        losses: torch.Tensor,
        parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        GTG, w = self.get_weighted_loss(losses, shared_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return None, {
            "GTG": GTG,
            "weights": w,
        }  # NOTE: to align with all other weight methods


METHODS = {
    "famo": FAMO,
    "stch": STCH,
    "fairgrad": FairGrad,
}
