from typing import Any

import numpy as np
import torch
import utils.airflow.airflow_metrics as metrics
from lightning import LightningModule
import scipy
from .modules.mtl.weight_methods import WeightMethods
from src.models.modules.airflow_models.Airflow_NN import (
    AirflowNNMTLDecoderI,
    AirflowNNMTLDecoderII,
)
from src.models.modules.airflow_models.PointNet import (
    PointNetMTLDecoderI,
    PointNetMTLDecoderII,
)


class AirflowModule(LightningModule):

    def __init__(
        self,
        optimizer,
        scheduler,
        net: torch.nn.Module,
        lambdas: float = 1.0,
        compile: bool = True,
        loss_term: str = "",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.model = net
        # loss function
        self.criterion = torch.nn.MSELoss(reduction="none")
        self.lambdas = lambdas

        self.test_list = None
        self.data_dir = None
        self.out_coef_norm = None
        self.automatic_optimization = False
        self.loss_term = loss_term
        self.test_step_outputs = []
        self.decoder: str = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            data_module = self.trainer.datamodule
            self.test_list = data_module.test_list
            self.data_dir = data_module.data_dir
            self.out_coef_norm = torch.stack(data_module.mean_std[2:]).to(self.device)
            self.in_coef_norm = torch.stack(data_module.mean_std[:2]).to(self.device)
            if isinstance(self.model, AirflowNNMTLDecoderI) or isinstance(
                self.model, PointNetMTLDecoderI
            ):
                self.decoder = "decoderI"
                task_num = 2
            elif isinstance(self.model, AirflowNNMTLDecoderII) or isinstance(
                self.model, PointNetMTLDecoderII
            ):
                self.decoder = "decoderII"
                task_num = 5
            else:
                raise Exception(
                    "Model error only support decoderI or decoderII"
                )

            if self.loss_term in ["stch", "fairgrad", "famo"]:
                self.weight_method = WeightMethods(
                    method=self.loss_term, n_tasks=task_num, device=self.device
                )
            else:
                raise Exception(
                    "Must select a MTL optimization strategies from [stch, fairgrad, famo]"
                )
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(self, data: torch.Tensor):
        return self.model(data)

    def model_step(self, batch):
        pos, x, y, surf = batch
        pos, x, y, surf = pos[0], x[0], y[0], surf[0]

        data = torch.cat((pos, x), dim=1)
        out_surf, out_vol = self.model(data, surf)
        loss_surf = self.criterion(out_surf, y[surf, 2].unsqueeze(1))
        loss_vol = self.criterion(out_vol, y[~surf])
        return loss_vol, loss_surf, out_vol, out_surf

    def training_step(self, batch: Any, batch_idx: int):
        opt = self.optimizers()
        opt.zero_grad()
        loss_vol, loss_surf, _, _ = self.model_step(batch)
        loss_surf_var = loss_surf.mean(dim=0)
        loss_volume_var = loss_vol.mean(dim=0)
        if self.decoder == "decoderI":
            losses = torch.stack((loss_volume_var.mean(), loss_surf_var.mean()))
        elif self.decoder == "decoderII":
            losses = torch.concatenate(
                (loss_volume_var, loss_surf_var.mean().unsqueeze(0))
            )
        else:
            raise Exception("Model error only support decoderI or decoderII")
        loss, extra_outputs = self.weight_method.backward(
            losses=losses,
            shared_parameters=list(self.model.shared_parameters()),
            task_specific_parameters=list(
                self.model.task_specific_parameters()
            ),
            last_shared_parameters=list(self.model.last_shared_parameters()),
            epoch=self.current_epoch,
        )
        opt.step()
        if "famo" in self.loss_term:
            with torch.no_grad():
                loss_vol, loss_surf, _, _ = self.model_step(batch)
                loss_surf_var = loss_surf.mean(dim=0)
                loss_volume_var = loss_vol.mean(dim=0)
                if self.decoder == "decoderI":
                    losses = torch.stack(
                        (loss_volume_var.mean(), loss_surf_var.mean())
                    )
                elif self.decoder == "decoderII":
                    losses = torch.concatenate(
                        (loss_volume_var, loss_surf_var.mean().unsqueeze(0))
                    )
                self.weight_method.method.update(losses.detach())

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log_dict(
            {
                "train/loss_surf": loss_surf_var.mean(),
                "train/loss_volume": loss_volume_var.mean(),
                "train/loss": (
                    loss_surf_var.mean() + loss_volume_var.mean()
                    if loss is None
                    else loss.mean()
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            {
                "train/volume_ux": loss_volume_var[0],
                "train/volume_uy": loss_volume_var[1],
                "train/volume_p": loss_volume_var[2],
                "train/volume_nut": loss_volume_var[3],
                "train/surface_p": loss_surf_var,
            },
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        loss_vol, loss_surf, _, _ = self.model_step(batch)
        loss_surf_var = loss_surf.mean(dim=0)
        loss_volume_var = loss_vol.mean(dim=0)
        loss = loss_surf_var.mean() + loss_volume_var.mean()
        self.log_dict(
            {
                "val/loss_surf": loss_surf_var.mean(),
                "val/loss_volume": loss_volume_var.mean(),
                "val/loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log_dict(
            {
                "val/volume_ux": loss_volume_var[0],
                "val/volume_uy": loss_volume_var[1],
                "val/volume_p": loss_volume_var[2],
                "val/volume_nut": loss_volume_var[3],
                "val/surface_p": loss_surf_var,
            },
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        pos, x, y, surf = batch
        pos, x, y, surf = pos[0], x[0], y[0], surf[0]

        data = torch.cat((pos, x), dim=1)
        out_surf, out_vol = self.model(data, surf)
        loss_surf = self.criterion(out_surf, y[surf, 2].unsqueeze(1))
        loss_vol = self.criterion(out_vol, y[~surf])
        out = torch.zeros(
            (out_surf.shape[0] + out_vol.shape[0], out_vol.shape[1]),
            device=self.device,
        )
        out[surf, 2] = out_surf.squeeze(1)
        out[~surf] = out_vol

        loss_surf_var = loss_surf.mean(dim=0)
        loss_volume_var = loss_vol.mean(dim=0)

        true_coff, predict_coff, rel_err_force, rel_err_wss, rel_err_p = (
            metrics.compute_metric(
                out,
                self.test_list[batch_idx],
                surf.detach().cpu().numpy(),
                self.data_dir,
                self.out_coef_norm,
            )
        )
        self.test_step_outputs.append([true_coff, predict_coff])

        self.log_dict(
            {
                "test/volume_ux": loss_volume_var[0],
                "test/volume_uy": loss_volume_var[1],
                "test/volume_p": loss_volume_var[2],
                "test/volume_nut": loss_volume_var[3],
                "test/surface_p": loss_surf_var,
                "test/relative_error_force_x": rel_err_force[0],
                "test/relative_error_force_y": rel_err_force[1],
                "test/relative_error_wss_x": rel_err_wss[0],
                "test/relative_error_wss_y": rel_err_wss[1],
                "test/relative_error_p": rel_err_p,
            },
            on_step=True,
            on_epoch=True,
        )

    def on_test_epoch_end(self) -> None:
        spearman_coef_x = scipy.stats.spearmanr(
            np.array(self.test_step_outputs)[:, 0, 0],
            np.array(self.test_step_outputs)[:, 1, 0],
        )[0]
        spearman_coef_y = scipy.stats.spearmanr(
            np.array(self.test_step_outputs)[:, 0, 1],
            np.array(self.test_step_outputs)[:, 1, 1],
        )[0]

        self.log_dict(
            {
                "test/spearman_coef_x": spearman_coef_x,
                "test/spearman_coef_y": spearman_coef_y,
            },
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = self.hparams.optimizer(
            [
                {"params": self.model.parameters()},
            ]
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        return {"optimizer": optimizer}
