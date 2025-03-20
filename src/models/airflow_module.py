from typing import Any

import numpy as np
import torch
import utils.airflow.airflow_metrics as metrics
from lightning import LightningModule
import scipy


class AirflowModule(LightningModule):

    def __init__(
        self,
        optimizer,
        scheduler,
        net: torch.nn.Module,
        lambdas: float = 1.0,
        is_whole: bool = True,
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
        self.test_step_outputs = []
        self.loss_term = loss_term

    def setup(self, stage: str) -> None:
        if stage == "fit":
            data_module = self.trainer.datamodule
            self.test_list = data_module.test_list
            self.data_dir = data_module.data_dir
            self.out_coef_norm = torch.stack(data_module.mean_std[2:]).to(
                self.device
            )
            self.in_coef_norm = torch.stack(data_module.mean_std[:2]).to(
                self.device
            )
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def forward(self, data: torch.Tensor):
        return self.model(data)

    def model_step(self, batch: Any):

        pos, x, y, surf = batch
        pos, x, y, surf = (
            pos[0],
            x[0],
            y[0],
            surf[0],
        )
        data = torch.cat((pos, x), dim=1)
        out = self.model(data)
        loss = self.criterion(out, y)
        loss_surf = self.criterion(out[surf, :], y[surf, :])
        loss_vol = self.criterion(out[~surf, :], y[~surf, :])
        return loss_vol, loss_surf, loss

    def training_step(self, batch: Any, batch_idx: int):
        opt = self.optimizers()
        opt.zero_grad()
        loss_vol, loss_surf, loss = self.model_step(batch)
        loss_vol_var = loss_vol.mean(dim=0)
        loss_surf_var = loss_surf.mean(dim=0)
        if self.hparams.is_whole:
            loss = loss.mean()
            self.manual_backward(loss)
        elif not self.hparams.is_whole:
            loss = loss_vol.mean() + loss_surf.mean()
            self.manual_backward(loss)
        opt.step()

        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log_dict(
            {
                # "train/loss_wss": loss_wss.mean(),
                "train/loss_surf": loss_surf_var.mean(),
                "train/loss_volume": loss_vol_var.mean(),
                "train/loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_dict(
            {
                "train/volume_ux": loss_vol_var[0],
                "train/volume_uy": loss_vol_var[1],
                "train/volume_p": loss_vol_var[2],
                "train/volume_nut": loss_vol_var[3],
                "train/surface_p": loss_surf_var[2],
            },
            on_step=False,
            on_epoch=True,
        )
        # we can return here dict with any tensors
        # and then read it in some callback or in `on_training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        # we can return here dict with any tensors
        # and then read it in some callback or in `on_training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!

    def validation_step(self, batch: Any, batch_idx: int):
        loss_vol, loss_surf, loss = self.model_step(batch)
        loss_volume_var = loss_vol.mean(dim=0)
        loss_surf_var = loss_surf.mean(dim=0)
        if self.hparams.is_whole:
            loss = loss.mean()
        else:
            loss = loss_surf_var.mean() + loss_volume_var.mean()
        self.log_dict(
            {
                "val/loss_surf": loss_surf.mean(),
                "val/loss_volume": loss_vol.mean(),
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
                "val/surface_p": loss_surf_var[2],
            },
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        out = None
        pos, x, y, surf = batch
        pos, x, y, surf = (
            pos[0],
            x[0],
            y[0],
            surf[0],
        )
        data = torch.cat((pos, x), dim=1)
        out = self.model(data)
        out[surf, :2] = torch.zeros_like(out[surf, :2])
        out[surf, 3] = torch.zeros_like(out[surf, 3])

        true_coff, predict_coff, rel_err_force, rel_err_wss, rel_err_p = (
            metrics.compute_metric(
                out,
                self.test_list[batch_idx],
                surf.detach().cpu().numpy(),
                self.data_dir,
                self.out_coef_norm,
            )
        )
        surf_loss = self.criterion(out[surf], y[surf]).mean(dim=0)
        volume_loss = self.criterion(out[~surf], y[~surf]).mean(dim=0)
        self.log_dict(
            {
                "test/volume_ux": volume_loss[0],
                "test/volume_uy": volume_loss[1],
                "test/volume_p": volume_loss[2],
                "test/volume_nut": volume_loss[3],
                "test/surface_p": surf_loss[2],
                "test/relative_error_force_x": rel_err_force[0],
                "test/relative_error_force_y": rel_err_force[1],
                "test/relative_error_wss_x": rel_err_wss[0],
                "test/relative_error_wss_y": rel_err_wss[1],
                "test/relative_error_p": rel_err_p,
            },
            on_step=True,
            on_epoch=True,
        )
        self.test_step_outputs.append([true_coff, predict_coff])

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

        optimizer = self.hparams.optimizer(params=self.model.parameters())
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
