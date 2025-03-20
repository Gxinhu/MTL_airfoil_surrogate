from lightning import Callback


class EnableValidation(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.limit_val_batches = 0
        if trainer.current_epoch + 2 >= pl_module.warmup_epochs:
            trainer.limit_val_batches = 1.0
