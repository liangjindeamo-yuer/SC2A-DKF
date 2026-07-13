from typing import Mapping, Any
import torch
from idf.utils.common import instantiate_from_config
from idf.utils.metrics import calculate_psnr_pt, calculate_ssim_pt
from torchvision.transforms.functional import center_crop
from idf.models.lit_denoising import LitDenoising
from idf.utils.misc import const_like
import numpy as np

class LitADenoising(LitDenoising):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        denoiser_config: Mapping[str, Any],
        loss_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] = None,
        misc_config: Mapping[str, Any] = None,
    ):
        super().__init__(data_config, denoiser_config, loss_config, 
                         optimizer_config, scheduler_config,misc_config,)

        self.model = instantiate_from_config(denoiser_config)
        
        self.misc_config = misc_config
        if self.misc_config.compile:
            self.model = torch.compile(self.model)

        self.loss = instantiate_from_config(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config

        self.val_dataset_names = [k for k in self.data_config.validate.keys()]
        
        # data normalization
        self.data_scale = np.float32(data_config.norm.sigma_data) / np.float32(data_config.norm.raw_std)
        self.data_bias = np.float32(data_config.norm.mu_data) - np.float32(data_config.norm.raw_mean) * self.data_scale
        
        self.save_hyperparameters()

    def forward(self, noisy, adaptive_iter=False, max_iter=None, alpha_schedule=None):
        x = self.normalize(noisy)
        pred = self.model(x, adaptive_iter=adaptive_iter, max_iter=max_iter, alpha_schedule=alpha_schedule)
        pred = self.normalize(pred, reverse=True)
        return pred
        
    def normalize(self, x, reverse=False):
        if not reverse:
            if self.data_scale is not None:
                x = x * const_like(x, self.data_scale).reshape(1, -1, 1, 1)
            if self.data_bias is not None:
                x = x + const_like(x, self.data_bias).reshape(1, -1, 1, 1)
        else:
            if self.data_scale is not None:
                x = x - const_like(x, self.data_bias).reshape(1, -1, 1, 1)
            if self.data_bias is not None:    
                x = x / const_like(x, self.data_scale).reshape(1, -1, 1, 1)

        return x
        
    @torch.no_grad()
    def get_input(self, batch, config, norm_data=True):
        x = batch[config.input_key]
        y = batch[config.target_key]
        if norm_data:
            x = self.normalize(x)
            y = self.normalize(y)
        return x, y
   
    def training_step(self, batch, batch_idx):
        x, y = self.get_input(batch, self.data_config.train)
        self.log("bs", self.global_batch_size, prog_bar=True, logger=False)
        self.log('lr', self.get_lr(), prog_bar=True, logger=False)

        losses = dict()
        pred = self.model(x)
        losses['train/loss'] = self.loss(pred, y)
        if hasattr(self.model, "get_extra_losses"):
            extra_losses = self.model.get_extra_losses()
            for name, value in extra_losses.items():
                losses[f"train/{name}"] = value
        losses['train/total'] = sum(losses.values())
        self.log_dict(losses, prog_bar=True)
        if hasattr(self.model, "get_diagnostics"):
            diagnostics = self.model.get_diagnostics()
            if diagnostics:
                self.log_dict(
                    {
                        f"train/{k}": v.detach()
                        for k, v in diagnostics.items()
                        if f"train/{k}" not in losses
                    },
                    prog_bar=False,
                )
        return losses['train/total']
    
    def on_validation_start(self):
        self.sampled_images = []
        self.sample_steps_val = 50
        print(f"[Inference Settings] {self.misc_config.adaptive_iteration=}, {self.misc_config.max_iteration=}")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_name = self.val_dataset_names[dataloader_idx]
        val_config = self.data_config.validate[val_name]
        self._validation_step(batch, batch_idx, val_config, suffix=f"_{val_name}")
    
    def _validation_step(self, batch, batch_idx, val_config, suffix=""):
        x, y = self.get_input(batch, val_config, norm_data=False)
        assert x.shape[0] == 1

        pred = self(x, adaptive_iter=self.misc_config.adaptive_iteration, 
                    max_iter=self.misc_config.max_iteration,
                    alpha_schedule=self.misc_config.get('alpha_schedule'))
        
        pred = torch.clamp(pred, 0.0, 1.0)
        
        # Evaluate metrics.
        losses = {}
        losses[f'val{suffix}/psnr'] = calculate_psnr_pt(y, pred, 0, test_y_channel=False).mean()
        losses[f'val{suffix}/ssim'] = calculate_ssim_pt(y, pred, 0, test_y_channel=False).mean()

        self.log_dict(losses, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        if hasattr(self.model, "get_diagnostics"):
            diagnostics = self.model.get_diagnostics()
            if diagnostics:
                self.log_dict(
                    {f"val{suffix}/{k}": v.detach() for k, v in diagnostics.items()},
                    sync_dist=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )
        
        if batch_idx % 500 == 0:
            self.sampled_images.append(center_crop(x, (256,256))[0].cpu())
            self.sampled_images.append(center_crop(y, (256,256))[0].cpu())
            self.sampled_images.append(center_crop(pred, (256,256))[0].cpu())
