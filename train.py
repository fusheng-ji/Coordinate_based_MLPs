import torch
from torch import nn
import numpy as np
from opt import get_opts
from einops import rearrange
# dataset
from dataset import ImageDataset
from torch.utils.data import DataLoader

# models
from models import MLP, PE, Siren

# metrics
from metrics import psnr

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
# log
import wandb
   
class CoordMlp(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # wandb logger
        wandb.init(project="coord_mlps", entity="wenboji", name=hparams.exp_name)
        wandb.config = {
            "leaning_rate": hparams.lr,
            "epochs": hparams.num_epochs,
            "batch_size": hparams.batch_size,
            "architecture": hparams.arch
        }
        if hparams.use_pe == 'True':  # positional encoding
            # joint [2**i, 0; 0, 2**i] to generate positional matrix
            P = 2 * np.pi * torch.cat([torch.eye(2)*2**i for i in range(10)], 1) # (2, 2*10)
            self.pe = PE(P)
        if hparams.arch in ['relu', 'gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:  # different activation function
            kwargs = {'a': hparams.a, 'b': hparams.b}
            act = hparams.arch
            if hparams.use_pe == 'True':
                n_in = self.pe.out_dim
            else:
                n_in = 2
            self.mlp = MLP(n_in=n_in, act=act,
                           act_trainable=hparams.act_trainable,
                           **kwargs)
        elif hparams.arch == 'ff':      # fourier feature
            P = 2 * np.pi * hparams.sc * torch.normal(torch.zeros(2, 256), 
                                                      torch.ones(2, 256)) # (2, 2*10)
            self.pe = PE(P)
            self.mlp = MLP(n_in = self.pe.out_dim)
        elif hparams.arch == 'siren':   # siren
            self.mlp = Siren(first_omega_0 = hparams.omega_0,
                             hidden_omega_0 = hparams.omega_0)
            
        self.loss = nn.MSELoss()
            
    def forward(self, x):
        if hparams.use_pe == 'True' or hparams.arch == 'ff':
            x = self.pe(x)
        return self.mlp(x)
    def setup(self, stage=None):
        self.train_dataset = ImageDataset(hparams.image_path,
                                          hparams.img_wh, 
                                          'train')
        
        self.val_dataset = ImageDataset(hparams.image_path,
                                        hparams.img_wh, 
                                        'val')
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
    def configure_optimizers(self):
        self.optimizer = Adam(self.mlp.parameters(), lr=self.hparams.lr)
        return self.optimizer
    def training_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])
        
        loss = self.loss(rgb_pred, batch['rgb'])
        
        psnr_ = psnr(rgb_pred, batch['rgb'])
        
        self.log('train_loss', loss)
        self.log('train_psnr', psnr_, prog_bar=True)
        wandb.log({"train/loss": loss,
                   "train/psnr": psnr_})
        
        if hparams.arch in ['gaussian', 'quadratic',
                            'multi-quadratic', 'laplacian',
                            'super-gaussian', 'expsin']:
            for i in [1, 3, 5]: # activation layers
                self.log(f'act/l{i}_a', self.mlp.net[i].a)
                if hasattr(self.mlp.net[i], "b"):
                    self.log(f'act/l{i}_b', self.mlp.net[i].b)
        return loss
    def validation_step(self, batch, batch_idx):
        rgb_pred = self(batch['uv'])
        
        loss = self.loss(rgb_pred, batch['rgb']) 
        psnr_ = psnr(rgb_pred, batch['rgb'])
        
        log = {'val_loss': loss,
               'val_psnr': psnr_,
               'rgb_gt': batch['rgb'],
               'rgb_pred': rgb_pred} # (B, 3)
        wandb.log({"val_loss": loss,
                   "val_psnr": psnr_})
        return log
    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        rgb_gt = torch.cat([x['rgb_gt'] for x in outputs])
        rgb_gt = rearrange(rgb_gt, '(h w) c -> c h w',
                           h=hparams.img_wh[1],
                           w=hparams.img_wh[0])
        
        rgb_pred = torch.cat([x['rgb_pred'] for x in outputs])
        rgb_pred = rearrange(rgb_pred, '(h w) c -> c h w',
                             h=hparams.img_wh[1],
                             w=hparams.img_wh[0])
        
        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        # log the predicted rgb every epoch
        wandb.log({"val/loss": mean_loss,
                   "val/psnr": mean_psnr,
                   "rgb_gt_pred": wandb.Image(torch.stack([rgb_gt, rgb_pred])),
                   "step": self.global_step
                   })
if __name__=='__main__':
    hparams = get_opts()
    coord_mlp = CoordMlp(hparams)
     
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [pbar]
    
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0, # test validation is right or false at ith step
                      log_every_n_steps=1,
                      check_val_every_n_epoch=20, # do validation behind every 20 epochs
                      benchmark=True)
    
    trainer.fit(coord_mlp)