import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from prdc import compute_prdc
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from typing import Optional, Tuple


class PlotParams():
    def __init__(self, labelsize=14):
        self.labelsize = labelsize

    def set_params(self):
        mpl.rc('font', family='serif', size=15)
        # mpl.rc('text', usetex=True)
        mpl.rcParams['axes.linewidth'] = 1.3
        mpl.rcParams['xtick.major.width'] = 1
        mpl.rcParams['ytick.major.width'] = 1
        mpl.rcParams['xtick.minor.width'] = 1
        mpl.rcParams['ytick.minor.width'] = 1
        mpl.rcParams['xtick.major.size'] = 10
        mpl.rcParams['ytick.major.size'] = 10
        mpl.rcParams['xtick.minor.size'] = 5
        mpl.rcParams['ytick.minor.size'] = 5
        mpl.rcParams['xtick.labelsize'] = self.labelsize
        mpl.rcParams['ytick.labelsize'] = self.labelsize

plotter = PlotParams()
plotter.set_params()


### VAE ###
def kl(m_1: Tensor, lv_1: Tensor, m_2: Tensor, lv_2: Tensor) -> Tensor:
    latent_kl = (0.5 * (-1 + (lv_2 - lv_1) + lv_1.exp() / lv_2.exp()
                 + (m_2 - m_1).pow(2) / lv_2.exp()).mean(dim=0))

    return latent_kl.sum()

def _get_mu_var(m_1, v_1, m_2, v_2, a=0.5, storer=None):
    v_a = 1 / ((1 - a) / v_1 + a / v_2)
    m_a = v_a * ((1 - a) * m_1 / v_1 + a * m_2 / v_2)

    return m_a, v_a

def gjs(mean, logvar, dual=True, a=0.5, invert_alpha=True):
    mean_0 = torch.zeros_like(mean)
    var_0 = torch.ones_like(logvar)
    logvar_0 = torch.zeros_like(logvar)

    if invert_alpha:
        mean_a, var_a = _get_mu_var(mean, logvar.exp(), mean_0, var_0, a=1-a)
    else:
        mean_a, var_a = _get_mu_var(mean, logvar.exp(), mean_0, var_0, a=a)
    logvar_a = var_a.log()

    if dual:
        kl_1 = kl(mean_a, var_a, mean, logvar)
        kl_2 = kl(mean_a, var_a, mean_0, logvar_0)
    else:
        kl_1 = kl(mean, logvar, mean_a, logvar_a)
        kl_2 = kl(mean_0, logvar_0, mean_a, logvar_a)

    return (1 - a) * kl_1 + a * kl_2


class Reparameteriser(nn.Module):
    def __init__(self, size: int, divergence='kl') -> None:
        super().__init__()
        self.divergence = divergence

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean, logvar = x.split(x.shape[1] // 2, dim=1)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mean + std * eps
        else:
            sample = mean

        if self.divergence == 'kl':
            div = kl(mean, logvar, torch.zeros_like(mean), torch.zeros_like(logvar))
        elif self.divergence == 'gjs':
            div = gjs(mean, logvar, dual=False, a=0.5, invert_alpha=True)

        return sample, div


class VAENet(nn.Module):
    def __init__(self, input_size: int, divergence: str = 'kl', latent_size: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * latent_size)
        )
        self.reparametiser = Reparameteriser(size=10, divergence=divergence)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x, div = self.reparametiser(x)
        x = self.decoder(x)

        return x, div


class VAE(pl.LightningModule):
    def __init__(self, input_size: int = 114, divergence: str = 'kl') -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = VAENet(input_size=input_size, divergence=divergence)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x, nan_mask = torch.split(x, x.shape[1] // 2, dim=1)
        xhat, div_loss = self(x)
        recon_loss = F.mse_loss(nan_mask * x + (1 - nan_mask) * xhat, x)
        self._log('train', recon_loss, div_loss)
        return recon_loss + 0.01 * div_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x, nan_mask = torch.split(x, x.shape[1] // 2, dim=1)
        xhat, div_loss = self(x)
        recon_loss = F.mse_loss(nan_mask * x + (1 - nan_mask) * xhat, x)
        self._log('valid', recon_loss, div_loss)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0, amsgrad=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=100, eta_min=1e-6, last_epoch=-1
                )
            }
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def _log(self, split, recon_loss, div_loss):
        with torch.no_grad():
            self.log(f"{split}/recon_loss", recon_loss)
            self.log(f"{split}/div_loss", div_loss)
            self.log(f"{split}/loss", recon_loss+div_loss)


### SBM ###
class SBMNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def _gennorm_score(x: Tensor, mu: Optional[float] = 0.0, alpha: Optional[float] = 1.0, beta: Optional[float] = 2.0) -> Tensor:
    return - (beta / alpha ** beta) * torch.sign(x - mu) * torch.abs(x - mu) ** (beta - 1)


def dsm(scorenet: nn.Module, samples: Tensor, sigmas: Tensor, beta: float, nan_mask: Optional[Tensor] = None) -> Tensor:
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))

    if beta == 2.0:
        noise = torch.randn_like(samples)
    else:
        alpha = 2 ** 0.5
        gamma = np.random.gamma(shape=1+1/beta, scale=2**(beta/2), size=samples.shape)
        delta = alpha * gamma ** (1 / beta) / (2 ** 0.5)
        noise = torch.tensor((2 * np.random.rand(*samples.shape) - 1) * delta).float().to(samples.device)

    noised_samples = samples + noise * used_sigmas
    if beta == 2.0:
        target = - 1 / (used_sigmas ** 2) * noise
    else:
        target = _gennorm_score(noised_samples, mu=samples, alpha=used_sigmas * 2.0 ** 0.5, beta=beta)
    scores = scorenet(noised_samples)
    loss = 1 / 2. * (scores - target) ** 2
    if nan_mask is not None:
        loss *= 1 - nan_mask
    loss = loss.sum(dim=1) * used_sigmas.squeeze() ** 2

    return loss.mean(dim=0)


class SBM(pl.LightningModule):
    def __init__(self, input_size: int = 114, beta: float = 2.0) -> None:
        super().__init__()
        self.net = SBMNet(input_size=input_size)
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to('cuda')
        self.sigmas = torch.tensor(np.exp(np.linspace(np.log(0.001), np.log(10), 20))).float().cuda()
        self.beta = beta

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x, nan_mask = torch.split(x, x.shape[1] // 2, dim=1)
        dsm_loss = dsm(self.net, x, self.sigmas, self.beta, nan_mask)
        self._log('train', dsm_loss)
        return dsm_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x, nan_mask = torch.split(x, x.shape[1] // 2, dim=1)
        dsm_loss = dsm(self.net, x, self.sigmas, self.beta, nan_mask)
        self._log('valid', dsm_loss)
        return dsm_loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0.0, amsgrad=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=100, eta_min=1e-6, last_epoch=-1
                )
            }
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def _log(self, split, dsm_loss):
        with torch.no_grad():
            self.log(f"{split}/loss", dsm_loss)


### Train ###
def main(args):
    train = pd.read_csv(os.path.join('data', 'set-a.csv'))
    valid = pd.read_csv(os.path.join('data', 'set-b.csv'))
    for c in ['recordid']:
        del train[c]
        del valid[c]

    scaler = StandardScaler()
    X_train = np.nan_to_num(scaler.fit_transform(train.values))
    X_valid = np.nan_to_num(scaler.transform(valid.values))
    train_nan_mask = train.isna().values
    valid_nan_mask = train.isna().values
    X_train = np.concatenate((X_train, train_nan_mask), axis=1)
    X_valid = np.concatenate((X_valid, valid_nan_mask), axis=1)

    train_dataset = TensorDataset(torch.tensor(X_train).float())
    valid_dataset = TensorDataset(torch.tensor(X_valid).float())

    train_loader = DataLoader(train_dataset, batch_size=128, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, pin_memory=True, shuffle=False)

    checkpoint_callback = ModelCheckpoint(monitor="valid/loss", mode='min')

    if args.model_type == 'vae':
        model = VAE(input_size=X_train.shape[1] // 2, divergence=args.divergence)
    elif args.model_type == 'sbm':
        model = SBM(beta=args.beta)
    else:
        raise AssertionError('model_type unrecognised')

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        callbacks=[checkpoint_callback],
        log_every_n_steps=16
    )
    trainer.fit(model, train_loader, valid_loader)

    if args.model_type == 'vae':
        sample = np.random.randn(4000, 64)
        sample = model.net.decoder(torch.tensor(sample).float())

    elif args.model_type == 'sbm':
        sigmas = np.exp(np.linspace(np.log(10), np.log(0.001), 20))
        sample = np.random.rand(4000, 114) - 0.5
        sample = torch.tensor(sample).float()
        for sigma in tqdm(sigmas):
            eps = 0.001 * (sigma / sigmas[0]) ** 2
            for _ in range(200):
                sample += eps * model(sample) + (2 * eps) ** 0.5 * torch.randn_like(sample)

    sample = sample.detach().numpy()
    sample = scaler.inverse_transform(sample[:, :114])
    sample = pd.DataFrame(sample, columns=train.columns)

    plt.figure(figsize=(10, 5))
    for plot_idx, i in enumerate([0, 2, 3, 7, 9, 10]):
        plt.subplot(2, 3, 1+plot_idx)
        # plt.hist(train.iloc[:, i], label=f'Train', bins=np.arange(0, 300, 3), alpha=0.5)
        plt.hist(valid.iloc[:, i], label=f'Valid', bins=np.arange(0, 300, 3), alpha=0.5)
        n = valid.iloc[:, i].notna().sum()
        plt.hist(sample.iloc[:n, i], label=f'Generated', bins=np.arange(0, 300, 3), alpha=0.5)
        plt.xlim(sample.iloc[:n, i].min()-10, sample.iloc[:n, i].max()+10)
        if plot_idx == 0:
            # plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
            plt.legend()
        if plot_idx > 2:
            plt.xlabel('Value')
        if plot_idx in [0, 3]:
            plt.ylabel('Frequency')
        plt.title(sample.columns[i])
    plt.tight_layout()
    if args.model_type == 'vae':
        plt.savefig(os.path.join(f'{args.model_type}_{args.divergence}.pdf'))
    elif args.model_type == 'sbm':
        plt.savefig(os.path.join('plots', f'{args.model_type}_{args.beta}.pdf'))
    plt.show()

    notna = valid.iloc[:, [0, 2, 3, 7, 9, 10]].notna().all(axis=1).values
    metrics = compute_prdc(
        real_features=valid.values[notna][:, [0, 2, 3, 7, 9, 10]],
        fake_features=sample.values[notna][:, [0, 2, 3, 7, 9, 10]],
        nearest_k=5
    )
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['vae', 'sbm'])
    parser.add_argument('--divergence', type=str, choices=['kl', 'gjs'])
    parser.add_argument('--beta', type=float)
    args = parser.parse_args()
    main(args)
