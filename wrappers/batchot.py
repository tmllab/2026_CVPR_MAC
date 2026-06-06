import copy
import torch
import torch.nn.functional as F
import math
import numpy as np
import ot
from .utils import select_low_loss_indices, get_ot_pair
import warnings

class MACWrapper:
    def __init__(self, model, vae=None, add_weight=1.0, ln=True, model_type='full'):
        self.model = model
        self.vae = vae
        self.ema_model = copy.deepcopy(model).eval()
        self.ln = ln
        self.BOOTSTRAP_EVERY = 8
        self.DENOISE_TIMESTEPS = 128
        self.CLASS_DROPOUT_PROB = 0.1
        self.NUM_CLASSES = 10
        self.decay = 0.999
        self.add_weight = add_weight
        self.model_type = model_type

    @torch.no_grad()
    def update_ema(self):
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def create_targets(self, images, labels, percentile, global_step):

        self.ema_model.eval()

        current_batch_size = images.shape[0]

        FORCE_T = -1
        FORCE_DT = -1

        labels_dropout = torch.bernoulli(torch.full(labels.shape, self.CLASS_DROPOUT_PROB)).to(images.device)
        labels_dropped = torch.where(labels_dropout.bool(), self.NUM_CLASSES, labels)

        # sample t(normalized)
        t = torch.randint(low=0, high=self.DENOISE_TIMESTEPS, size=(images.shape[0],), dtype=torch.float32)
        # print(f"t: {t}")
        t /= self.DENOISE_TIMESTEPS
        # print(f"t: {t}")
        force_t_vec = torch.ones(images.shape[0]) * FORCE_T
        # force_t_vec = torch.full((images.shape[0],), FORCE_T, dtype=torch.float32)
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(images.device)
        # t_full = t.view(-1, 1, 1, 1)
        t_full = t[:, None, None, None]


        x_0 = torch.randn_like(images).to(images.device)
        x_1 = images

        a = np.ones((current_batch_size,)) / current_batch_size
        b = np.ones((current_batch_size,)) / current_batch_size

        x_0_reshape = x_0.view(x_0.shape[0], -1)
        x_1_reshape = x_1.view(x_1.shape[0], -1)
        M = torch.cdist(x_0_reshape, x_1_reshape) ** 2
        eps = 1e-12
        mean = M.mean()
        std = M.std().clamp_min(eps)
        M = (M - mean) / std
        M = M.cpu().numpy()

        gamma = ot.sinkhorn(a, b, M, reg=0.05)

        p = gamma.flatten()
        if (not np.isfinite(p.sum())) or np.abs(p.sum()) < 1e-8:
            warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        p = p / p.sum()
        choices = np.random.choice(
            gamma.shape[0] * gamma.shape[1], p=p, size=current_batch_size, replace=True
        )
        row_ind, col_ind = np.divmod(choices, gamma.shape[1])

        # 3) Reorder.
        x_0 = x_0[row_ind]
        x_1 = x_1[col_ind]
        labels_dropped = labels_dropped[col_ind]
        batch = (x_1, labels_dropped)

        if self.model_type == 'select':
            if self.add_weight == 0:
                weights = torch.ones(images.shape[0], device=images.device)
            else:
                indices_low = select_low_loss_indices(self.ema_model, batch, x_0, percentile, model='batchot')
                weights = torch.ones(images.shape[0], device=images.device)
                weights[indices_low] = 1 + self.add_weight

        elif self.model_type == 'full':
            z0, x, c = get_ot_pair(self.ema_model, batch, x_0, global_step, model='batchot')
            x_0 = z0
            x_1 = x
            labels_dropped = c
            weights = None

        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0

        dt_flow = int(math.log2(self.DENOISE_TIMESTEPS))
        dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(images.device)

        return x_t, v_t, t, dt_base, labels_dropped, weights

    def forward(self, x, c, percentile, global_step):
        x_t, v_t, t, dt_base, labels_dropped, weights = self.create_targets(x, c, percentile, global_step)
        vtheta = self.model(x_t, t, dt_base, labels_dropped)

        if self.model_type == 'select':
            per_sample_loss = F.mse_loss(vtheta, v_t, reduction='none')
            per_sample_loss = per_sample_loss.mean(dim=(1,2,3))
            loss = (per_sample_loss * weights).mean()
        elif self.model_type == 'full':
            loss = F.mse_loss(vtheta, v_t)

        return loss
    
    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=64, cfg=2.0):
        if self.vae is not None:
            self.vae.to(z.device)
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)
            dt_base = torch.ones_like(t).to(z.device) * int(math.log2(self.DENOISE_TIMESTEPS))

            vc = self.model(z, t, dt_base, cond)
            if null_cond is not None:
                vu = self.model(z, t, dt_base, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z + dt * vc
        
        if self.vae is not None:
            decoded = self.vae.decode(z / self.vae.config.scaling_factor)[0]
        else:
            decoded = z
        images.append(decoded)
        return images
