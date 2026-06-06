import copy
import torch
import torch.nn.functional as F
import math
from .utils import select_low_loss_indices, get_ot_pair

class MACWrapper:
    def __init__(self, model, vae=None, add_weight=0.1, ln=True, model_type='full'):
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

    def generate_targets(self, images, z0, labels, indices=None, booststar_every=8):
        x_high, z0_high, cond_high = images, z0, labels
        self.ema_model.eval()

        if self.add_weight==0:
            weights = torch.ones(images.shape[0], device=images.device)
        else:
            if indices is not None:
                weights = torch.ones(images.shape[0], device=images.device)
                weights[indices] = 1 + self.add_weight
            else:
                weights = None

        FORCE_T = -1
        FORCE_DT = -1

        # bootstrap object
        current_batch_size = x_high.shape[0]
        # 1. create step sizes dt
        bootstrap_batch_size = current_batch_size // booststar_every #=8
        log2_sections = int(math.log2(self.DENOISE_TIMESTEPS))
        # print(f"log2_sections: {log2_sections}")
        # print(f"bootstrap_batch_size: {bootstrap_batch_size}")

        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)
        # print(f"dt_base: {dt_base}")

        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)])
        # print(f"dt_base: {dt_base}")
        
        force_dt_vec = torch.ones(bootstrap_batch_size) * FORCE_DT
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(images.device)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]
        # print(f"dt: {dt}")

        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]
        # print(f"dt_bootstrap: {dt_bootstrap}")

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        # print(f"dt_sections: {dt_sections}")

        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
            ]).to(images.device)
        
        # print(f"t[randint]: {t}")
        t = t / dt_sections
        # print(f"t[normalized]: {t}")
        
        force_t_vec = torch.ones(bootstrap_batch_size, dtype=torch.float32).to(images.device) * FORCE_T
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(images.device)
        t_full = t[:, None, None, None]

        # print(f"t_full: {t_full}")

        # 3. generate bootstrap targets:
        x_1 = x_high[:bootstrap_batch_size]
        x_0 = z0_high[:bootstrap_batch_size]

        # get dx at timestep t
        x_t = (1 - (1-1e-5) * t_full)*x_0 + t_full*x_1

        bst_labels = cond_high[:bootstrap_batch_size]


        with torch.no_grad():
            v_b1 = self.ema_model(x_t, t, dt_base_bootstrap, bst_labels)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = torch.clip(x_t2, -4, 4)
        
        with torch.no_grad():
            v_b2 = self.ema_model(x_t2, t2, dt_base_bootstrap, bst_labels)

        v_target = (v_b1 + v_b2) / 2

        v_target = torch.clip(v_target, -4, 4)
        
        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t
        bst_l = bst_labels

        # 4. generate flow-matching targets

        labels_dropout = torch.bernoulli(torch.full(cond_high.shape, self.CLASS_DROPOUT_PROB)).to(images.device)
        labels_dropped = torch.where(labels_dropout.bool(), self.NUM_CLASSES, cond_high)

        # sample t(normalized)
        t = torch.randint(low=0, high=self.DENOISE_TIMESTEPS, size=(x_high.shape[0],), dtype=torch.float32)
        # print(f"t: {t}")
        t /= self.DENOISE_TIMESTEPS
        # print(f"t: {t}")
        force_t_vec = torch.ones(x_high.shape[0]) * FORCE_T
        # force_t_vec = torch.full((images.shape[0],), FORCE_T, dtype=torch.float32)
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(images.device)
        # t_full = t.view(-1, 1, 1, 1)
        t_full = t[:, None, None, None]

        # print(f"t_full: {t_full}")

        # sample flow pairs x_t, v_t
        x_0 = z0_high
        x_1 = x_high
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - 1e-5) * x_0

        dt_flow = int(math.log2(self.DENOISE_TIMESTEPS))
        dt_base = (torch.ones(x_high.shape[0], dtype=torch.int32) * dt_flow).to(images.device)

        # num_select = max(1, int(len(indices) * self.ratio))
        # selected_indices = indices[torch.randperm(len(indices))[:num_select]]
        # dt_base[selected_indices] = 0

        # 5. merge flow and bootstrap
        bst_size = current_batch_size // booststar_every
        bst_size_data = current_batch_size - bst_size

        # print(f"bst_size: {bst_size}")
        # print(f"bst_size_data: {bst_size_data}")

        x_t = torch.cat([bst_xt, x_t[:bst_size_data]], dim=0)
        t = torch.cat([bst_t, t[:bst_size_data]], dim=0)

        dt_base = torch.cat([bst_dt, dt_base[:bst_size_data]], dim=0)
        v_t = torch.cat([bst_v, v_t[:bst_size_data]], dim=0)
        labels_dropped = torch.cat([bst_l, labels_dropped[:bst_size_data]], dim=0)
        if weights is not None:
            final_weights = torch.cat([weights[:bootstrap_batch_size],weights[:bst_size_data]], dim=0)
        else:
            final_weights = None

        return x_t, v_t, t, dt_base, labels_dropped, final_weights

    def forward(self, x, c, percentile, global_step):
        z0 = torch.randn_like(x)
        batch = (x, c)

        if self.model_type == 'select':
            if self.add_weight==0:
                x_t, v_t, t, dt_base, labels_dropped, weights = self.generate_targets(x, z0, c, None, self.BOOTSTRAP_EVERY)
            else:
                indices_low = select_low_loss_indices(self.ema_model, batch, z0, percentile, model='shortcut')
                x_t, v_t, t, dt_base, labels_dropped, weights = self.generate_targets(x, z0, c, indices_low, self.BOOTSTRAP_EVERY)
            vtheta = self.model(x_t, t, dt_base, labels_dropped)
            per_sample_loss = F.mse_loss(vtheta, v_t, reduction='none')
            per_sample_loss = per_sample_loss.mean(dim=(1,2,3))
            loss = (per_sample_loss * weights).mean()

        elif self.model_type == 'full':
            z0, x, c = get_ot_pair(self.ema_model, batch, z0, global_step, model='shortcut')
            x_t, v_t, t, dt_base, labels_dropped, weights = self.generate_targets(x, z0, c, None, self.BOOTSTRAP_EVERY)
            vtheta = self.model(x_t, t, dt_base, labels_dropped)
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
            dt_base = torch.ones_like(t).to(z.device) * math.log2(sample_steps)

            vc = self.model(z, t, dt_base, cond)
            if null_cond is not None:
                vu = self.model(z, t, dt_base, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z + dt * vc
        
        if self.vae is not None:
            decoded = self.vae.decode(z/self.vae.config.scaling_factor)[0]
        else:
            decoded = z
        images.append(decoded)
    
        return images
