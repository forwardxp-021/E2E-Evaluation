import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_logsumexp(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return torch.logsumexp(masked, dim=dim)


def multi_positive_infonce(
    z: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float = 0.1,
    neg_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    SupCon-style multi-positive InfoNCE.

    Args:
        z: [B, Z] L2-normalized embeddings.
        pos_mask: [B, B] bool. pos_mask[i, j] is True if j is a positive for i.
        temperature: scalar tau.
        neg_mask: [B, B] bool or None. If provided, denominator uses (positives U negatives).
                  If None, denominator uses all non-self samples.
    """
    bsz = z.size(0)
    device = z.device

    sim = (z @ z.T) / temperature
    eye = torch.eye(bsz, dtype=torch.bool, device=device)

    pos_mask = pos_mask & (~eye)

    if neg_mask is None:
        denom_mask = ~eye
    else:
        neg_mask = neg_mask & (~eye)
        denom_mask = (pos_mask | neg_mask) & (~eye)

    has_pos = pos_mask.any(dim=1)
    has_denom = denom_mask.any(dim=1)
    valid = has_pos & has_denom

    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=z.dtype, requires_grad=True), {
            "valid_anchors": 0,
            "mean_pos_per_anchor": 0.0,
        }

    log_num = _masked_logsumexp(sim, pos_mask, dim=1)
    log_den = _masked_logsumexp(sim, denom_mask, dim=1)
    losses = -(log_num - log_den)
    loss = losses[valid].mean()

    stats = {
        "valid_anchors": int(valid.sum().item()),
        "mean_pos_per_anchor": float(pos_mask.sum(dim=1)[valid].float().mean().item()),
    }
    return loss, stats


class SoftContrastiveLoss(nn.Module):
    """
    Feature-guided soft contrastive loss (soft-label InfoNCE / KL form).

    - Uses all pairs with soft weights from feature similarity.
    - No hard positive/negative mining inside the loss.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        tau_feat_min: float = 1e-3,
        feat_norm: str = "none",
        tau_mode: str = "anchor_median",
        gate_topm: int = 0,
        feat_sim: str = "tau",
        ls_k: int = 10,
        ls_mode: str = "row",
        ls_sigma_min: float = 1e-3,
        eps: float = 1e-8,
        debug_sim: bool = False,
        debug_topk: int = 10,
    ):
        super().__init__()
        self.temperature = temperature
        self.tau_feat_min = tau_feat_min
        self.feat_norm = feat_norm
        self.tau_mode = tau_mode
        self.gate_topm = gate_topm
        self.feat_sim = feat_sim
        self.ls_k = ls_k
        self.ls_mode = ls_mode
        self.ls_sigma_min = ls_sigma_min
        self.eps = eps
        self.debug_sim = debug_sim
        self.debug_topk = debug_topk
        self._debug_printed = False

    def forward(self, z: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bsz = z.size(0)
        device = z.device

        if self.feat_norm not in {"none", "batch_std", "l2"}:
            raise ValueError(f"Unsupported feat_norm: {self.feat_norm}")
        if self.tau_mode not in {"batch_median", "anchor_median"}:
            raise ValueError(f"Unsupported tau_mode: {self.tau_mode}")
        if self.feat_sim not in {"tau", "local_scale"}:
            raise ValueError(f"Unsupported feat_sim: {self.feat_sim}")
        if self.ls_mode not in {"row", "sym"}:
            raise ValueError(f"Unsupported ls_mode: {self.ls_mode}")
        if self.ls_k < 1:
            raise ValueError(f"ls_k must be >= 1, got {self.ls_k}")

        # 1) Normalize embedding
        z = F.normalize(z, dim=1)

        # 2) Embedding similarity
        sim_emb = (z @ z.T) / max(self.temperature, self.eps)

        # 3) Feature normalization
        if self.feat_norm == "batch_std":
            feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + 1e-6)
        elif self.feat_norm == "l2":
            feat = F.normalize(feat, dim=1)

        # 4) Feature distance
        dist_feat = torch.cdist(feat, feat, p=2)

        # 5) Remove diagonal
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        dist_feat = dist_feat.masked_fill(eye, float("inf"))

        # 5.5) Optional gate to top-M nearest neighbors per anchor.
        if self.gate_topm > 0 and bsz > 1:
            m = min(self.gate_topm, bsz - 1)
            topm_idx = torch.topk(dist_feat, k=m, dim=1, largest=False).indices
            gate_mask = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)
            gate_mask.scatter_(1, topm_idx, True)
            dist_feat = torch.where(gate_mask, dist_feat, torch.full_like(dist_feat, float("inf")))

        finite_mask = torch.isfinite(dist_feat)

        # 6) Build feature-similarity logits
        if self.feat_sim == "tau":
            # Keep original tau behavior unchanged.
            if self.tau_mode == "batch_median":
                finite_dist = dist_feat[finite_mask]
                if finite_dist.numel() == 0:
                    tau_feat = torch.tensor(self.tau_feat_min, device=device, dtype=dist_feat.dtype)
                else:
                    tau_feat = torch.median(finite_dist.detach())
                    tau_feat = torch.clamp(tau_feat, min=self.tau_feat_min)
                tau_row = tau_feat
                tau_stats = {"tau_feat": float(tau_feat.detach().item())}
            else:
                tau_row = torch.full((bsz, 1), self.tau_feat_min, device=device, dtype=dist_feat.dtype)
                for i in range(bsz):
                    row_vals = dist_feat[i, finite_mask[i]]
                    if row_vals.numel() > 0:
                        tau_row[i, 0] = torch.clamp(torch.median(row_vals.detach()), min=self.tau_feat_min)
                tau_stats = {
                    "tau_feat_mean": float(tau_row.mean().detach().item()),
                    "tau_feat_median": float(tau_row.median().detach().item()),
                }
            logits_feat = -dist_feat / tau_row
        else:
            sigma = torch.full((bsz,), self.ls_sigma_min, device=device, dtype=dist_feat.dtype)
            for i in range(bsz):
                row_vals = dist_feat[i, finite_mask[i]]
                if row_vals.numel() == 0:
                    continue
                k_eff = min(self.ls_k, int(row_vals.numel()))
                sigma_i = torch.topk(row_vals, k=k_eff, largest=False).values[-1]
                sigma[i] = torch.clamp(sigma_i.detach(), min=self.ls_sigma_min)
            sigma = torch.clamp(sigma, min=self.ls_sigma_min)

            if self.ls_mode == "row":
                logits_feat = -dist_feat / sigma[:, None]
            else:
                denom = sigma[:, None] * sigma[None, :] + self.eps
                logits_feat = -(dist_feat.pow(2)) / denom

            tau_stats = {
                "sigma_mean": float(sigma.mean().detach().item()),
                "sigma_median": float(sigma.median().detach().item()),
            }

        logits_feat = logits_feat.masked_fill(~finite_mask, float("-inf"))
        sim_feat = F.softmax(logits_feat, dim=1)

        # Diagonal should stay zero in supervision target.
        sim_feat = sim_feat.masked_fill(eye, 0.0)
        row_sum = sim_feat.sum(dim=1, keepdim=True)
        valid_rows = row_sum.squeeze(1) > self.eps
        if valid_rows.any():
            sim_feat = torch.where(valid_rows[:, None], sim_feat / (row_sum + self.eps), sim_feat)

        # If any row has no finite candidates (very small batches with aggressive gating),
        # fall back to uniform over non-self positions for that row.
        invalid_idx = torch.where(~valid_rows)[0]
        if invalid_idx.numel() > 0 and bsz > 1:
            for ridx in invalid_idx.tolist():
                row = torch.full((bsz,), 1.0 / float(bsz - 1), device=device, dtype=sim_feat.dtype)
                row[ridx] = 0.0
                sim_feat[ridx] = row

        if self.debug_sim and (not self._debug_printed) and bsz > 0:
            vals = sim_feat[0, : min(self.debug_topk, bsz)].detach().cpu().tolist()
            print(f"[SoftContrastiveLoss] sim_feat[0][: {min(self.debug_topk, bsz)} ] = {vals}")
            self._debug_printed = True

        # 8) Log-softmax for embedding
        sim_emb = sim_emb - sim_emb.max(dim=1, keepdim=True).values
        log_prob = F.log_softmax(sim_emb, dim=1)

        # 9) Final soft cross entropy
        loss_vec = -(sim_feat * log_prob).sum(dim=1)
        loss = loss_vec.mean()

        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

        entropy = -(sim_feat * torch.log(sim_feat + self.eps)).sum(dim=1)
        effective_k = 1.0 / (sim_feat.pow(2).sum(dim=1) + self.eps)

        stats = {
            "valid_anchors": int(bsz),
            "sim_feat_entropy_mean": float(entropy.mean().detach().item()),
            "effective_k_mean": float(effective_k.mean().detach().item()),
        }
        stats.update(tau_stats)
        return loss, stats
