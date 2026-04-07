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
        eps: float = 1e-8,
        debug_sim: bool = False,
        debug_topk: int = 10,
    ):
        super().__init__()
        self.temperature = temperature
        self.tau_feat_min = tau_feat_min
        self.eps = eps
        self.debug_sim = debug_sim
        self.debug_topk = debug_topk
        self._debug_printed = False

    def forward(self, z: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bsz = z.size(0)
        device = z.device

        # 1) Normalize embedding
        z = F.normalize(z, dim=1)

        # 2) Embedding similarity
        sim_emb = (z @ z.T) / max(self.temperature, self.eps)

        # 3) Feature normalization
        feat = (feat - feat.mean(dim=0, keepdim=True)) / (feat.std(dim=0, keepdim=True) + 1e-6)

        # 4) Feature distance
        dist_feat = torch.cdist(feat, feat, p=2)

        # 5) Remove diagonal
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        dist_feat = dist_feat.masked_fill(eye, float("inf"))

        # 6) Adaptive feature temperature (median distance)
        finite_dist = dist_feat[torch.isfinite(dist_feat)]
        if finite_dist.numel() == 0:
            tau_feat = torch.tensor(self.tau_feat_min, device=device, dtype=dist_feat.dtype)
        else:
            tau_feat = torch.median(finite_dist.detach())
            tau_feat = torch.clamp(tau_feat, min=self.tau_feat_min)

        # 7) Soft similarity from feature distance (critical fix)
        sim_feat = F.softmax(-dist_feat / tau_feat, dim=1)

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

        stats = {
            "valid_anchors": int(bsz),
            "mean_pos_per_anchor": float((sim_feat > 0).sum(dim=1).float().mean().item()),
        }
        return loss, stats
