import torch
import torch.nn as nn
import torch.nn.functional as F

# Default cf_valid_frac bucket edges for condition gating.
# Samples are bucketed into [0, edge0), [edge0, edge1), [edge1, inf).
DEFAULT_CF_BUCKET_EDGES: list[float] = [0.2, 0.6]


def masked_pairwise_l2(
    feat: torch.Tensor,
    feat_valid: torch.Tensor,
    min_common_dims: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked pairwise L2 distances and common valid-dimension counts.

    For each pair (i, j), this uses mask m_ij = valid_i * valid_j, averages squared
    difference over common valid dimensions, then applies sqrt. Pairs with common
    dimension count below ``min_common_dims`` are set to +inf in the returned distance.

    Returns:
        dist: [B, B] masked pairwise distance matrix.
        common_counts: [B, B] number of common valid dimensions per pair.
    """
    # feat_valid is binary {0,1} mask; >0 keeps behavior robust to float mask tensors.
    valid = (feat_valid > 0).to(feat.dtype)
    common_dims = valid[:, None, :] * valid[None, :, :]
    common_counts = common_dims.sum(dim=-1)

    diff2 = (feat[:, None, :] - feat[None, :, :]).pow(2)
    # Clamp denominator to prevent division-by-zero when a pair has no common valid dims.
    denom = common_counts.clamp(min=1.0)
    dist = torch.sqrt((diff2 * common_dims).sum(dim=-1) / denom)

    if min_common_dims > 1:
        dist = dist.masked_fill(common_counts < float(min_common_dims), float("inf"))
    return dist, common_counts


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


def build_cond_mask(
    cond: torch.Tensor,
    speed_tol: float,
    dist_tol: float,
    vrel_tol: float,
    cf_bucket_edges: list[float],
) -> torch.Tensor:
    """Build pairwise condition compatibility mask [B, B].

    cond columns: [speed_mean, dist_mean, vrel_mean, cf_valid_frac (optional)]
    Two samples are compatible if all their condition differences are within the
    respective tolerances AND they fall in the same cf_valid_frac bucket.
    Self-pairs are included in the mask; callers should exclude them as needed.
    """
    device = cond.device
    bsz = cond.size(0)

    # Speed, dist, vrel gating using absolute difference.
    speed = cond[:, 0]
    dist = cond[:, 1]
    vrel = cond[:, 2]

    speed_diff = (speed[:, None] - speed[None, :]).abs()
    dist_diff = (dist[:, None] - dist[None, :]).abs()
    vrel_diff = (vrel[:, None] - vrel[None, :]).abs()

    mask = (speed_diff <= speed_tol) & (dist_diff <= dist_tol) & (vrel_diff <= vrel_tol)

    # CF bucket gating: only when cf_valid_frac column is present.
    if cond.size(1) >= 4 and len(cf_bucket_edges) > 0:
        cf_frac = cond[:, 3]
        # Compute bucket index for each sample using digitize-style logic.
        edges = sorted(cf_bucket_edges)
        bucket = torch.zeros(bsz, dtype=torch.long, device=device)
        for edge in edges:
            bucket = bucket + (cf_frac > edge).long()
        bucket_match = bucket[:, None] == bucket[None, :]
        mask = mask & bucket_match

    return mask


class SoftContrastiveLoss(nn.Module):
    """
    Feature-guided soft contrastive loss (soft-label InfoNCE / KL form).

    - Uses all pairs with soft weights from feature similarity.
    - No hard positive/negative mining inside the loss.
    - Optionally applies condition gating (cond_mode='hard_box') to restrict
      comparisons to samples with similar operating conditions.
    - Supports loss_mode in {softkl, supcon, hybrid} for flexible supervision.
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
        ls_alpha: float = 1.0,
        feat_dist_mode: str = "plain",
        min_common_dims: int = 5,
        eps: float = 1e-8,
        debug_sim: bool = False,
        debug_topk: int = 10,
        # Condition gating parameters.
        cond_mode: str = "off",
        cond_speed_tol: float = 2.0,
        cond_dist_tol: float = 5.0,
        cond_vrel_tol: float = 1.0,
        cond_cf_bucket_edges: list[float] | None = None,
        min_cond_candidates: int = 8,
        # Loss mode parameters.
        loss_mode: str = "softkl",
        pos_topk: int = 8,
        w_supcon: float = 1.0,
        w_soft: float = 0.2,
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
        self.ls_alpha = ls_alpha
        self.feat_dist_mode = feat_dist_mode
        self.min_common_dims = min_common_dims
        self.eps = eps
        self.debug_sim = debug_sim
        self.debug_topk = debug_topk
        self._debug_printed = False
        # Condition gating.
        self.cond_mode = cond_mode
        self.cond_speed_tol = cond_speed_tol
        self.cond_dist_tol = cond_dist_tol
        self.cond_vrel_tol = cond_vrel_tol
        self.cond_cf_bucket_edges: list[float] = cond_cf_bucket_edges if cond_cf_bucket_edges is not None else list(DEFAULT_CF_BUCKET_EDGES)
        self.min_cond_candidates = min_cond_candidates
        # Loss mode.
        self.loss_mode = loss_mode
        self.pos_topk = pos_topk
        self.w_supcon = w_supcon
        self.w_soft = w_soft

    def _build_cond_gate(
        self, bsz: int, cond: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor | None:
        """Build [B, B] boolean condition gate mask or return None if gating is off/unavailable."""
        if self.cond_mode == "off" or cond is None:
            return None
        raw_mask = build_cond_mask(
            cond,
            speed_tol=self.cond_speed_tol,
            dist_tol=self.cond_dist_tol,
            vrel_tol=self.cond_vrel_tol,
            cf_bucket_edges=self.cond_cf_bucket_edges,
        )
        # Exclude self.
        eye = torch.eye(bsz, dtype=torch.bool, device=device)
        raw_mask = raw_mask & (~eye)

        # Per-anchor fallback: if an anchor has fewer than min_cond_candidates comparable
        # samples, fall back to all non-self samples for that anchor.
        n_cands = raw_mask.sum(dim=1)  # [B]
        fallback_rows = n_cands < self.min_cond_candidates
        if fallback_rows.any():
            fallback_mask = ~eye  # use all non-self pairs
            # Apply fallback per row where needed.
            raw_mask = torch.where(fallback_rows[:, None], fallback_mask, raw_mask)
        return raw_mask, fallback_rows

    def forward(
        self,
        z: torch.Tensor,
        feat: torch.Tensor,
        feat_valid: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
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
        if self.feat_dist_mode not in {"plain", "masked"}:
            raise ValueError(f"Unsupported feat_dist_mode: {self.feat_dist_mode}")
        if self.ls_k < 1:
            raise ValueError(f"ls_k must be >= 1, got {self.ls_k}")
        if self.ls_alpha <= 0:
            raise ValueError(f"ls_alpha must be > 0 to keep sharpening valid, got {self.ls_alpha}")
        if self.min_common_dims < 1:
            raise ValueError(f"min_common_dims must be >= 1, got {self.min_common_dims}")
        if self.loss_mode not in {"softkl", "supcon", "hybrid"}:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}")

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
        common_stats = {}
        if self.feat_dist_mode == "masked" and feat_valid is not None:
            dist_feat, common_counts = masked_pairwise_l2(feat, feat_valid, min_common_dims=self.min_common_dims)
            if bsz > 1:
                off_diag = ~torch.eye(bsz, dtype=torch.bool, device=device)
                off_diag_counts = common_counts[off_diag]
                common_stats = {
                    "common_dims_mean": float(off_diag_counts.float().mean().detach().item()),
                    "common_dims_p10": float(torch.quantile(off_diag_counts.float(), 0.10).detach().item()),
                    "common_dims_p50": float(torch.quantile(off_diag_counts.float(), 0.50).detach().item()),
                    "common_dims_p90": float(torch.quantile(off_diag_counts.float(), 0.90).detach().item()),
                }
            else:
                common_stats = {
                    "common_dims_mean": 0.0,
                    "common_dims_p10": 0.0,
                    "common_dims_p50": 0.0,
                    "common_dims_p90": 0.0,
                }
        else:
            dist_feat = torch.cdist(feat, feat, p=2)

        # 5) Remove diagonal
        eye = torch.eye(bsz, device=device, dtype=torch.bool)
        dist_feat = dist_feat.masked_fill(eye, float("inf"))

        # 5a) Build condition gate and apply to feature distance.
        cond_result = self._build_cond_gate(bsz, cond, device)
        cond_mask: torch.Tensor | None = None
        fallback_rows: torch.Tensor | None = None
        cond_stats: dict = {}
        if cond_result is not None:
            cond_mask, fallback_rows = cond_result
            # Gate feature distances: pairs outside cond_mask become +inf.
            dist_feat = dist_feat.masked_fill(~cond_mask, float("inf"))
            n_cands = cond_mask.sum(dim=1).float()
            n_fallback = int(fallback_rows.sum().item()) if fallback_rows is not None else 0
            cond_stats = {
                "cond_candidates_mean": float(n_cands.mean().detach().item()),
                "cond_fallback_frac": float(n_fallback) / max(1, bsz),
            }

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
                # k_eff is always >= 1 (ls_k validated, row non-empty), so [k_eff - 1] is safe.
                # Use the k_eff-th nearest finite distance as local scale sigma_i.
                sigma_i = torch.topk(row_vals, k=k_eff, largest=False).values[k_eff - 1]
                sigma[i] = torch.clamp(sigma_i.detach(), min=self.ls_sigma_min)

            if self.ls_mode == "row":
                logits_feat = -((dist_feat / sigma[:, None]) * self.ls_alpha)
            else:
                # Symmetric self-tuning kernel where dist_feat(i,j)=d(i,j):
                # exp(-dist_feat(i,j)^2 / (sigma_i * sigma_j)).
                denom = sigma[:, None] * sigma[None, :] + self.eps
                logits_feat = -((dist_feat.pow(2)) / denom) * self.ls_alpha

            tau_stats = {
                "sigma_mean": float(sigma.mean().detach().item()),
                "sigma_median": float(sigma.median().detach().item()),
                "ls_alpha": float(self.ls_alpha),
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
        sim_emb_shifted = sim_emb - sim_emb.max(dim=1, keepdim=True).values
        log_prob = F.log_softmax(sim_emb_shifted, dim=1)

        # 9) Compute loss according to loss_mode.
        if self.loss_mode == "softkl":
            loss, mode_stats = self._softkl_loss(sim_feat, log_prob, bsz)
        elif self.loss_mode == "supcon":
            loss, mode_stats = self._supcon_loss(z, dist_feat, finite_mask, cond_mask, bsz, device)
        else:  # hybrid
            loss_soft, soft_stats = self._softkl_loss(sim_feat, log_prob, bsz)
            loss_sup, sup_stats = self._supcon_loss(z, dist_feat, finite_mask, cond_mask, bsz, device)
            loss = self.w_soft * loss_soft + self.w_supcon * loss_sup
            mode_stats = {
                "supcon_loss": float(loss_sup.detach().item()),
                "softkl_loss": float(loss_soft.detach().item()),
            }
            mode_stats.update({f"supcon_{k}": v for k, v in sup_stats.items() if k != "valid_anchors"})

        if not torch.isfinite(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)

        entropy = -(sim_feat * torch.log(sim_feat + self.eps)).sum(dim=1)
        effective_k = 1.0 / (sim_feat.pow(2).sum(dim=1) + self.eps)

        stats: dict = {
            "valid_anchors": int(bsz),
            "sim_feat_entropy_mean": float(entropy.mean().detach().item()),
            "effective_k_mean": float(effective_k.mean().detach().item()),
        }
        stats.update(common_stats)
        stats.update(tau_stats)
        stats.update(cond_stats)
        stats.update(mode_stats)
        return loss, stats

    def _softkl_loss(
        self,
        sim_feat: torch.Tensor,
        log_prob: torch.Tensor,
        bsz: int,
    ) -> tuple[torch.Tensor, dict]:
        """Soft cross-entropy KL loss."""
        loss_vec = -(sim_feat * log_prob).sum(dim=1)
        loss = loss_vec.mean()
        return loss, {"valid_anchors": bsz}

    def _supcon_loss(
        self,
        z: torch.Tensor,
        dist_feat: torch.Tensor,
        finite_mask: torch.Tensor,
        cond_mask: torch.Tensor | None,
        bsz: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict]:
        """Multi-positive SupCon loss: positives = TopK nearest by (gated) feature distance."""
        eye = torch.eye(bsz, dtype=torch.bool, device=device)

        # Determine candidate pool for each anchor.
        if cond_mask is not None:
            # Only use condition-compatible pairs as candidates.
            candidate_mask = cond_mask & (~eye) & finite_mask
        else:
            candidate_mask = (~eye) & finite_mask

        # TopK positives per anchor from the candidate pool.
        k = min(self.pos_topk, bsz - 1)
        # Use +inf for non-candidates so they're never chosen as positives.
        dist_for_topk = dist_feat.masked_fill(~candidate_mask, float("inf"))
        # Build pos_mask: anchor i's positives are the k nearest by feature dist among candidates.
        pos_mask = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)
        if k > 0 and candidate_mask.any():
            # For each anchor, select top-k finite candidates.
            topk_vals, topk_idx = torch.topk(dist_for_topk, k=k, dim=1, largest=False)
            valid_topk = torch.isfinite(topk_vals)  # [B, k]
            # Scatter valid positives into pos_mask.
            for ki in range(k):
                valid_row = valid_topk[:, ki]
                if valid_row.any():
                    row_idx = torch.where(valid_row)[0]
                    col_idx = topk_idx[valid_row, ki]
                    pos_mask[row_idx, col_idx] = True

        # Negatives: remaining candidates (not positives, not self).
        neg_mask = candidate_mask & (~pos_mask)

        loss, sup_stats = multi_positive_infonce(
            z=z,
            pos_mask=pos_mask,
            temperature=self.temperature,
            neg_mask=neg_mask,
        )
        return loss, sup_stats
