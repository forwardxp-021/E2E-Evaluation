from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class PairIndex:
    pos_index: np.ndarray  # [N, K_pos]
    neg_index: np.ndarray  # [N, K_neg]


def _pairwise_distance(feat: np.ndarray, metric: str = "cosine") -> np.ndarray:
    feat = feat.astype(np.float32, copy=False)
    if metric == "cosine":
        norm = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12
        feat_n = feat / norm
        sim = feat_n @ feat_n.T
        dist = 1.0 - sim
    elif metric == "l2":
        sq = np.sum(feat * feat, axis=1, keepdims=True)
        dist2 = sq + sq.T - 2.0 * (feat @ feat.T)
        dist2 = np.maximum(dist2, 0.0)
        dist = np.sqrt(dist2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return dist


def precompute_knn_pairs(
    feat: np.ndarray,
    k_pos: int = 8,
    k_neg: int = 32,
    metric: str = "cosine",
) -> PairIndex:
    n = feat.shape[0]
    if n <= max(k_pos, k_neg):
        raise ValueError(f"N={n} is too small for k_pos={k_pos}, k_neg={k_neg}")

    dist = _pairwise_distance(feat, metric=metric)

    row_idx = np.arange(n)

    # positives: nearest neighbors in feature distance
    pos_work = dist.copy()
    pos_work[row_idx, row_idx] = np.inf
    pos_part = np.argpartition(pos_work, kth=k_pos - 1, axis=1)[:, :k_pos]
    pos_vals = np.take_along_axis(pos_work, pos_part, axis=1)
    pos_ord = np.argsort(pos_vals, axis=1)
    pos_index = np.take_along_axis(pos_part, pos_ord, axis=1)

    # negatives: farthest neighbors in feature distance
    neg_work = dist.copy()
    neg_work[row_idx, row_idx] = -np.inf
    neg_part = np.argpartition(-neg_work, kth=k_neg - 1, axis=1)[:, :k_neg]
    neg_vals = np.take_along_axis(neg_work, neg_part, axis=1)
    neg_ord = np.argsort(-neg_vals, axis=1)
    neg_index = np.take_along_axis(neg_part, neg_ord, axis=1)

    return PairIndex(pos_index=pos_index.astype(np.int64), neg_index=neg_index.astype(np.int64))


class TrajFeatureDataset(Dataset):
    def __init__(
        self,
        traj_path: str,
        feat_path: str,
        split_path: str,
        k_pos: int = 8,
        k_neg: int = 32,
        metric: str = "cosine",
        pair_cache_path: str | None = None,
        build_pairs: bool = True,
        feat_raw_path: str | None = None,
    ):
        traj_loaded = np.load(traj_path, allow_pickle=True)
        # Some serialized object arrays contain nested object dtypes per sample.
        # Convert once at load time so collate can safely build float tensors.
        if isinstance(traj_loaded, np.ndarray) and traj_loaded.dtype == object:
            self.traj = [np.asarray(t, dtype=np.float32) for t in traj_loaded]
        else:
            self.traj = [np.asarray(t, dtype=np.float32) for t in traj_loaded]
        self.feat = np.load(feat_path, allow_pickle=False).astype(np.float32)
        self.feat_valid = None
        if feat_raw_path is not None:
            feat_raw = np.load(feat_raw_path, allow_pickle=False).astype(np.float32)
            if feat_raw.shape != self.feat.shape:
                raise ValueError(
                    f"feat_raw shape {feat_raw.shape} must match feat shape {self.feat.shape}"
                )
            # Valid mask for missing-aware distance: only finite raw dimensions are usable.
            # Keep float32 for direct tensor batching/device transfer and loss-side arithmetic.
            self.feat_valid = np.isfinite(feat_raw).astype(np.float32)
        self.split = np.load(split_path, allow_pickle=True)

        if not (len(self.traj) == len(self.feat) == len(self.split)):
            raise ValueError("traj/feat/split lengths must match")

        self.pair_index = None
        if build_pairs:
            self.pair_index = self._load_or_build_pairs(
                pair_cache_path=pair_cache_path,
                k_pos=k_pos,
                k_neg=k_neg,
                metric=metric,
            )

    def _load_or_build_pairs(
        self,
        pair_cache_path: str | None,
        k_pos: int,
        k_neg: int,
        metric: str,
    ) -> PairIndex:
        if pair_cache_path is not None:
            p = Path(pair_cache_path)
            if p.exists():
                cache = np.load(p, allow_pickle=False)
                return PairIndex(pos_index=cache["pos_index"], neg_index=cache["neg_index"])

        pair_index = precompute_knn_pairs(self.feat, k_pos=k_pos, k_neg=k_neg, metric=metric)

        if pair_cache_path is not None:
            p = Path(pair_cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, pos_index=pair_index.pos_index, neg_index=pair_index.neg_index)

        return pair_index

    def indices_by_split(self, split_name: str) -> np.ndarray:
        return np.where(self.split == split_name)[0]

    def __len__(self) -> int:
        return len(self.traj)

    def __getitem__(self, index: int) -> dict:
        if self.pair_index is None:
            pos_global = np.empty((0,), dtype=np.int64)
            neg_global = np.empty((0,), dtype=np.int64)
        else:
            pos_global = self.pair_index.pos_index[index]
            neg_global = self.pair_index.neg_index[index]

        return {
            "traj": self.traj[index],
            "feat": self.feat[index],
            "feat_valid": None if self.feat_valid is None else self.feat_valid[index],
            "global_idx": index,
            "pos_global": pos_global,
            "neg_global": neg_global,
            "split": self.split[index],
        }


def collate_variable_traj(batch: list[dict]) -> dict:
    traj_list = [torch.as_tensor(item["traj"], dtype=torch.float32) for item in batch]
    lengths = torch.tensor([x.shape[0] for x in traj_list], dtype=torch.long)
    traj = pad_sequence(traj_list, batch_first=True)

    feat = torch.as_tensor(np.stack([item["feat"] for item in batch], axis=0), dtype=torch.float32)
    feat_valid = None
    if batch[0]["feat_valid"] is not None:
        feat_valid = torch.as_tensor(np.stack([item["feat_valid"] for item in batch], axis=0), dtype=torch.float32)
    global_idx = torch.as_tensor([item["global_idx"] for item in batch], dtype=torch.long)
    pos_global = torch.as_tensor(np.stack([item["pos_global"] for item in batch], axis=0), dtype=torch.long)
    neg_global = torch.as_tensor(np.stack([item["neg_global"] for item in batch], axis=0), dtype=torch.long)

    return {
        "traj": traj,
        "lengths": lengths,
        "feat": feat,
        "feat_valid": feat_valid,
        "global_idx": global_idx,
        "pos_global": pos_global,
        "neg_global": neg_global,
    }
