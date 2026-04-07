import argparse
import csv
import glob
import hashlib
import os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2


def align_tracks(ego_track, front_track):
    seq_ego, seq_front = [], []
    T = min(len(ego_track.states), len(front_track.states))
    for t in range(T):
        ego_state = ego_track.states[t]
        front_state = front_track.states[t]
        if ego_state.valid and front_state.valid:
            seq_ego.append([
                ego_state.center_x,
                ego_state.center_y,
                ego_state.velocity_x,
                ego_state.velocity_y,
            ])
            seq_front.append([
                front_state.center_x,
                front_state.center_y,
                front_state.velocity_x,
                front_state.velocity_y,
            ])
    return np.asarray(seq_ego), np.asarray(seq_front)


def extract_ego_and_front(scenario):
    if scenario.sdc_track_index is None:
        return None, None, -1

    ego_track = scenario.tracks[scenario.sdc_track_index]
    if len(ego_track.states) == 0:
        return np.empty((0, 4)), None, -1

    front = None
    front_id = -1
    min_forward_dist = 999.0

    for i, track in enumerate(scenario.tracks):
        if i == scenario.sdc_track_index:
            continue

        forward_dists = []
        valid_count = 0
        T = min(len(track.states), len(ego_track.states))

        for t in range(T):
            e = ego_track.states[t]
            o = track.states[t]
            if not (e.valid and o.valid):
                continue

            dx = o.center_x - e.center_x
            dy = o.center_y - e.center_y

            if t < len(ego_track.states) - 1 and ego_track.states[t + 1].valid:
                ego_dx = ego_track.states[t + 1].center_x - e.center_x
                ego_dy = ego_track.states[t + 1].center_y - e.center_y
            else:
                ego_dx = e.velocity_x
                ego_dy = e.velocity_y

            dot = dx * ego_dx + dy * ego_dy
            dist = np.sqrt(dx * dx + dy * dy)
            if dot > 0 and dist < 100:
                forward_dists.append(dist)
                valid_count += 1

        if valid_count > 0:
            mean_forward_dist = float(np.mean(forward_dists))
            if mean_forward_dist < min_forward_dist:
                min_forward_dist = mean_forward_dist
                front = track
                front_id = i

    if front is None:
        return np.empty((0, 4)), None, -1

    ego_aligned, front_aligned = align_tracks(ego_track, front)
    return ego_aligned, front_aligned, front_id


def compute_features(ego, front):
    if front is None or len(front) < len(ego):
        return None

    ego_speed = np.linalg.norm(ego[:, 2:4], axis=1)
    front_speed = np.linalg.norm(front[: len(ego), 2:4], axis=1)

    rel_speed = ego_speed - front_speed
    ego_acc = np.diff(ego_speed, prepend=ego_speed[0])
    front_acc = np.diff(front_speed, prepend=front_speed[0])
    rel_acc = ego_acc - front_acc

    dist = np.linalg.norm(ego[:, :2] - front[: len(ego), :2], axis=1)
    thw = dist / (ego_speed + 1e-3)
    jerk = np.diff(ego_acc, prepend=ego_acc[0])

    front_brake = front_acc < -0.5
    ego_brake = ego_acc < -0.5
    reaction_time = 0.0
    for i in range(len(front_brake)):
        if front_brake[i]:
            for j in range(i, min(i + 10, len(ego_brake))):
                if ego_brake[j]:
                    reaction_time = j - i
                    break
            break

    dx = np.diff(ego[:, 0], prepend=ego[0, 0])
    dy = np.diff(ego[:, 1], prepend=ego[0, 1])
    heading = np.arctan2(dy, dx)
    yaw_rate = np.diff(heading, prepend=heading[0])

    lane_change_count = np.sum(np.abs(yaw_rate) > 0.1)
    lane_change_duration = np.mean(np.abs(yaw_rate) > 0.1)

    speed_norm = ego_speed / (front_speed + 1e-3)

    features = [
        rel_speed.mean(),
        rel_speed.std(),
        np.mean(rel_speed > 0),
        thw.mean(),
        thw.std(),
        thw.min(),
        jerk.mean(),
        jerk.std(),
        np.percentile(jerk, 95),
        rel_acc.mean(),
        rel_acc.std(),
        reaction_time,
        yaw_rate.std(),
        lane_change_count,
        lane_change_duration,
        speed_norm.mean(),
        speed_norm.std(),
        ego_speed.std(),
        ego_acc.std(),
        ego_speed.mean(),
    ]
    return np.asarray(features, dtype=np.float32)


def get_ego_speeds(scenario):
    ego_idx = scenario.sdc_track_index
    if ego_idx is None or ego_idx < 0 or ego_idx >= len(scenario.tracks):
        return np.array([])
    ego_track = scenario.tracks[ego_idx]
    speeds = []
    for st in ego_track.states:
        if st.valid:
            speeds.append(np.hypot(st.velocity_x, st.velocity_y))
    return np.asarray(speeds)


def assign_split(scenario_id, train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    digest = hashlib.md5(str(scenario_id).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def write_summary(output_dir, summary_map):
    output_dir = Path(output_dir)
    txt_path = output_dir / "summary.txt"
    csv_path = output_dir / "summary.csv"

    lines = ["Dataset summary"] + [f"{k}: {v}" for k, v in summary_map.items()]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in summary_map.items():
            writer.writerow([k, v])


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build traj/feat/meta/split from Waymo TFRecords")
    parser.add_argument("--tfrecord_glob", type=str, default="/mnt/d/WMdata/*.tfrecord-*")
    parser.add_argument("--output_dir", type=str, default=str(root / "output"))
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--min_ego_speed", type=float, default=5.5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(args.tfrecord_glob))
    if args.limit_files is not None:
        files = files[: args.limit_files]
    print(f"Found {len(files)} TFRecord files via glob: {args.tfrecord_glob}")

    traj_data, feat_data, meta_data, split_data = [], [], [], []
    all_speeds = []
    filtered_scenarios = 0
    kept_scenarios = 0

    for file in files:
        dataset_tf = tf.data.TFRecordDataset(file)
        for data in dataset_tf:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())

            speeds = get_ego_speeds(scenario)
            if len(speeds) == 0 or np.min(speeds) < args.min_ego_speed:
                filtered_scenarios += 1
                continue

            ego, front, front_id = extract_ego_and_front(scenario)
            if ego is None or len(ego) == 0:
                continue

            feat = compute_features(ego, front)
            if feat is None:
                continue

            split = assign_split(scenario.scenario_id, args.train_ratio, args.val_ratio, args.test_ratio)

            traj_data.append(ego)
            feat_data.append(feat)
            meta_data.append((scenario.scenario_id, 0, -1 if front is None else front_id))
            split_data.append(split)
            all_speeds.extend(speeds.tolist())
            kept_scenarios += 1

            if args.log_every and kept_scenarios % args.log_every == 0:
                print(f"Processed {kept_scenarios} kept scenarios...")

    traj_data = np.asarray(traj_data, dtype=object)
    feat_data = np.asarray(feat_data, dtype=np.float32)
    meta_data = np.asarray(meta_data, dtype=object)
    split_data = np.asarray(split_data, dtype=object)

    if len(feat_data) == 0:
        raise RuntimeError("No valid scenarios after filtering. Try lowering --min_ego_speed")

    feat_data = (feat_data - feat_data.mean(axis=0)) / (feat_data.std(axis=0) + 1e-6)

    split_counter = Counter(split_data.tolist())

    traj_path = os.path.join(args.output_dir, "traj.npy")
    feat_path = os.path.join(args.output_dir, "feat.npy")
    meta_path = os.path.join(args.output_dir, "meta.npy")
    split_path = os.path.join(args.output_dir, "split.npy")

    np.save(traj_path, traj_data)
    np.save(feat_path, feat_data)
    np.save(meta_path, meta_data)
    np.save(split_path, split_data)

    speeds_np = np.asarray(all_speeds, dtype=float) if all_speeds else np.asarray([0.0])
    summary_map = {
        "tfrecord_glob": args.tfrecord_glob,
        "min_ego_speed": args.min_ego_speed,
        "filtered_out_scenarios": filtered_scenarios,
        "kept_scenarios": kept_scenarios,
        "speed_mean": f"{speeds_np.mean():.6f}",
        "speed_std": f"{speeds_np.std():.6f}",
        "speed_min": f"{speeds_np.min():.6f}",
        "speed_max": f"{speeds_np.max():.6f}",
        "train_count": split_counter.get("train", 0),
        "val_count": split_counter.get("val", 0),
        "test_count": split_counter.get("test", 0),
        "traj_shape": str(traj_data.shape),
        "feat_shape": str(feat_data.shape),
        "meta_shape": str(meta_data.shape),
        "split_shape": str(split_data.shape),
    }
    write_summary(args.output_dir, summary_map)

    print(f"Saved traj to {traj_path}")
    print(f"Saved feat to {feat_path}")
    print(f"Saved meta to {meta_path}")
    print(f"Saved split to {split_path}")
    print("Shapes:", traj_data.shape, feat_data.shape, meta_data.shape, split_data.shape)


if __name__ == "__main__":
    main()
