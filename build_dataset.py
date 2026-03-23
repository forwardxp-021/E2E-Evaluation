import tensorflow as tf
import numpy as np
import glob
import os
from waymo_open_dataset.protos import scenario_pb2

WINDOW = 50

def extract_ego_and_front(scenario):
    if scenario.sdc_track_index is None:
        return None, None

    ego_track = scenario.tracks[scenario.sdc_track_index]

    ego = []
    for s in ego_track.states:
        if s.valid:
            ego.append([s.center_x, s.center_y, s.velocity_x, s.velocity_y])
    ego = np.array(ego)

    if len(ego) == 0:
        return ego, None

    # 改进：识别同车道的前向车辆
    front = None
    min_forward_dist = float('inf')

    for i, track in enumerate(scenario.tracks):
        if i == scenario.sdc_track_index:
            continue

        states = track.states
        forward_dists = []
        valid_count = 0

        for t in range(min(len(states), len(ego_track.states))):
            if states[t].valid and ego_track.states[t].valid:
                # 计算自车到其他车辆的向量
                dx = states[t].center_x - ego_track.states[t].center_x
                dy = states[t].center_y - ego_track.states[t].center_y
                
                # 计算自车行驶方向向量
                if t < len(ego_track.states) - 1 and ego_track.states[t+1].valid:
                    ego_dx = ego_track.states[t+1].center_x - ego_track.states[t].center_x
                    ego_dy = ego_track.states[t+1].center_y - ego_track.states[t].center_y
                else:
                    # 如果没有下一个状态，使用当前速度方向
                    ego_dx = ego_track.states[t].velocity_x
                    ego_dy = ego_track.states[t].velocity_y
                
                # 计算向量点积，判断是否在自车前方
                dot_product = dx * ego_dx + dy * ego_dy
                
                # 计算距离
                dist = np.sqrt(dx*dx + dy*dy)
                
                # 只有在自车前方且距离合理的车辆才被考虑
                if dot_product > 0 and dist < 100:  # 100米以内的前向车辆
                    forward_dists.append(dist)
                    valid_count += 1

        if valid_count > 0:
            mean_forward_dist = np.mean(forward_dists)
            if mean_forward_dist < min_forward_dist:
                min_forward_dist = mean_forward_dist
                front = track

    if front is None:
        return ego, None

    front_traj = []
    for s in front.states:
        if s.valid:
            front_traj.append([s.center_x, s.center_y, s.velocity_x, s.velocity_y])

    return ego, np.array(front_traj)


def compute_features(ego, front):
    if front is None or len(front) < len(ego):
        return None

    ego_speed = np.linalg.norm(ego[:, 2:4], axis=1)
    front_speed = np.linalg.norm(front[:len(ego), 2:4], axis=1)

    rel_speed = ego_speed - front_speed

    ego_acc = np.diff(ego_speed, prepend=ego_speed[0])
    front_acc = np.diff(front_speed, prepend=front_speed[0])

    rel_acc = ego_acc - front_acc

    dist = np.linalg.norm(ego[:, :2] - front[:len(ego), :2], axis=1)
    thw = dist / (ego_speed + 1e-3)

    jerk = np.diff(ego_acc, prepend=ego_acc[0])

    # =========================
    # 1️⃣ reaction time（简化版）
    # =========================
    front_brake = front_acc < -0.5
    ego_brake = ego_acc < -0.5

    reaction_time = np.nan  # 默认为NaN表示无效
    for i in range(len(front_brake)):
        if front_brake[i]:
            for j in range(i, min(i+10, len(ego_brake))):
                if ego_brake[j]:
                    reaction_time = j - i
                    break
            break
    
    # 如果没有检测到反应时间，使用中位值替代
    if np.isnan(reaction_time):
        reaction_time = 0.0

    # =========================
    # 2️⃣ yaw_rate（横向）
    # =========================
    dx = np.diff(ego[:, 0], prepend=ego[0,0])
    dy = np.diff(ego[:, 1], prepend=ego[0,1])
    heading = np.arctan2(dy, dx)
    yaw_rate = np.diff(heading, prepend=heading[0])

    # =========================
    # 3️⃣ lane change（简化）
    # =========================
    lane_change_count = np.sum(np.abs(yaw_rate) > 0.1)
    lane_change_duration = np.mean(np.abs(yaw_rate) > 0.1)

    # =========================
    # 4️⃣ speed normalization
    # =========================
    traffic_speed = front_speed  # 简化
    speed_norm = ego_speed / (traffic_speed + 1e-3)

    # =========================
    # FINAL FEATURE (19D)
    # =========================
    features = [

        # rel speed (3)
        rel_speed.mean(),
        rel_speed.std(),
        np.mean(rel_speed > 0),

        # thw (3)
        thw.mean(),
        thw.std(),
        thw.min(),

        # jerk (3)
        jerk.mean(),
        jerk.std(),
        np.percentile(jerk, 95),

        # rel acc (2)
        rel_acc.mean(),
        rel_acc.std(),

        # reaction (1)
        reaction_time,

        # lateral (3)
        yaw_rate.std(),
        lane_change_count,
        lane_change_duration,

        # env norm (2)
        speed_norm.mean(),
        speed_norm.std(),

        # stability (2)
        ego_speed.std(),
        ego_acc.std()
    ]

    return np.array(features)


traj_data = []
feat_data = []

# 数据路径配置：默认读取仓库内 data/，也可通过 WAYMO_DATA_PATH 覆盖
repo_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
data_path = os.getenv('WAYMO_DATA_PATH', repo_data_dir)
files = sorted(glob.glob(os.path.join(data_path, '**', '*.tfrecord*'), recursive=True))

if not files:
    print(f"Warning: No tfrecord files found in {data_path}")
    print("Please upload TFRecord files under data/ or set WAYMO_DATA_PATH to your data directory")

for file in files:
    dataset_tf = tf.data.TFRecordDataset(file)

    for data in dataset_tf:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())

        ego, front = extract_ego_and_front(scenario)

        if ego is None or len(ego) < WINDOW:
            continue

        for j in range(len(ego) - WINDOW):
            seg_ego = ego[j:j+WINDOW]
            seg_front = front[j:j+WINDOW] if front is not None else None

            feat = compute_features(seg_ego, seg_front)

            if feat is None:
                continue

            traj_data.append(seg_ego)
            feat_data.append(feat)

traj_data = np.array(traj_data)
feat_data = np.array(feat_data)

# 特征标准化
print("Standardizing features...")
feat_data = (feat_data - feat_data.mean(axis=0)) / (feat_data.std(axis=0) + 1e-6)
print("Feature standardization completed.")

np.save("traj.npy", traj_data)
np.save("feat.npy", feat_data)

print("Saved:", traj_data.shape, feat_data.shape)