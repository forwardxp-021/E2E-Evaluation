#!/usr/bin/env python3
"""Versioned physical input; no legacy tensors or outcome-dependent operations.

Tracks are [x_center,y_center,vx_world,vy_world,heading_world,valid] float64.
All angular velocities differentiate WORLD heading, never relative heading.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

VERSION = 'canonical83_v1'
SLOTS = ['front', 'left_front', 'left_rear', 'right_front', 'right_rear']
EGO_FIELDS = ['x', 'y', 'vx', 'vy', 'heading', 'speed', 'accel', 'yaw_rate']
NEIGHBOR_FIELDS = ['presence', 'rel_x', 'rel_y', 'rel_vx', 'rel_vy', 'distance',
                   'longitudinal_gap', 'lateral_gap', 'closing_speed', 'ttc', 'thw',
                   'speed', 'accel', 'heading_relative', 'yaw_rate_world']
NAMES = EGO_FIELDS + [s + '.' + f for s in SLOTS for f in NEIGHBOR_FIELDS]
PRESENCE = np.array([8+15*j for j in range(5)])
ANGLES = np.array([4]+[21+15*j for j in range(5)])
GRID_US = np.arange(80, dtype=np.int64)*100000


def wrap(x):
    return (np.asarray(x)+np.pi) % (2*np.pi)-np.pi


def rotate(v, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([c*v[..., 0]+s*v[..., 1], -s*v[..., 0]+c*v[..., 1]], -1)


def resample(track, times_us, origin_us):
    """No extrapolation; no invalid endpoint/gap bridging. Integer time lookup."""
    a = np.asarray(track, dtype=np.float64)
    t = np.asarray(times_us, dtype=np.int64)-int(origin_us)
    if len(t) != len(a) or np.any(np.diff(t) <= 0):
        raise ValueError('Track/time mismatch or non-increasing physical time')
    out = np.zeros((80, 6), dtype=np.float64)
    good = (a[:, 5] > .5) & np.isfinite(a[:, :5]).all(1)
    for i, q in enumerate(GRID_US):
        j = int(np.searchsorted(t, q))
        if j < len(t) and t[j] == q and good[j]:
            out[i] = a[j]; out[i, 5] = 1
        elif 0 < j < len(t) and t[j]-t[j-1] <= 150000 and good[j-1] and good[j]:
            w = float(q-t[j-1])/float(t[j]-t[j-1])
            out[i, :4] = (1-w)*a[j-1, :4]+w*a[j, :4]
            out[i, 4] = wrap(a[j-1, 4]+w*wrap(a[j, 4]-a[j-1, 4]))
            out[i, 5] = 1
    return out


def derivative(values, valid, angular=False, ids=None):
    out = np.zeros(80, dtype=np.float64); mask = np.zeros(80, bool)
    mask[1:] = valid[1:] & valid[:-1]
    if ids is not None:
        mask[1:] &= ids[1:] == ids[:-1]
    delta = np.diff(values)
    if angular:
        delta = wrap(delta)
    out[1:] = delta/.1
    out[~mask] = 0
    return out, mask


def canonical(ego, neighbors, slot_ids, lane_known):
    """Assemble already resampled world tracks, preserving slot identities/masks."""
    e = np.asarray(ego, np.float64); n = np.asarray(neighbors, np.float64)
    ids = np.asarray(slot_ids).astype('U64')
    if e.shape != (80, 6) or n.shape != (80, 5, 6) or ids.shape != (80, 5):
        raise ValueError('Expected ego[80,6], neighbor[80,5,6], ids[80,5]')
    ev = (e[:, 5] > .5) & np.isfinite(e[:, :5]).all(1)
    if not ev.any():
        raise ValueError('No valid ego frame')
    i0 = np.flatnonzero(ev)[0]; h0 = e[i0, 4]
    x = np.zeros((80, 83), np.float64); m = np.zeros((80, 83), bool)
    x[:, :2] = rotate(e[:, :2]-e[i0, :2], h0)
    x[:, 2:4] = rotate(e[:, 2:4], h0)
    x[:, 4] = wrap(e[:, 4]-h0); x[:, 5] = np.linalg.norm(e[:, 2:4], axis=1)
    m[:, :6] = ev[:, None]
    x[:, 6], m[:, 6] = derivative(x[:, 5], ev)
    x[:, 7], m[:, 7] = derivative(e[:, 4], ev, True)
    known = np.asarray(lane_known, bool)
    if known.shape == (80,):
        known = np.repeat(known[:,None],5,axis=1)
    if known.shape != (80,5):
        raise ValueError('Lane observability must be [80] or [80,5]')
    known = known & ev[:,None]
    for j in range(5):
        a = n[:, j]; k = 8+15*j
        valid = known[:,j] & (a[:, 5] > .5) & np.isfinite(a[:, :5]).all(1) & (ids[:, j] != '')
        x[:, k] = valid; m[:, k] = known[:,j]
        dp = a[:, :2]-e[:, :2]; dv = a[:, 2:4]-e[:, 2:4]
        rp, rv = rotate(dp, e[:, 4]), rotate(dv, e[:, 4])
        d = np.linalg.norm(dp, axis=1)
        closing = -np.sum(dp*dv, axis=1)/np.maximum(d, 1e-6)
        forward = rotate(e[:, 2:4], e[:, 4])[:, 0]
        x[:, k+1:k+3] = rp; x[:, k+3:k+5] = rv; x[:, k+5] = d
        x[:, k+6:k+8] = rp; x[:, k+8] = closing
        x[:, k+9:k+11] = 999
        tc = valid & (closing > 1e-3) & (d > 1e-6)
        hw = valid & (rp[:, 0] > 0) & (forward > 1e-3)
        x[tc, k+9] = np.minimum(d[tc]/closing[tc], 999)
        x[hw, k+10] = np.minimum(rp[hw, 0]/forward[hw], 999)
        x[:, k+11] = np.linalg.norm(a[:, 2:4], axis=1)
        x[:, k+12], am = derivative(x[:, k+11], valid, ids=ids[:, j])
        x[:, k+13] = wrap(a[:, 4]-e[:, 4])
        x[:, k+14], ym = derivative(a[:, 4], valid, True, ids[:, j])
        m[:, k+1:k+15] = valid[:, None]
        m[:, k+9] = tc; m[:, k+10] = hw; m[:, k+12] = am; m[:, k+14] = ym
    # Invalid TTC/THW retain an explicit sentinel; all other invalid numbers zero.
    sentinel = np.zeros(83, bool)
    sentinel[[17+15*j for j in range(5)]+[18+15*j for j in range(5)]] = True
    x[~m & ~sentinel[None, :]] = 0
    return dict(x=x.astype(np.float32), channel_valid=m, ego_valid=ev,
                slot_ids=ids, time_us=GRID_US.copy(), version=VERSION)


def assign_slots(ego, tracks, lane_infos):
    """Actual production lane-aware selector; no geometry-only fallback."""
    from tools.lane_aware_assignment import assign_neighbors_lane_aware
    n = np.zeros((80, 5, 6)); ids = np.full((80, 5), '', dtype='U64'); known = np.zeros((80,5), bool)
    cfg = dict(allow_geometric_adjacent_lane_inference=False)
    for t in range(80):
        if ego[t, 5] <= .5:
            continue
        def state(a):
            return dict(x=a[0], y=a[1], velocity_x=a[2], velocity_y=a[3],
                        heading=a[4], speed=float(np.hypot(a[2], a[3])), valid=True)
        candidates = {k: state(v[t]) for k, v in tracks.items() if v[t, 5] > .5}
        result = assign_neighbors_lane_aware(state(ego[t]), candidates, lane_infos,
                                            'lane_aware_only', cfg)
        known[t] = result.lane_assignment_available
        lane=lane_infos.get(result.current_lane_id) if lane_infos else None
        if lane is not None and lane.topology_source == 'nuplan_connector_adjacent_unknown':
            known[t,1:] = False
        for j, slot in enumerate(SLOTS):
            aid = result.slot_to_agent.get(slot)
            if aid is not None:
                ids[t, j] = aid; n[t, j] = tracks[aid][t]
    return canonical(ego, n, ids, known)


def nuplan_lane_infos(lane_objects, connector_objects=()):
    """Versioned map adapter for official Lane/LaneConnector objects.

    nuPlan adjacency is whole-lane, not Waymo's per-polyline index relations.
    Connector adjacency is unknown in the official API: side slots are masked,
    never inferred from nearby parallel geometry. This distinction is deliberate.
    """
    from tools.waymo_lane_utils import LaneInfo, _build_lane_geom
    lanes={}
    for connector,objects in [(False,lane_objects),(True,connector_objects)]:
        for obj in objects:
            pts=np.array([[p.x,p.y] for p in obj.baseline_path.discrete_path],np.float64)
            if len(pts)<2:
                continue
            geom=_build_lane_geom(pts)
            if geom is None:
                continue
            names=['seg_heading','seg_len','s_prefix','seg_start_xy','seg_vec_xy','seg_den',
                   'bbox_min_xy','bbox_max_xy','bbox_center_xy']
            left,right=(None,None) if connector else obj.adjacent_edges
            relation=lambda nb: [] if nb is None else [dict(lane_id=str(nb.id),self_start_index=0,
                self_end_index=len(pts)-2,neighbor_start_index=0,
                neighbor_end_index=len(nb.baseline_path.discrete_path)-2)]
            lanes[str(obj.id)]=LaneInfo(lane_id=str(obj.id),centerline_xy=pts,**dict(zip(names,geom)),
                left_neighbor_lane_ids=[] if left is None else [str(left.id)],
                right_neighbor_lane_ids=[] if right is None else [str(right.id)],
                entry_lane_ids=[str(x.id) for x in obj.incoming_edges],exit_lane_ids=[str(x.id) for x in obj.outgoing_edges],
                topology_source='nuplan_connector_adjacent_unknown' if connector else 'nuplan_whole_lane_adjacency',
                left_neighbor_relations=relation(left),right_neighbor_relations=relation(right))
    return lanes


def waymo_tracks(scenario):
    tracks = {}
    fields = ['center_x', 'center_y', 'velocity_x', 'velocity_y', 'heading']
    for tr in scenario.tracks:
        if tr.object_type != 1:
            continue
        a = np.array([[getattr(s, f) for f in fields]+[
            s.valid and all(s.HasField(f) for f in fields)] for s in tr.states], np.float64)
        tracks[str(tr.id)] = a
    return tracks


def from_waymo(scenario, ego_id, start=0):
    from tools.waymo_lane_utils import extract_lane_polylines
    times = np.rint(np.array(scenario.timestamps_seconds, np.float64)*1e6).astype(np.int64)
    tracks = {k: resample(a, times, times[start]) for k, a in waymo_tracks(scenario).items()}
    ego = tracks.pop(str(ego_id))
    return assign_slots(ego, tracks, extract_lane_polylines(scenario))


def from_nuplan(ego_states, observations, lane_infos, ego_velocity_frame):
    """Official model EgoState velocities are BODY-frame; tracked agents are WORLD.

    DB/reconstructed EgoStates need an explicit evidenced frame declaration. No
    speed/yaw synthesis or undocumented automatic frame inference is permitted.
    """
    if ego_velocity_frame not in ('body', 'world'):
        raise ValueError('Explicit evidenced ego velocity frame body/world is required')
    if len(ego_states) != len(observations):
        raise ValueError('Ego/observation same-index length mismatch')
    times = np.array([s.time_point.time_us for s in ego_states], np.int64)
    e = []; tracks = {}
    for i, (s, obs) in enumerate(zip(ego_states, observations)):
        c = s.center; v = s.dynamic_car_state.center_velocity_2d
        vel = np.array([v.x, v.y])
        if ego_velocity_frame == 'body':
            vel = rotate(vel, -c.heading)
        e.append([c.x, c.y, *vel, c.heading, 1])
        for a in obs.tracked_objects.tracked_objects:
            if a.tracked_object_type.name != 'VEHICLE':
                continue
            aid = a.track_token
            if not aid:
                raise ValueError('nuPlan neighbor has no stable track token')
            if aid not in tracks:
                tracks[aid] = np.zeros((len(times), 6))
            tracks[aid][i] = [a.center.x, a.center.y, a.velocity.x, a.velocity.y, a.center.heading, 1]
    ego = resample(e, times, times[0])
    n = {k: resample(v, times, times[0]) for k, v in tracks.items()}
    return assign_slots(ego, n, lane_infos)


def semantic13(sample):
    x, m = sample['x'].astype(np.float64), sample['channel_valid']
    out = np.zeros(13); mask = np.zeros(13, bool)
    def put(k, a, valid, fn):
        v = a[valid]
        if len(v) >= 2:
            out[k] = fn(v); mask[k] = True
    rms = lambda a: np.sqrt(np.mean(a*a))
    put(0, x[:, 5], m[:, 5], np.mean); put(1, x[:, 5], m[:, 5], np.std)
    put(2, x[:, 5], m[:, 5], lambda a: np.quantile(a, .95))
    if m[0, 5] and m[-1, 5]:
        out[3] = x[-1, 5]-x[0, 5]; mask[3] = True
    for k, fn in [(4, rms), (5, lambda a: np.mean(np.abs(a))), (6, lambda a: np.quantile(np.abs(a), .95))]:
        put(k, x[:, 6], m[:, 6], fn)
    jerk, jm = derivative(x[:, 6], m[:, 6])
    put(7, jerk, jm, rms); put(8, jerk, jm, lambda a: np.quantile(np.abs(a), .95))
    put(9, x[:, 7], m[:, 7], rms); put(10, x[:, 7], m[:, 7], lambda a: np.mean(np.abs(a)))
    if sample['ego_valid'].all():
        out[11] = np.abs(wrap(np.diff(x[:, 4]))).sum()
        out[12] = np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1).sum()
        mask[11:] = True
    return out.astype(np.float32), mask
