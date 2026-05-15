#!/usr/bin/env python3
import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def sanitize_track_window(window, dt, track_label='', max_invalid_ratio=0.2):
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        return None, None, {'reason': 'bad_shape', 'track': track_label}

    xy = arr[:, :2]
    vv = arr[:, 2:4]
    hd = arr[:, 4]
    valid = np.asarray(arr[:, 5] > 0.5, dtype=np.float32)

    finite_xy = np.isfinite(xy).all(axis=1)
    finite_vv = np.isfinite(vv).all(axis=1)
    valid_mask = (valid > 0.5) & finite_xy & finite_vv

    invalid_ratio = 1.0 - float(np.mean(valid_mask))
    if invalid_ratio > max_invalid_ratio:
        return None, valid_mask.astype(np.float32), {'reason': 'too_many_invalid', 'track': track_label, 'invalid_ratio': invalid_ratio}

    # fill x,y,vx,vy with linear interpolation over valid_mask
    for c in range(4):
        col = arr[:, c]
        bad = ~np.isfinite(col)
        col[bad] = np.nan
        if np.all(~np.isnan(col)):
            pass
        elif np.all(np.isnan(col)):
            col[:] = 0.0
        else:
            idx = np.arange(len(col))
            good = ~np.isnan(col)
            col[:] = np.interp(idx, idx[good], col[good])
        arr[:, c] = col

    hd = arr[:, 4]
    hd_bad = ~np.isfinite(hd)
    hd[hd_bad] = np.nan
    if np.all(np.isnan(hd)):
        hd = np.arctan2(arr[:, 3], arr[:, 2])
    else:
        idx = np.arange(len(hd))
        good = ~np.isnan(hd)
        hd = np.interp(idx, idx[good], hd[good])
    arr[:, 4] = wrap_angle(hd)
    arr[:, 5] = valid_mask.astype(np.float32)

    return arr[:, :6].astype(np.float32), valid_mask.astype(np.float32), {'reason': 'ok', 'track': track_label, 'invalid_ratio': invalid_ratio}


def localize(xy, origin, base_heading):
    pts = np.asarray(xy, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    c, s = np.cos(base_heading), np.sin(base_heading)
    rot = np.array([[c, s], [-s, c]], dtype=np.float32)
    return (pts - origin[None, :]) @ rot.T


LANE_DEBUG_FIELDS = [
    'slot_name', 'candidate_agent_id', 'selected_agent_id', 'method',
    'distance', 'lateral_offset', 'heading_diff_deg', 'score', 'reason'
]


def normalize_debug_row(row):
    out = {k: '' for k in LANE_DEBUG_FIELDS}
    if isinstance(row, dict):
        for k in LANE_DEBUG_FIELDS:
            v = row.get(k, '')
            out[k] = '' if v is None else v
    return out
