#!/usr/bin/env python3
"""Executable D6 candidate roster/draw validation; no science entry and no V/C access."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
CELLS=('5to10_empty','5to10_busy','10to15_empty','10to15_busy')


def validate_pool(rows,role):
    if not role.startswith('D_'):raise PermissionError('D6 rejects Primary/V/C execution')
    ids=[r['session_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate SESSION; chunks/scenarios cannot add sources')
    if any(r['cell'] not in CELLS for r in rows):raise ValueError('Unknown/missing fixed task cell')
    counts={c:sum(r['cell']==c for r in rows) for c in CELLS}
    if any(v<10 for v in counts.values()):raise ValueError('NOT_ASSESSABLE: require10 distinct SESSION per cell; no empty-cell renormalization')
    return counts


def draw(rows,rng,role='D_FIXTURE',ab=False):
    validate_pool(rows,role);left=[];right=[]
    for c in CELLS:
        pool=[r['session_id'] for r in rows if r['cell']==c];chosen=rng.choice(pool,10,replace=False)
        left.extend(chosen[:5]);right.extend(chosen[5:])
    treatment=np.zeros(20,bool)
    if ab:treatment[rng.choice(20,10,replace=False)]=True
    return left,right,treatment
