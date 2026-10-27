import numpy as np
import pytest
from tools.rbr_v2_d6_contract import CELLS,draw,validate_pool


def test_fair_unpaired_source_draw_fixture():
    rows=[dict(session_id=c+'_'+str(i),cell=c) for c in CELLS for i in range(10)]
    a,b,t=draw(rows,np.random.RandomState(6),ab=True)
    assert len(a)==len(b)==20 and not set(a)&set(b) and t.sum()==10
    for cell in CELLS:assert sum(s.startswith(cell+'_') for s in a)==sum(s.startswith(cell+'_') for s in b)==5
    assert draw(rows,np.random.RandomState(6),ab=False)[2].sum()==0


def test_support_and_role_fail_closed():
    rows=[dict(session_id='s'+str(i),cell=CELLS[1]) for i in range(80)]
    with pytest.raises(ValueError,match='NOT_ASSESSABLE'):validate_pool(rows,'D_METADATA')
    with pytest.raises(PermissionError):validate_pool(rows,'V_AA')


def test_source_alias_and_unknown_cell_rejected():
    with pytest.raises(ValueError,match='Duplicate'):validate_pool([dict(session_id='same',cell=c) for c in CELLS],'D_FIXTURE')
    with pytest.raises(ValueError,match='Unknown'):validate_pool([dict(session_id='s',cell='new_cell')],'D_FIXTURE')
