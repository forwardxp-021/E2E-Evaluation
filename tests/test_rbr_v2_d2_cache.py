import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import rbr_v2_d2_round2 as cache

class CacheTests(unittest.TestCase):
 def test_lossless_own_source_roundtrip_and_mtime(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'evidence';out.mkdir();local=Path(tmp)/'local';local.mkdir();(out/'static').mkdir()
   p=local/'fixture.db';data=b'SQLite format 3\0'+bytes(range(256))*200;h=hashlib.sha256(data).hexdigest();p.write_bytes(data);mtime=p.stat().st_mtime_ns
   (out/'static/x.json').write_text(json.dumps(dict(status='EXCLUDED',session_id='NONSClENTIFIC_FIXTURE',log=dict(source_file=str(p)),download_sha256=h)))
   with patch.object(cache,'OUT',out),patch.object(cache,'LOCAL',local):
    cache.pack_excluded();self.assertFalse(p.exists());q=cache.restore(p);self.assertEqual(q.read_bytes(),data);self.assertEqual(q.stat().st_mtime_ns,mtime)
 def test_corrupt_archive_refused(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);(out/'cache_transforms').mkdir();archive=out/'bad';archive.write_bytes(b'corrupt');p=out/'x.db'
   (out/'cache_transforms/x.db.json').write_text(json.dumps(dict(raw_path=str(p),archive_path=str(archive),archive_sha256='0'*64,raw_bytes=50,raw_sha256='1'*64)))
   with patch.object(cache,'OUT',out),patch.object(cache,'LOCAL',out):
    with self.assertRaisesRegex(ValueError,'archive changed'):cache.restore(p)

if __name__=='__main__':unittest.main()
