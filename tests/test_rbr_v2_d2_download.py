import io,unittest,zipfile
from unittest.mock import patch
from tools import rbr_v2_d2_download as d

class RangeTests(unittest.TestCase):
 def test_real_zip_central_directory_range_reader(self):
  f=io.BytesIO()
  with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('nuplan-v1.1/splits/trainval/x.db',b'SQLite format 3\0'+b'x'*10000)
  data=f.getvalue();calls=[]
  def fetch(key,start,n,size,etag):
   calls.append((start,n));self.assertGreaterEqual(start,0);self.assertLessEqual(start+n,size);return data[start:start+n]
  with patch.object(d,'fetch',fetch):
   remote=d.RemoteZip('official',len(data),'etag');z=zipfile.ZipFile(remote);items=z.infolist()
   self.assertEqual(len(items),1);self.assertEqual(items[0].file_size,10016)
   self.assertEqual(z.read(items[0]),b'SQLite format 3\0'+b'x'*10000)
  self.assertTrue(calls)
 def test_out_of_bounds_before_request(self):
  with self.assertRaises(ValueError):d.fetch('x',-1,20,50,'e')
  with self.assertRaises(ValueError):d.fetch('x',30,21,50,'e')
 def test_unexpected_whole_archive_metadata_refused(self):
  r=d.RemoteZip('x',1_000_000_000,'e')
  with self.assertRaisesRegex(RuntimeError,'20MB'):r.read(30_000_000)

if __name__=='__main__':unittest.main()
