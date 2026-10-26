import io,json,sqlite3,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from tools import rbr_v2_d2_fast_download as fast

class MemberTests(unittest.TestCase):
 def test_segmented_sqlite_crc_sha_and_no_redownload(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'evidence';out.mkdir();local=root/'local';local.mkdir();(out/'official_LICENSE.txt').write_text('Fixture only')
   source=root/'source.db'
   with sqlite3.connect(source) as db:db.execute('CREATE TABLE fixture(x)');db.executemany('INSERT INTO fixture VALUES (?)',[(str(i)) for i in range(10)])
   raw=source.read_bytes();b=io.BytesIO()
   with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('trainval/fixture.db',raw)
   payload=b.getvalue()
   with zipfile.ZipFile(io.BytesIO(payload)) as z:info=z.infolist()[0]
   e=dict(filename='fixture.db',archive='official_fixture.zip',member=info.filename,archive_size=len(payload),etag='fixture',header_offset=info.header_offset,compression=info.compress_type,compressed_bytes=info.compress_size,uncompressed_bytes=info.file_size,crc32=info.CRC)
   calls=[]
   def fetch(key,start,n,size,etag):calls.append((start,n));return payload[start:start+n]
   with patch.object(fast,'OUT',out),patch.object(fast,'LOCAL',local),patch.object(fast,'SEGMENT',64),patch.object(fast,'fetch',fetch):
    r=fast.download_entry(e);self.assertTrue(r['crc_verified']);self.assertEqual((local/'fixture.db').read_bytes(),raw);n=len(calls);r2=fast.download_entry(e);self.assertEqual(n,len(calls));self.assertEqual(r2['sha256'],r['sha256'])

if __name__=='__main__':unittest.main()
