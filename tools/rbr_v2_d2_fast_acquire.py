#!/usr/bin/env python3
"""Same frozen source/SDC loop with improved transport; source criteria unchanged."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.rbr_v2_d2_acquire as acquisition
from tools.rbr_v2_d2_fast_download import download_entry

if __name__=='__main__':
 acquisition.download_entry=download_entry
 acquisition.run()
