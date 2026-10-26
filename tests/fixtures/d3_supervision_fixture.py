"""Non-scientific delayed helper fixture; never imports nuPlan or a model."""
import subprocess,sys,time
if sys.argv[1]=='helper':time.sleep(.25)
else:
 subprocess.Popen([sys.executable,__file__,'helper'])
 time.sleep(.08)
 print('FIXTURE_PARENT_COMPLETE_NO_MODEL',flush=True)
