import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from Confocal_GUI.device import PulseSpinCore
except ImportError as e:
    sys.exit(1)

if __name__ == '__main__':
    pulse = PulseSpinCore()
    pulse.gui()