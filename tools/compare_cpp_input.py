"""
Validation helper: compare the C++ network input against the Python parser.

Run the C++ binary once with LMNET_DUMP=1 to dump the assembled 6-channel input
for frame 1 (work_dir/cpp_input_f1.bin), then run this script (in the `inference`
container) to compare each channel against the Python LaserScan + precomputed
residual. Geometric channels match at corr > 0.99; the on-the-fly residual
channel is close (corr ~0.93) to the offline residual_images_1 npy.
"""

import os
import sys
import numpy as np
import yaml

root = 'mos_SalsaNext/train'
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'tasks/semantic'))

from common.laserscan import LaserScan

arch = yaml.safe_load(open('data/model_salsanext_residual_1/arch_cfg.yaml'))
sensor = arch['dataset']['sensor']
means = np.array(sensor['img_means'], np.float32)
stds = np.array(sensor['img_stds'], np.float32)
H = sensor['img_prop']['height']
W = sensor['img_prop']['width']
fu = sensor['fov_up']
fd = sensor['fov_down']

frame = '000001'
scan = LaserScan(project=True, H=H, W=W, fov_up=fu, fov_down=fd)
# For a single input scan the parser passes index_pose == current_pose, i.e. an
# identity transform, so the current scan xyz stay in the raw sensor frame.
scan.open_scan('data/sequences/08/velodyne/%s.bin' % frame,
               np.eye(4), np.eye(4), if_transform=False)

proj = np.stack([scan.proj_range,
                 scan.proj_xyz[..., 0], scan.proj_xyz[..., 1], scan.proj_xyz[..., 2],
                 scan.proj_remission], 0).astype(np.float32)
proj = (proj - means[:, None, None]) / stds[:, None, None]

residual = np.load('data/sequences/08/residual_images_1/%s.npy' % frame).astype(np.float32)
full = np.concatenate([proj, residual[None]], 0)
full = full * scan.proj_mask[None].astype(np.float32)   # (6,H,W)

cpp = np.fromfile('work_dir/cpp_input_f1.bin', dtype=np.float32).reshape(6, H, W)

names = ['range', 'x', 'y', 'z', 'remission', 'residual']
print('%-10s %10s %10s %12s' % ('channel', 'max|diff|', 'mean|diff|', 'corr'))
for c in range(6):
    a = full[c].ravel()
    b = cpp[c].ravel()
    d = np.abs(a - b)
    corr = np.corrcoef(a, b)[0, 1]
    print('%-10s %10.4f %10.5f %12.5f' % (names[c], d.max(), d.mean(), corr))

# residual-specific: how many pixels nonzero in each
pr = full[5]; cr = cpp[5]
print('\nresidual nonzero px  py:%d  cpp:%d' % (int((pr != 0).sum()), int((cr != 0).sum())))
print('residual sum        py:%.3f  cpp:%.3f' % (float(pr.sum()), float(cr.sum())))
