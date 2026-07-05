"""
Export the SalsaNext LiDAR-MOS model to ONNX.

Create: 2022.12.20 (SG.SUH)
Ported to the Dockerized LiDAR-MOS repo (PyTorch 2.x, opset 11).

The model takes a single (1, 6, 64, 2048) range-image tensor
(channels: range, x, y, z, remission, residual) and outputs per-class
softmax scores of shape (1, 3, 64, 2048).
"""

import os
import sys
import argparse

sys.path.append('.')
# Make the SalsaNext package importable regardless of the current working dir.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import torch.nn as nn
import onnx

from mos_SalsaNext.train.tasks.semantic.modules.SalsaNext import SalsaNext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='data/model_salsanext_residual_1')
    parser.add_argument('--onnx_path', type=str, default='work_dir/lmnet.onnx')
    parser.add_argument('--opset', type=int, default=11)
    # ONNX export runs fine on CPU and avoids any GPU/driver coupling in the
    # exported graph; pass --device cuda to export on GPU instead.
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    args = parser.parse_args()

    arch = yaml.safe_load(open(args.model + '/arch_cfg.yaml', 'r'))
    data = yaml.safe_load(open(args.model + '/data_cfg.yaml', 'r'))

    # n_classes is the size of the inverse learning map (MOS: 0/static/moving).
    n_classes = len(data['learning_map_inv'])

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')

    with torch.no_grad():
        model = SalsaNext(n_classes, arch)
        model = nn.DataParallel(model)
        w_dict = torch.load(args.model + '/SalsaNext_valid_best',
                            map_location=lambda storage, loc: storage)
        model.load_state_dict(w_dict['state_dict'], strict=True)

    model = model.module.to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.onnx_path) or '.', exist_ok=True)

    # (batch, channels=range/x/y/z/remission/residual, H=64, W=2048)
    proj_in = torch.randn(1, 6, 64, 2048, device=device)

    torch.onnx.export(
        model, proj_in, args.onnx_path,
        input_names=['input_0'], output_names=['output_0'],
        opset_version=args.opset, verbose=False,
    )

    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)

    print('Exported ONNX to %s (n_classes=%d, opset=%d, device=%s)'
          % (args.onnx_path, n_classes, args.opset, device))
