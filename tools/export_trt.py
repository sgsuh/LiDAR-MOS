"""
Create: 2022.12.21
Author: SG.SUH
Python: 3.8.5
PyTorch: 1.12.0
"""

import sys

sys.path.append('.')

import argparse
import tensorrt as trt
import onnx

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--onnx_path', type = str, default = 'work_dir/lmnet.onnx')
    parser.add_argument('--trt_path', type = str, default = 'work_dir/lmnet.engine')
    parser.add_argument('--fp16', type = bool, default = True)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    parser = trt.OnnxParser(network, logger)
    onnx_model = onnx.load(args.onnx_path)

    parser.parse(onnx_model.SerializeToString())

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    assert engine is not None, 'Failed to create TensorRT engine'

    with open(args.trt_path, mode = 'wb') as f:
        f.write(bytearray(engine.serialize()))


if __name__ == '__main__':
    main()