"""Check packages, fonts and GPU computation without running CSI training."""

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from matplotlib import font_manager


def main():
    project_dir = Path(__file__).resolve().parent.parent
    ast.parse((project_dir / 'cha_gpt.py').read_text(encoding='utf-8-sig'))
    print('Python:', sys.version.split()[0])
    for name, package in [
        ('TensorFlow', tf), ('NumPy', np), ('pandas', pd),
        ('matplotlib', matplotlib), ('scikit-learn', sklearn),
    ]:
        print(f'{name}: {package.__version__}')

    print('Font:', font_manager.findfont('NanumGothic', fallback_to_default=False))
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError('TensorFlow cannot access a GPU.')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU:', tf.config.experimental.get_device_details(gpu))

    tf.config.set_soft_device_placement(False)
    with tf.device('/GPU:0'):
        matrix = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        convolution = tf.nn.conv1d(tf.ones((1, 8, 1)), tf.ones((3, 1, 1)), 1, 'VALID')
        np.testing.assert_allclose(matrix.numpy(), 2.0)
        np.testing.assert_allclose(convolution.numpy(), 3.0)
        if 'GPU:0' not in matrix.device or 'GPU:0' not in convolution.device:
            raise RuntimeError('GPU operations were placed on another device.')
        print('GPU matrix multiplication and convolution: OK')

        @tf.function(jit_compile=True)
        def compiled_operation(value):
            return tf.math.sin(value) + 1.0

        compiled_result = compiled_operation(tf.zeros((2, 2)))
        np.testing.assert_allclose(compiled_result.numpy(), 1.0)
        print('GPU XLA compilation: OK')

    print('Environment check passed. CSI training was not started.')


if __name__ == '__main__':
    main()
