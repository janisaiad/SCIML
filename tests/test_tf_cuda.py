import pytest
import tensorflow as tf


def test_tf_cuda():

    assert tf.test.is_gpu_available()