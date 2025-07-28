import pytest

def test_jax_gpu():
    """test if JAX can detect and use GPU"""
    
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    print(devices)
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    assert len(gpu_devices) > 0, "No GPU devices found"
    """
    [GpuDevice(id=0, platform='cuda', process_index=0, visible_device_list=['0'], memory_limit=None), GpuDevice(id=1, platform='cuda', process_index=0, visible_device_list=['1'], memory_limit=None)]
    """
    
    """
    x = jnp.array([1, 2, 3]) # test basic computation on GPU
    y = jnp.array([4, 5, 6]) 
    
    
    result = jax.device_put(x + y, gpu_devices[0]) # move computation to GPU and verify
    expected = jnp.array([5, 7, 9])
    
    assert jnp.array_equal(result, expected), "GPU computation failed"
    assert str(result.device()) == str(gpu_devices[0]), "Result not on GPU"
    """

if __name__ == "__main__":
    test_jax_gpu()