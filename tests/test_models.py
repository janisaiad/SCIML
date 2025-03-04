import pytest





def test_models():
    try:
        from model.deeponet.deeponet import DeepONet
    except ImportError:
        pytest.fail("DeepONet is not installed")
    
    
    
    