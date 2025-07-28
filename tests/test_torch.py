import pytest

def test_torch_cuda():
    """Test if PyTorch can use CUDA GPU acceleration if available"""
    import torch
    is_cuda_available = torch.cuda.is_available()  # check cuda availability
    
    if is_cuda_available:
        device_count = torch.cuda.device_count()
        
        x = torch.randn(100, 100)
        
        x_cuda = x.cuda()
        
        assert x_cuda.is_cuda
        assert x_cuda.device.type == 'cuda'  # verify tensor is on correct device
        
        y_cuda = x_cuda * 2
        assert y_cuda.is_cuda
        
        y_cpu = y_cuda.cpu()
        assert torch.allclose(y_cpu, x.cpu() * 2)  # validate computation accuracy
    print("CUDA is available")
    print(f"Number of CUDA devices: {device_count}")
    print("CUDA test passed")
    print("CUDA test passed")
if __name__ == "__main__":
    test_torch_cuda()