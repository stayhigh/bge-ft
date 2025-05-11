import torch
import os

def check_cuda_ready():
    retcode = 1 if torch.cuda.is_available() else 0      
    print("GPU is available" if retcode == 1 else "GPU is not available")  
    return retcode

def show_cuda_info():
    print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0))
    # 检查 PyTorch 构建时使用的架构
    print(torch.cuda.get_arch_list())
    # 查找已安装 GPU 中的一张卡的架构
    print(torch.cuda.get_device_properties(torch.device('cuda')))

if __name__ == "__main__":
    retcode=check_cuda_ready()
    if retcode == 1:
        show_cuda_info()