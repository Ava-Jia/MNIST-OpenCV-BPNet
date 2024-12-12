import torch
import torchvision

# 创建一个简单的张量
x = torch.rand(5, 3)
print("测试张量：")
print(x)

# 检查 CUDA 是否可用
print("\nCUDA 是否可用:", torch.cuda.is_available())

# 如果 CUDA 可用，测试 GPU 张量
if torch.cuda.is_available():
    print("\nGPU 测试：")
    x = x.cuda()
    print(x)