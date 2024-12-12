import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义数据转换方式
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的标准化参数
])

# 加载训练集
train_dataset = torchvision.datasets.MNIST(
    root='./data',  # 数据存储路径
    train=True,     # 指定为训练集
    download=True,  # 如果数据不存在则下载
    transform=transform
)

# 加载测试集
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64,    # 每批处理的图像数量
    shuffle=True      # 随机打乱数据
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)

# 可视化一些样本图像
def visualize_samples(dataloader):
    # 获取一批数据
    images, labels = next(iter(dataloader))
    
    # 显示前9张图片
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.show()