import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt

# 第一部分：数据准备
def prepare_data(batch_size=64):
    """
    准备MNIST数据集，创建数据加载器
    
    参数:
        batch_size: 每批处理的图像数量
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 定义数据转换：转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的标准化参数
    ])
    
    # 加载训练集
    print("正在下载并加载训练集...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',  
        train=True,     
        download=True,  
        transform=transform
    )
    
    # 加载测试集
    print("正在下载并加载测试集...")
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )
    
    print(f"数据加载完成！训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    return train_loader, test_loader

# 第二部分：模型定义
class ImprovedBPNet(nn.Module):
    def __init__(self):
        super(ImprovedBPNet, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),  # 将28x28的图像展平为784维向量
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.network(x)

# 第三部分：训练函数
def train_model(model, train_loader, test_loader, epochs=10):
    """
    训练模型并记录训练过程
    
    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
    返回:
        history: 包含训练损失和测试准确率的历史记录
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'test_accuracy': []
    }
    
    # 训练循环
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        start_time = time.time()
        
        # 进度条相关变量
        total_batches = len(train_loader)
        
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 打印训练进度
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Step [{i+1}/{total_batches}], '
                      f'Loss: {loss.item():.4f}')
        
        # 计算epoch平均损失
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
        
        # 在测试集上评估模型
        model.eval()  # 设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        history['test_accuracy'].append(accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Test Accuracy: {accuracy:.2f}%, '
              f'Time: {time.time() - start_time:.2f}s')
    
    return history

# 第四部分：主程序执行
if __name__ == "__main__":
    # 1. 准备数据
    print("正在准备数据...")
    train_loader, test_loader = prepare_data(batch_size=64)
    
    # 2. 创建模型
    print("正在创建模型...")
    model = ImprovedBPNet()
    
    # 3. 训练模型
    print("开始训练模型...")
    history = train_model(model, train_loader, test_loader, epochs=10)
    
    # 4. 保存模型
    print("保存模型...")
    torch.save(model.state_dict(), 'handwritten_digit_model.pth')
    print("模型训练完成并保存！")