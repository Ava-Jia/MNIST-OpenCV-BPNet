import cv2
import torch
import numpy as np
from torchvision import transforms
from BPNetModel import ImprovedBPNet  # 导入我们之前定义的模型

class DigitRecognitionSystem:
    def __init__(self, model_path='handwritten_digit_model.pth'):
        """
        初始化数字识别系统
        model_path: 训练好的模型权重文件路径
        """
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ImprovedBPNet().to(self.device)
        
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 设置为评估模式
        
        # 定义图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def preprocess_frame(self, frame):
        """
        预处理摄像头捕获的画面
        frame: 原始摄像头画面
        """
        # 定义感兴趣区域(ROI)的大小
        roi_size = 280
        height, width = frame.shape[:2]
        
        # 计算中心区域的坐标
        x = (width - roi_size) // 2
        y = (height - roi_size) // 2
        
        # 提取ROI
        roi = frame[y:y+roi_size, x:x+roi_size]
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 应用自适应阈值处理
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 调整大小为28x28（MNIST格式）
        resized = cv2.resize(binary, (28, 28))
        
        # 转换为模型输入格式
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        
        return tensor, roi, binary

    def draw_results(self, frame, prediction, confidence, roi_binary):
        """
        在画面上绘制识别结果和辅助信息
        """
        height, width = frame.shape[:2]
        roi_size = 280
        
        # 计算ROI位置
        x = (width - roi_size) // 2
        y = (height - roi_size) // 2
        
        # 绘制ROI边框
        cv2.rectangle(frame, (x, y), (x+roi_size, y+roi_size), (0, 255, 0), 2)
        
        # 显示预处理后的图像
        roi_binary_display = cv2.resize(roi_binary, (140, 140))
        frame[20:160, 20:160] = cv2.cvtColor(roi_binary_display, cv2.COLOR_GRAY2BGR)
        
        # 显示识别结果
        result_text = f"Prediction: {prediction}"
        conf_text = f"Confidence: {confidence:.2f}%"
        
        cv2.putText(frame, result_text, (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, conf_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame

    def run(self):
        """
        运行实时识别系统
        """
        print("正在启动实时识别系统...")
        print("按 'Q' 键退出程序")
        
        while True:
            # 捕获摄像头画面
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
            
            # 预处理画面
            tensor, roi, binary = self.preprocess_frame(frame)
            
            # 执行预测
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                confidence = confidence.item() * 100
                prediction = prediction.item()
            
            # 绘制结果
            frame = self.draw_results(frame, prediction, confidence, binary)
            
            # 显示画面
            cv2.imshow('Digit Recognition', frame)
            
            # 检查是否按下'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """
        确保在对象销毁时释放摄像头资源
        """
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    # 创建并运行识别系统
    system = DigitRecognitionSystem()
    system.run()