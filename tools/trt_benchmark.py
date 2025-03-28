import time
import numpy as np
from tools.test_trt_inference import TRTLaneNet

def benchmark_trt_model(trt_path, input_shape=(256, 512, 3), warmup=10, runs=100):
    """
    TensorRT模型基准测试工具
    :param trt_path: TensorRT引擎路径
    :param input_shape: 输入图像形状 (H, W, C)
    :param warmup: 预热次数
    :param runs: 正式测试次数
    """
    # 初始化TensorRT模型
    detector = TRTLaneNet(trt_path)
    
    # 生成与测试脚本一致的输入数据
    # 生成符合实际分布的随机图像数据 (uint8范围)
    dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    # 应用与测试脚本相同的预处理
    dummy_image = (dummy_input.astype(np.float32) / 127.5) - 1.0  # 标准化到[-1, 1]范围


    
    # Warmup运行
    for _ in range(warmup):
        detector.inference(dummy_image)
    
    # 正式测试
    start = time.time()
    for _ in range(runs):
        detector.inference(dummy_image)
    elapsed = time.time() - start
    
    # 打印结果
    print(f"TensorRT引擎基准测试结果 ({trt_path}):")
    print(f"平均推理时间: {elapsed/runs*1000:.2f} ms")
    print(f"FPS: {runs/elapsed:.2f}\n")

# TensorRT特有的优化技术包括：
# - 层融合 (Layer Fusion)
# - 精度校准 (FP16/INT8)
# - 内核自动调优 (Auto-Tuning)
# - 动态张量内存管理

if __name__ == "__main__":
    benchmark_trt_model("./model/tusimple/bisenetv2_lanenet/lanenet.engine") 
    # FPS: 265