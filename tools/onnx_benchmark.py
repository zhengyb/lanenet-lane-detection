import time
import numpy as np
import onnxruntime as ort

def benchmark_onnx_model(onnx_path, input_shape=(1, 256, 512, 3), warmup=100, runs=1000):
    """
    ONNX模型基准测试工具
    :param onnx_path: ONNX模型路径
    :param input_shape: 输入张量形状 (默认符合实际NHWC格式)
    :param warmup: 预热次数
    :param runs: 正式测试次数
    """
    # 高级配置选项
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 显式指定CUDA提供器配置
    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True
    }
    
    sess = ort.InferenceSession(
        onnx_path,
        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider'],
        sess_options=sess_options
    )
    
    # 生成与测试脚本一致的输入数据
    # 生成符合实际分布的随机图像数据 (uint8范围)
    dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    # 应用与测试脚本相同的预处理
    dummy_input = (dummy_input.astype(np.float32) / 127.5) - 1.0  # 标准化到[-1, 1]范围

    dummy_input = ort.OrtValue.ortvalue_from_numpy(dummy_input, 'cuda', 0)
    
    # 绑定IO
    io_binding = sess.io_binding()
    io_binding.bind_ortvalue_input(sess.get_inputs()[0].name, dummy_input)
    for output in sess.get_outputs():
        io_binding.bind_output(output.name, 'cuda')
    
    # Warmup
    for _ in range(warmup):
        sess.run_with_iobinding(io_binding)
    
    # 正式测试
    start = time.time()
    for _ in range(runs):
        sess.run_with_iobinding(io_binding)
    elapsed = time.time() - start
    
    print(f"优化后平均推理时间: {elapsed/runs*1000:.2f} ms")
    print(f"FPS: {runs/elapsed:.2f}")
if __name__ == "__main__":
    benchmark_onnx_model("./model/tusimple/bisenetv2_lanenet/lanenet.onnx") 
    # FPS: 138