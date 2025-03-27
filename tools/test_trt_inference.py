import cv2
import numpy as np
import os
from tools.utils import make_instance_seg_img_visuable
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA设备

def list_trt_engine_layers(engine_path: str):
    """
    列出TensorRT引擎的所有层信息（兼容TensorRT 8.x+）
    """
    # 验证文件有效性
    if not Path(engine_path).exists():
        raise FileNotFoundError(f"引擎文件 {engine_path} 不存在")
    if Path(engine_path).stat().st_size < 1*1024*1024:  # 至少1MB
        raise ValueError("文件大小异常，可能不是有效的TensorRT引擎")

    # 初始化TensorRT组件
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    try:
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    except Exception as e:
        raise RuntimeError(f"引擎加载失败: {str(e)}") from e

    # 收集引擎信息
    engine_info = {
        "inputs": [],
        "outputs": [],
        "layers": []
    }

    # 获取绑定信息（使用新API）
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        tensor_info = {
            "name": tensor_name,
            "dtype": engine.get_tensor_dtype(tensor_name).name,
            "shape": tuple(engine.get_tensor_shape(tensor_name)),
            "is_input": tensor_mode == trt.TensorIOMode.INPUT
        }
        if tensor_info["is_input"]:
            engine_info["inputs"].append(tensor_info)
        else:
            engine_info["outputs"].append(tensor_info)

    # 获取层详细信息（TensorRT 8.x+兼容方式）
    inspector = engine.create_engine_inspector()
    for i in range(engine.num_layers):
        layer_info = inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE)
        engine_info["layers"].append(layer_info)

    # 显式释放检查器
    del inspector

    print("输入张量:")
    for inp in engine_info["inputs"]:
        print(f"  {inp['name']} | {inp['dtype']} | {inp['shape']}")
    
    print("\n输出张量:")
    for out in engine_info["outputs"]:
        print(f"  {out['name']} | {out['dtype']} | {out['shape']}")

    return engine_info

class TRTLaneNet:
    def __init__(self, trt_path):
        # 验证文件存在性
        if not Path(trt_path).exists():
            raise FileNotFoundError(f"模型文件 {trt_path} 不存在")
            
        # 验证文件大小
        file_size = Path(trt_path).stat().st_size
        if file_size < 1*1024*1024:  # 至少1MB
            raise ValueError(f"模型文件异常，大小仅 {file_size/1024/1024:.2f}MB")

        # Initialize TensorRT components
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(trt_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Setup bindings
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            # Allocate memory
            size = trt.volume(shape) * dtype.itemsize
            allocation = cuda.mem_alloc(size)
            self.bindings.append(int(allocation))
            
            # Store input/output info
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'allocation': allocation,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': name,
                    'allocation': allocation,
                    'shape': shape,
                    'dtype': dtype
                })
        
        # Set output names based on engine
        self.binary_output_name = "LaneNet/binary_segmentation_result:0"
        self.instance_output_name = "LaneNet/instance_segmentation_result:0"
        self.stream = cuda.Stream()

        # 添加TensorRT到numpy的类型映射
        self.trt_to_np = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.INT32: np.int32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8
        }

    def preprocess(self, image):
        """预处理输入图像"""
        resized = cv2.resize(image, (512, 256))
        return (resized.astype(np.float32) / 127.5) - 1.0  # 归一化到[-1, 1]

    def inference(self, image):
        """执行推理"""
        input_data = self.preprocess(image)
        input_data = np.expand_dims(input_data, axis=0)   # 添加batch维度
        
        # 将数据拷贝到GPU
        cuda.memcpy_htod_async(self.inputs[0]['allocation'],
                              input_data.ravel(), self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 分配输出内存（修复类型转换）
        outputs = {}
        for output in self.outputs:
            np_dtype = self.trt_to_np[output['dtype']]
            outputs[output['name']] = np.empty(output['shape'], dtype=np_dtype)
            cuda.memcpy_dtoh_async(outputs[output['name']],
                                 output['allocation'],
                                 self.stream)
        
        # 同步流
        self.stream.synchronize()
        
        # 获取输出结果
        binary_mask = outputs[self.binary_output_name]
        instance_embedding = outputs[self.instance_output_name]
        
        # 调整输出形状 (batch维度在第一个位置)
        return binary_mask[0], instance_embedding[0]



# 使用示例
def test_trt_inference():
    detector = TRTLaneNet("./model/tusimple/bisenetv2_lanenet/lanenet.engine")
    image = cv2.imread("./data/tusimple_test_image/0.jpg")
    binary_mask, instance_embedding = detector.inference(image)
    
    # 处理INT32类型的二进制掩码输出
    binary_seg_result = (binary_mask * 255).astype(np.uint8)  # 如果输出是0/1的二值结果
    # 或者根据实际输出范围调整：
    # binary_seg_result = np.clip(binary_mask, 0, 255).astype(np.uint8)
    
    instance_seg_result = make_instance_seg_img_visuable(instance_embedding)
    
    os.makedirs("./output", exist_ok=True)
    cv2.imwrite("./output/binary_mask.jpg", binary_seg_result)
    cv2.imwrite("./output/instance_embedding.jpg", instance_seg_result)    
    print("Done")


if __name__ == "__main__":
    list_trt_engine_layers("./model/tusimple/bisenetv2_lanenet/lanenet.engine")
    test_trt_inference()

    
