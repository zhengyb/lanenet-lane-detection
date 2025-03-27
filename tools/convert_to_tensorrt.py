import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorrt as trt

# TODO: support INT8

def convert_to_tensorrt(
    pb_path: str,
    trt_path: str,
    precision_mode: str = "FP16",
    batch_size: int = 1,
    workspace_size: int = 2 << 30  # 2GB
):
    """转换冻结模型为TensorRT优化模型"""
    # 加载原始计算图
    with tf.gfile.GFile(pb_path, "rb") as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())

    # 配置转换参数
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=[
            "LaneNet/binary_segmentation_result:0",
            "LaneNet/instance_segmentation_result:0"
        ],
        max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size,
        precision_mode=precision_mode,
        minimum_segment_size=5  # 最小子图节点数
    )

    # 保存优化后的模型
    with tf.gfile.GFile(trt_path, "wb") as f:
        f.write(trt_graph.SerializeToString())
    print(f"TensorRT模型已保存至: {trt_path}")

def build_engine(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    return builder.build_engine(network, config)

# Conversion parameters
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=1<<30,
    maximum_cached_engines=100
)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='./path/to/saved_model',
    conversion_params=conversion_params
)
converter.convert()
converter.save('./path/to/trt_model')

if __name__ == "__main__":
    convert_to_tensorrt(
        pb_path="./model/tusimple/bisenetv2_lanenet/lanenet_frozen_model.pb",
        trt_path="./model/tusimple/bisenetv2_lanenet/lanenet.trt",
        precision_mode="FP16"  # 可选 FP32/FP16/INT8
    ) 