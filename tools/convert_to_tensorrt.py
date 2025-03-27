import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import os

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

if __name__ == "__main__":
    convert_to_tensorrt(
        pb_path="./model/tusimple/bisenetv2_lanenet/lanenet_frozen_model.pb",
        trt_path="./model/tusimple/bisenetv2_lanenet/lanenet.trt",
        precision_mode="FP16"  # 可选 FP32/FP16/INT8
    ) 