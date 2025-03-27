import tensorflow as tf
import tf2onnx

def convert_to_onnx(pb_path, onnx_path):
    # 加载冻结模型
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 定义输入输出节点
    input_names = ["input_tensor:0"]
    output_names = [
        "LaneNet/binary_segmentation_result:0",
        "LaneNet/instance_segmentation_result:0"
    ]

    # 转换到ONNX
    with tf.Session() as sess:
        tf.import_graph_def(graph_def, name="")
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            sess.graph,
            input_names=input_names,
            output_names=output_names
        )
        model_proto = onnx_graph.make_model("lanenet")
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())

# 使用示例
convert_to_onnx(
    pb_path="./model/tusimple/bisenetv2_lanenet/lanenet_frozen_model.pb",
    onnx_path="./model/tusimple/bisenetv2_lanenet/lanenet.onnx"
) 