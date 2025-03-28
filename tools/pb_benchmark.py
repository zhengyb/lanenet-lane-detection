import time
import tensorflow as tf
import numpy as np

def benchmark_model(pb_path, warmup=10, runs=100):
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        input_tensor = graph.get_tensor_by_name("input_tensor:0")
        outputs = [
            graph.get_tensor_by_name("LaneNet/binary_segmentation_result:0"),
            graph.get_tensor_by_name("LaneNet/instance_segmentation_result:0")
        ]

        # 生成随机输入, 归一化到[-1, 1]
        dummy_input = (np.random.randn(1, 256, 512, 3).astype(np.uint8) / 127.5) - 1.0

        with tf.Session() as sess:
            # Warmup
            for _ in range(warmup):
                sess.run(outputs, {input_tensor: dummy_input})

            # 正式测试
            start = time.time()
            for _ in range(runs):
                sess.run(outputs, {input_tensor: dummy_input})
            elapsed = time.time() - start

    print(f"平均推理时间: {elapsed/runs*1000:.2f} ms")
    print(f"FPS: {runs/elapsed:.2f}")

# 使用示例
benchmark_model("./model/tusimple/bisenetv2_lanenet/lanenet_frozen_model.pb") 
# FPS: 180