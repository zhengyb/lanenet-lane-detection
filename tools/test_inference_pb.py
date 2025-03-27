import cv2
import numpy as np
import tensorflow as tf
from utils import make_instance_seg_img_visuable


def test_inference(pb_path, test_image_path):
    # 加载测试图像
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, (512, 256))  # 匹配输入尺寸
    input_data = img.astype(np.float32) / 127.5 - 1.0  # 假设模型使用[-1,1]归一化
    input_data = np.expand_dims(input_data, axis=0)

    # 加载计算图
    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        # 获取输入输出张量
        input_tensor = graph.get_tensor_by_name("input_tensor:0")
        binary_output = graph.get_tensor_by_name("LaneNet/binary_segmentation_result:0")
        instance_output = graph.get_tensor_by_name("LaneNet/instance_segmentation_result:0")

        with tf.Session() as sess:
            # 运行推理
            binary_mask, instance_embedding = sess.run(
                [binary_output, instance_output],
                feed_dict={input_tensor: input_data}
            )

    # 检查输出形状
    print(f"二进制分割结果形状: {binary_mask.shape} (应为 [1, 256, 512, 1])")
    print(f"实例嵌入结果形状: {instance_embedding.shape} (应为 [1, 256, 512, 4])")

    binary_seg_result = np.array(binary_mask[0] * 255, dtype=np.uint8)
    instance_seg_result = make_instance_seg_img_visuable(instance_embedding[0])
    # save the result
    cv2.imwrite("./output/binary_mask.jpg", binary_seg_result)
    cv2.imwrite("./output/instance_embedding.jpg", instance_seg_result)


if __name__ == "__main__":
    model_path = "./model/tusimple/bisenetv2_lanenet/lanenet_frozen_model.pb"
    image_path = "./data/tusimple_test_image/0.jpg"
    test_inference(model_path, image_path)