import tensorflow as tf
import cv2
import numpy as np
import os
from tools.utils import make_instance_seg_img_visuable


class TRTLaneNet:
    def __init__(self, trt_path):
        # 初始化计算图
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph, config=self._get_config())  # 使用compat.v1 API
        
        with self.graph.as_default():  # 确保在计算图上下文中操作
            # 加载TRT计算图
            with tf.io.gfile.GFile(trt_path, "rb") as f:
                trt_graph = tf.compat.v1.GraphDef()  # 使用兼容模式GraphDef
                trt_graph.ParseFromString(f.read())
                tf.import_graph_def(trt_graph, name="")
            
            # 获取输入输出张量（必须在计算图上下文中）
            self.input_tensor = self.graph.get_tensor_by_name("input_tensor:0")
            self.binary_output = self.graph.get_tensor_by_name("LaneNet/binary_segmentation_result:0")
            self.instance_output = self.graph.get_tensor_by_name("LaneNet/instance_segmentation_result:0")

    def _get_config(self):
        """配置GPU参数"""
        return tf.compat.v1.ConfigProto(  # 使用兼容模式ConfigProto
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction=0.8
            )
        )

    def preprocess(self, image):
        """预处理输入图像"""
        resized = cv2.resize(image, (512, 256))
        return (resized.astype(np.float32) / 127.5) - 1.0  # 归一化到[-1, 1]

    def inference(self, image):
        """执行推理"""
        input_data = np.expand_dims(self.preprocess(image), axis=0)
        return self.sess.run(
            [self.binary_output, self.instance_output],
            feed_dict={self.input_tensor: input_data}
        )

# 使用示例
if __name__ == "__main__":
    detector = TRTLaneNet("./model/tusimple/bisenetv2_lanenet/lanenet.trt")
    image = cv2.imread("./data/tusimple_test_image/0.jpg")
    binary_mask, instance_embedding = detector.inference(image) 
    binary_seg_result = np.array(binary_mask[0] * 255, dtype=np.uint8)
    instance_seg_result = make_instance_seg_img_visuable(instance_embedding[0])
    # save the result
    os.makedirs("./output", exist_ok=True)
    os.remove("./output/binary_mask.jpg")
    os.remove("./output/instance_embedding.jpg")
    cv2.imwrite("./output/binary_mask.jpg", binary_seg_result)
    cv2.imwrite("./output/instance_embedding.jpg", instance_seg_result)    
    print("Done")