import onnxruntime
import numpy as np
import cv2
import os
from tools.utils import make_instance_seg_img_visuable

def test_onnx_model(onnx_path, image_path):
    # 初始化ONNX Runtime会话
    sess = onnxruntime.InferenceSession(onnx_path)
    
    # 打印输入输出信息
    for input in sess.get_inputs():
        print(f"Input node: {input.name}, Shape: {input.shape}, Type: {input.type}")
    for output in sess.get_outputs():
        print(f"Output node: {output.name}, Shape: {output.shape}, Type: {output.type}")

    # 修正输入预处理
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 添加颜色空间转换
    image = (image / 127.5) - 1.0  # 添加标准化（假设原模型使用此预处理）
    image_input = np.expand_dims(image, axis=0).astype(np.float32)
    
    # 执行推理
    outputs = sess.run(
        output_names=[
            "LaneNet/binary_segmentation_result:0",
            "LaneNet/instance_segmentation_result:0"
        ],
        input_feed={"input_tensor:0": image_input}
    )
    
    # 输出结果信息
    binary_mask = outputs[0]
    instance_embedding = outputs[1]
    binary_seg_result = np.array(binary_mask[0] * 255, dtype=np.uint8)
    instance_seg_result = make_instance_seg_img_visuable(instance_embedding[0])
    # save the result
    os.makedirs("./output", exist_ok=True)
    try:
        os.remove("./output/binary_mask.jpg")
        os.remove("./output/instance_embedding.jpg")
    except FileNotFoundError:
        pass
        
    cv2.imwrite("./output/binary_mask.jpg", binary_seg_result)
    cv2.imwrite("./output/instance_embedding.jpg", instance_seg_result)    
    print("Done")

if __name__ == "__main__":
    test_onnx_model("./model/tusimple/bisenetv2_lanenet/lanenet.onnx", "./data/tusimple_test_image/0.jpg") 