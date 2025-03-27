import tensorflow as tf

def inspect_frozen_model(pb_path):
    # 加载计算图
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 打印模型信息
    print(f"模型版本: {graph_def.version}")
    print("\n输入节点:")
    for node in graph_def.node:
        if node.op == "Placeholder":
            print(f"名称: {node.name}")
            print(f"形状: {node.attr['shape'].shape}")
            print(f"数据类型: {node.attr['dtype'].type}")

    print("\n输出节点:")
    # 更精确的节点筛选条件
    output_nodes = [
        n for n in graph_def.node 
        if "binary_segmentation_result" in n.name 
        or "instance_segmentation_result" in n.name
    ]
    
    for node in output_nodes:
        print(f"名称: {node.name}")
        print(f"操作类型: {node.op}")
        print(f"输入依赖: {node.input}")

# 使用示例
inspect_frozen_model("./model/tusimple/bisenetv2_lanenet/lanenet_frozen_model.pb") 