#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午8:50
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet.py
# @IDE: PyCharm
"""
Implement LaneNet Model
"""
import tensorflow as tf
import cv2
import numpy as np

from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet


class LaneNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg, name="LaneNet"):
        """

        """
        super(LaneNet, self).__init__()
        self._cfg = cfg
        self._net_flag = self._cfg.MODEL.FRONT_END
        self.name = name

        self._frontend = lanenet_front_end.LaneNetFrondEnd(
            phase=phase, net_flag=self._net_flag, cfg=self._cfg
        )
        self._backend = lanenet_back_end.LaneNetBackEnd(
            phase=phase, cfg=self._cfg
        )

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

            # 在返回前添加命名
            binary_output = tf.identity(binary_seg_prediction, name="binary_segmentation_result")
            instance_output = tf.identity(instance_seg_prediction, name="instance_segmentation_result")

        return binary_output, instance_output

    def compute_loss(self, input_tensor, binary_label, instance_label, name, reuse=False):
        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

        return calculated_losses


class LaneNet_frozen:
    def __init__(self, model_path, gpu_memory_fraction=0.8):
        """
        初始化冻结模型
        :param model_path: 冻结模型路径 (.pb文件)
        :param gpu_memory_fraction: GPU显存占用比例
        """
        self._model_path = model_path
        self._gpu_memory_fraction = gpu_memory_fraction
        
        # 模型输入输出配置
        self._input_size = (512, 256)  # (width, height)
        self._input_node_name = "input_tensor:0"
        self._output_node_names = [
            "LaneNet/binary_segmentation_result:0",
            "LaneNet/instance_segmentation_result:0"
        ]
        
        # 初始化计算图
        self._graph = tf.Graph()
        self._sess = None
        self._init_model()

    def _init_model(self):
        """加载冻结模型并准备会话"""
        with self._graph.as_default():
            # 加载计算图定义
            with tf.gfile.GFile(self._model_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

            # 获取输入输出张量
            self._input_tensor = self._graph.get_tensor_by_name(self._input_node_name)
            self._output_tensors = [
                self._graph.get_tensor_by_name(name) 
                for name in self._output_node_names
            ]
            
            # 配置会话参数
            config = tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    per_process_gpu_memory_fraction=self._gpu_memory_fraction
                )
            )
            self._sess = tf.Session(graph=self._graph, config=config)

    def inference(self, normalized_image):
        """
        执行推理
        :param input_tensor: 预处理后的输入张量
        :return: (binary_seg, instance_seg)
        """
        if self._sess is None:
            raise RuntimeError("模型未初始化")

        input_tensor = np.expand_dims(normalized_image, axis=0)
        return self._sess.run(
            self._output_tensors,
            feed_dict={self._input_tensor: input_tensor}
        )

    def postprocess(self, binary_seg, instance_seg):
        """
        后处理输出结果
        :param binary_seg: 二进制分割结果 (1, H, W, 1)
        :param instance_seg: 实例分割嵌入 (1, H, W, 4)
        :return: 处理后的结果字典
        """
        return {
            "binary_mask": np.squeeze(binary_seg > 0.5, axis=(0, -1)),
            "instance_embedding": np.squeeze(instance_seg, axis=0)
        }

    def close(self):
        """释放资源"""
        if self._sess is not None:
            self._sess.close()
            self._sess = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()