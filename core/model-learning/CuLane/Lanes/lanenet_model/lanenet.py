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

from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet
from tensorflow.keras.applications import VGG16


class LaneNet(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._embedding_dims = cfg.MODEL.EMBEDDING_FEATS_DIMS

        self.frontend = VGG16(weights='imagenet', include_top=False)
        self.conv_decode = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.binary_seg_logits = tf.keras.layers.Conv2D(2, (1, 1), name='binary_seg_logits')
        self.instance_seg_logits = tf.keras.layers.Conv2D(self._embedding_dims, (1, 1), name='instance_seg_logits')

        # Backend: embedding refinement and loss computation
        self.backend = lanenet_back_end.LaneNetBackEnd(embedding_dims=self._embedding_dims)


    def call(self, inputs, training=False):
        features = self.frontend(inputs, training=training)
        x = self.conv_decode(features)
        binary_seg_out = self.binary_seg_logits(x)
        instance_seg_out = self.instance_seg_logits(x)

        # Backend processes instance segmentation further (refinement)
        instance_seg_refined = self.backend(instance_seg_out, training=training)

        return binary_seg_out, instance_seg_refined
