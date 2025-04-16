#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午3:48
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_discriminative_loss.py
# @IDE: PyCharm Community Edition
"""
Discriminative Loss for instance segmentation
"""
import tensorflow as tf

def lane_net_loss(y_true, y_pred):
    """
    y_true: кортеж (binary_mask, instance_mask)
    y_pred: кортеж (binary_seg_logits, instance_seg_logits)
    """
    binary_true, instance_true = y_true
    binary_pred, instance_pred = y_pred

    # 1) Binary crossentropy loss для binary_segmentation
    bce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(binary_true, binary_pred)

    # 2) Discriminative loss для instance сегментації
    # Ти можеш використати params як у оригіналі LaneNet
    disc_loss, l_var, l_dist, l_reg = discriminative_loss(
        instance_pred,
        instance_true,
        feature_dim=instance_pred.shape[-1],
        image_shape=tf.shape(instance_pred)[1:3],
        delta_v=0.5,
        delta_d=1.5,
        param_var=1.0,
        param_dist=1.0,
        param_reg=0.001
    )

    # 3) Сумарний лосс (підлаштуй ваги під задачу)
    total_loss = bce_loss + disc_loss

    return total_loss