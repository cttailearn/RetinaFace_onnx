#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX推理模块

提供基于ONNX的RetinaFace人脸检测推理功能
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from typing import List, Dict, Tuple, Optional

class ONNXInference:
    """ONNX推理器类"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化ONNX推理器
        
        Args:
            model_path: ONNX模型文件路径
            device: 推理设备 ('cpu' 或 'cuda')
        """
        self.model_path = model_path
        self.device = device
        
        # 硬编码的配置参数（之前从配置文件中读取）
        self.input_size = (640, 640)
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.8  # NMS阈值，设置为0.8以检测多个人脸
        self.top_k = 5000
        self.keep_top_k = 750
        self.vis_thres = 0.3  # 进一步降低阈值以检测更多人脸
        
        # 先验框配置
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.variance = [0.1, 0.2]
        
        # 初始化ONNX会话
        self._init_session()
        
        # 生成先验框
        self.priors = self._generate_priors()
        
        print(f"ONNX推理器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"推理设备: {device}")
        print(f"输入尺寸: {self.input_size}")
    
    def _init_session(self):
        """初始化ONNX推理会话"""
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"输入节点: {self.input_name}")
        print(f"输出节点: {self.output_names}")
    
    def _generate_priors(self) -> np.ndarray:
        """生成先验框"""
        feature_maps = []
        for step in self.steps:
            feature_maps.append([self.input_size[0] // step, self.input_size[1] // step])
        
        anchors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in np.ndindex(f[0], f[1]):
                # 每个特征层位置生成2个anchor（对应配置中的anchor_num=2）
                for min_size in min_sizes:
                    s_kx = min_size / self.input_size[1]
                    s_ky = min_size / self.input_size[0]
                    # 计算anchor中心点坐标（归一化）
                    cx = (j + 0.5) * self.steps[k] / self.input_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.input_size[0]
                    anchors += [cx, cy, s_kx, s_ky]
        
        output = np.array(anchors).reshape(-1, 4)
        return output
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """预处理图像"""
        img_height, img_width = image.shape[:2]
        
        # 计算缩放比例
        scale = min(self.input_size[0] / img_height, self.input_size[1] / img_width)
        
        # 缩放图像
        new_height = int(img_height * scale)
        new_width = int(img_width * scale)
        resized_img = cv2.resize(image, (new_width, new_height))
        
        # 创建填充后的图像
        padded_img = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        padded_img[:new_height, :new_width] = resized_img
        
        # 转换为模型输入格式
        input_img = padded_img.astype(np.float32)
        input_img -= np.array([104, 117, 123])  # 减去均值
        input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0)  # 添加batch维度
        
        return input_img, scale, (img_width, img_height)
    
    def _decode_predictions(self, predictions: List[np.ndarray], scale: float, 
                          original_size: Tuple[int, int]) -> List[Dict]:
        """解码预测结果"""
        loc, conf, landms = predictions
        
        # 解码边界框
        boxes = self._decode_boxes(loc[0], self.priors, self.variance)
        
        # 解码置信度 - 应用softmax将logits转换为概率
        conf_softmax = np.exp(conf[0]) / np.sum(np.exp(conf[0]), axis=1, keepdims=True)
        scores = conf_softmax[:, 1]  # 取正类概率
        
        # 解码关键点
        landmarks = self._decode_landmarks(landms[0], self.priors, self.variance)
        
        # 过滤低置信度检测
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]
        
        # 按置信度排序
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]
        
        # NMS
        keep = self._nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep, :]
        landmarks = landmarks[keep, :]
        scores = scores[keep]
        
        # 保留top-k
        boxes = boxes[:self.keep_top_k, :]
        landmarks = landmarks[:self.keep_top_k, :]
        scores = scores[:self.keep_top_k]
        
        # 转换回原始图像坐标并计算正脸置信度
        detections = []
        for i in range(len(boxes)):
            if scores[i] < self.vis_thres:
                continue
            
            # 先转换到输入图像尺寸(640x640)，然后转换到原始图像坐标
            # boxes是归一化坐标[0,1]，需要乘以输入尺寸得到像素坐标
            box_pixels = boxes[i] * np.array([self.input_size[1], self.input_size[0], 
                                             self.input_size[1], self.input_size[0]])
            landmark_pixels = landmarks[i] * np.array([self.input_size[1], self.input_size[0]] * 5)
            
            # 转换到原始图像坐标（考虑缩放）
            box = box_pixels / scale
            landmark = landmark_pixels / scale
            
            # 计算正脸置信度
            frontal_confidence = self._calculate_frontal_confidence(landmark, box)
            
            # 融合检测置信度和正脸置信度
            # 使用加权几何平均，检测置信度权重0.7，正脸置信度权重0.3
            combined_confidence = (scores[i] ** 0.7) * (frontal_confidence ** 0.3)
            
            # 判断是否为正脸（5个关键点且正脸置信度>0.5）
            is_frontal = len(landmark) == 10 and frontal_confidence > 0.5
            
            detection = {
                'bbox': box.astype(int).tolist(),
                'confidence': float(scores[i]),  # 原始检测置信度
                'frontal_confidence': float(frontal_confidence),  # 正脸置信度
                'combined_confidence': float(combined_confidence),  # 融合置信度
                'is_frontal': is_frontal,  # 是否为正脸
                'landmarks': landmark.reshape(-1, 2).astype(int).tolist(),
                'landmark_count': len(landmark) // 2  # 关键点数量
            }
            detections.append(detection)
        
        return detections
    
    def _decode_boxes(self, loc: np.ndarray, priors: np.ndarray, variances: List[float]) -> np.ndarray:
        """解码边界框"""
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def _decode_landmarks(self, pre: np.ndarray, priors: np.ndarray, variances: List[float]) -> np.ndarray:
        """解码关键点"""
        landmarks = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                  priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                  priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                  priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                  priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]), axis=1)
        return landmarks
    
    def _calculate_frontal_confidence(self, landmarks: np.ndarray, bbox: np.ndarray) -> float:
        """计算正脸置信度
        
        Args:
            landmarks: 关键点坐标 [left_eye_x, left_eye_y, right_eye_x, right_eye_y, 
                                 nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y]
            bbox: 边界框 [x1, y1, x2, y2]
            
        Returns:
            正脸置信度 (0-1)
        """
        if len(landmarks) != 10:  # 5个关键点，每个2个坐标
            return 0.0
        
        # 重塑为5个点的坐标
        points = landmarks.reshape(5, 2)
        left_eye, right_eye, nose, left_mouth, right_mouth = points
        
        # 计算人脸中心和尺寸
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        confidence_factors = []
        
        # 1. 眼睛对称性检查
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # 眼睛中心应该接近人脸中心的水平位置
        eye_horizontal_deviation = abs(eye_center_x - face_center_x) / face_width
        eye_symmetry_score = max(0, 1 - eye_horizontal_deviation * 4)  # 偏差超过25%时置信度为0
        confidence_factors.append(eye_symmetry_score)
        
        # 2. 鼻子位置检查
        nose_horizontal_deviation = abs(nose[0] - face_center_x) / face_width
        nose_position_score = max(0, 1 - nose_horizontal_deviation * 6)  # 偏差超过16.7%时置信度为0
        confidence_factors.append(nose_position_score)
        
        # 3. 嘴部对称性检查
        mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
        mouth_horizontal_deviation = abs(mouth_center_x - face_center_x) / face_width
        mouth_symmetry_score = max(0, 1 - mouth_horizontal_deviation * 4)
        confidence_factors.append(mouth_symmetry_score)
        
        # 4. 五官垂直排列检查
        # 眼睛应该在鼻子上方，鼻子应该在嘴部上方
        vertical_order_score = 1.0
        if eye_center_y >= nose[1] or nose[1] >= (left_mouth[1] + right_mouth[1]) / 2:
            vertical_order_score = 0.3  # 垂直顺序错误，但不完全排除
        confidence_factors.append(vertical_order_score)
        
        # 5. 眼睛间距合理性检查
        eye_distance = abs(right_eye[0] - left_eye[0])
        expected_eye_distance = face_width * 0.3  # 眼距约为人脸宽度的30%
        eye_distance_ratio = min(eye_distance, expected_eye_distance) / max(eye_distance, expected_eye_distance)
        eye_distance_score = eye_distance_ratio
        confidence_factors.append(eye_distance_score)
        
        # 6. 关键点在边界框内的检查
        points_in_bbox = 0
        for point in points:
            if bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]:
                points_in_bbox += 1
        bbox_containment_score = points_in_bbox / 5.0
        confidence_factors.append(bbox_containment_score)
        
        # 计算综合置信度（加权平均）
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]  # 眼睛对称性权重最高
        frontal_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return min(1.0, max(0.0, frontal_confidence))
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """非极大值抑制"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def inference(self, image: np.ndarray) -> List[Dict]:
        """执行推理"""
        # 预处理
        input_data, scale, original_size = self._preprocess_image(image)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        inference_time = time.time() - start_time
        
        # 后处理
        detections = self._decode_predictions(outputs, scale, original_size)
        
        print(f"推理耗时: {inference_time:.3f}秒, 检测到 {len(detections)} 个人脸")
        
        return detections

def draw_detections(image: np.ndarray, detections: List[Dict], 
                   draw_landmarks: bool = True, show_frontal_info: bool = True) -> np.ndarray:
    """在图像上绘制检测结果"""
    result_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        frontal_confidence = detection.get('frontal_confidence', 0.0)
        combined_confidence = detection.get('combined_confidence', confidence)
        is_frontal = detection.get('is_frontal', False)
        landmark_count = detection.get('landmark_count', 0)
        landmarks = detection.get('landmarks', [])
        
        # 根据是否为正脸选择边界框颜色
        box_color = (0, 255, 0) if is_frontal else (0, 165, 255)  # 绿色表示正脸，橙色表示侧脸
        
        # 绘制边界框
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), box_color, 2)
        
        # 绘制置信度和正脸信息
        if show_frontal_info:
            # 第一行：检测置信度
            label1 = f'Det: {confidence:.3f}'
            cv2.putText(result_image, label1, (x1, y1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            # 第二行：正脸置信度
            label2 = f'Front: {frontal_confidence:.3f}'
            cv2.putText(result_image, label2, (x1, y1 - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            # 第三行：融合置信度和正脸标识
            frontal_text = "正脸" if is_frontal else "侧脸"
            label3 = f'Comb: {combined_confidence:.3f} ({frontal_text})'
            cv2.putText(result_image, label3, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            # 第四行：关键点数量
            label4 = f'Points: {landmark_count}'
            cv2.putText(result_image, label4, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        else:
            # 简化显示
            label = f'{combined_confidence:.3f}'
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # 绘制关键点
        if draw_landmarks and landmarks:
            # 正脸用蓝色关键点，侧脸用红色关键点
            landmark_color = (255, 0, 0) if is_frontal else (0, 0, 255)
            for landmark in landmarks:
                cv2.circle(result_image, tuple(landmark), 2, landmark_color, -1)
    
    return result_image