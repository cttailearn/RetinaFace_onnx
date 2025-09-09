#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量人脸检测脚本

支持通过命令行参数进行批量人脸检测，可处理单张图像或递归搜索目录下的所有图像
"""

import os
import sys
import cv2
import argparse
import time
from pathlib import Path
from typing import List, Optional
from inference_onnx import ONNXInference, draw_detections

def get_image_files(path: str, recursive: bool = True) -> List[str]:
    """
    获取图像文件列表
    
    Args:
        path: 图像文件路径或目录路径
        recursive: 是否递归搜索子目录
        
    Returns:
        图像文件路径列表
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        # 单个文件
        if path_obj.suffix.lower() in supported_formats:
            image_files.append(str(path_obj))
        else:
            print(f"警告: {path} 不是支持的图像格式")
    elif path_obj.is_dir():
        # 目录
        if recursive:
            # 递归搜索
            for ext in supported_formats:
                image_files.extend([str(p) for p in path_obj.rglob(f'*{ext}')])
                image_files.extend([str(p) for p in path_obj.rglob(f'*{ext.upper()}')])
        else:
            # 仅搜索当前目录
            for ext in supported_formats:
                image_files.extend([str(p) for p in path_obj.glob(f'*{ext}')])
                image_files.extend([str(p) for p in path_obj.glob(f'*{ext.upper()}')])
    else:
        print(f"错误: 路径 {path} 不存在")
        return []
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    print(f"找到 {len(image_files)} 个图像文件")
    
    return image_files

def generate_output_path(input_path: str, output_dir: Optional[str] = None, 
                        suffix: str = "_detected") -> str:
    """
    生成输出文件路径
    
    Args:
        input_path: 输入文件路径
        output_dir: 输出目录，如果为None则覆盖原文件
        suffix: 文件名后缀（当不覆盖原文件时使用）
        
    Returns:
        输出文件路径
    """
    input_path_obj = Path(input_path)
    
    if output_dir is None:
        # 覆盖原文件
        return str(input_path_obj)
    else:
        # 保存到指定目录
        output_dir_obj = Path(output_dir)
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # 生成新文件名
        stem = input_path_obj.stem
        suffix_part = suffix if not output_dir else ""
        new_name = f"{stem}{suffix_part}{input_path_obj.suffix}"
        
        return str(output_dir_obj / new_name)

def process_single_image(inferencer: ONNXInference, input_path: str, 
                        output_path: str, confidence_threshold: float) -> bool:
    """
    处理单张图像
    
    Args:
        inferencer: ONNX推理器
        input_path: 输入图像路径
        output_path: 输出图像路径
        confidence_threshold: 置信度阈值
        
    Returns:
        是否处理成功
    """
    try:
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"错误: 无法读取图像 {input_path}")
            return False
        
        # 临时设置置信度阈值
        original_vis_thres = inferencer.vis_thres
        inferencer.vis_thres = confidence_threshold
        
        # 执行推理
        start_time = time.time()
        detections = inferencer.inference(image)
        inference_time = time.time() - start_time
        
        # 恢复原始阈值
        inferencer.vis_thres = original_vis_thres
        
        # 绘制检测结果
        result_image = draw_detections(image, detections, 
                                     draw_landmarks=True, 
                                     show_frontal_info=True)
        
        # 保存结果
        success = cv2.imwrite(output_path, result_image)
        if not success:
            print(f"错误: 无法保存结果到 {output_path}")
            return False
        
        # 统计信息
        frontal_count = sum(1 for d in detections if d.get('is_frontal', False))
        profile_count = len(detections) - frontal_count
        
        print(f"✓ {Path(input_path).name}: {len(detections)}个人脸 "
              f"(正脸:{frontal_count}, 侧脸:{profile_count}) "
              f"耗时:{inference_time:.3f}s -> {Path(output_path).name}")
        
        return True
        
    except Exception as e:
        print(f"错误: 处理图像 {input_path} 时发生异常: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量人脸检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图像，覆盖原文件
  python batch_face_detection.py -m model.onnx -i image.jpg
  
  # 处理目录下所有图像，保存到指定目录
  python batch_face_detection.py -m model.onnx -i ./images -o ./results
  
  # 递归搜索并设置置信度阈值
  python batch_face_detection.py -m model.onnx -i ./dataset -c 0.5 --recursive
  
  # 不递归搜索，覆盖原文件
  python batch_face_detection.py -m model.onnx -i ./images --no-recursive --overwrite
        """
    )
    
    parser.add_argument('-m', '--model', required=True, type=str,
                       help='ONNX模型文件路径')
    parser.add_argument('-i', '--input', required=True, type=str,
                       help='输入图像文件或目录路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出目录路径（默认覆盖原文件）')
    parser.add_argument('-c', '--confidence', type=float, default=0.3,
                       help='置信度阈值 (默认: 0.3)')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='递归搜索子目录 (默认: True)')
    parser.add_argument('--no-recursive', action='store_true',
                       help='不递归搜索子目录')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖原文件（忽略--output参数）')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='推理设备 (默认: cpu)')
    parser.add_argument('--suffix', type=str, default='_detected',
                       help='输出文件名后缀（当不覆盖原文件时使用，默认: _detected）')
    
    args = parser.parse_args()
    
    # 参数验证
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    if args.confidence < 0 or args.confidence > 1:
        print(f"错误: 置信度阈值必须在0-1之间: {args.confidence}")
        sys.exit(1)
    
    # 处理递归参数
    recursive = args.recursive and not args.no_recursive
    
    # 处理输出参数
    output_dir = None if args.overwrite else args.output
    
    print("=== 批量人脸检测工具 ===")
    print(f"模型文件: {args.model}")
    print(f"输入路径: {args.input}")
    print(f"输出设置: {'覆盖原文件' if output_dir is None else f'保存到 {output_dir}'}")
    print(f"置信度阈值: {args.confidence}")
    print(f"递归搜索: {'是' if recursive else '否'}")
    print(f"推理设备: {args.device}")
    print()
    
    try:
        # 初始化推理器
        print("初始化ONNX推理器...")
        inferencer = ONNXInference(args.model, device=args.device)
        print("推理器初始化完成\n")
        
        # 获取图像文件列表
        print("搜索图像文件...")
        image_files = get_image_files(args.input, recursive=recursive)
        
        if not image_files:
            print("未找到任何图像文件")
            sys.exit(1)
        
        print(f"开始处理 {len(image_files)} 个图像文件...\n")
        
        # 批量处理
        success_count = 0
        total_start_time = time.time()
        
        for i, input_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 处理中: {Path(input_path).name}", end=" ")
            
            # 生成输出路径
            output_path = generate_output_path(input_path, output_dir, args.suffix)
            
            # 处理图像
            if process_single_image(inferencer, input_path, output_path, args.confidence):
                success_count += 1
            else:
                print(f"✗ 处理失败")
        
        total_time = time.time() - total_start_time
        
        # 统计结果
        print(f"\n=== 处理完成 ===")
        print(f"总计处理: {len(image_files)} 个文件")
        print(f"成功处理: {success_count} 个文件")
        print(f"失败处理: {len(image_files) - success_count} 个文件")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均耗时: {total_time/len(image_files):.3f}秒/图像")
        
        if success_count == len(image_files):
            print("\n🎉 所有图像处理成功！")
        else:
            print(f"\n⚠️  有 {len(image_files) - success_count} 个图像处理失败")
            
    except KeyboardInterrupt:
        print("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()