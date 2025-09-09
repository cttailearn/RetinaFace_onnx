import torch
import json
import argparse
from collections import OrderedDict

# 导入模型定义
from models import RetinaFace

def convert_pt_to_onnx(model_path, config_path, output_path, input_shape=(1, 3, 640, 640)):
    """
    将PyTorch模型转换为ONNX格式
    
    参数:
        model_path: PyTorch模型权重路径
        config_path: 配置文件路径
        output_path: 输出ONNX模型路径
        input_shape: 输入张量形状
    """
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config["models"]
    
    # 创建模型实例
    model = RetinaFace(model_config)
    
    # 加载模型权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    # 处理可能的权重键名不匹配问题
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # 移除可能的模块前缀（如果模型是用DataParallel训练的）
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape, device=device)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # ONNX算子集版本
        do_constant_folding=True,
        input_names=['input'],
        output_names=['loc', 'conf', 'landms'],  # 根据实际输出修改
        dynamic_axes={
            'input': {0: 'batch_size'},  # 批处理维度动态
            'loc': {0: 'batch_size'},
            'conf': {0: 'batch_size'},
            'landms': {0: 'batch_size'}
        }
    )
    
    print(f"模型已成功导出到: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch to ONNX转换工具')
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--config_path', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_path', type=str, default='model.onnx', help='输出ONNX模型路径')
    parser.add_argument('--input_shape', type=str, default='1,3,640,640', help='输入形状，格式: batch,channel,height,width')
    
    args = parser.parse_args()
    
    # 解析输入形状
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    convert_pt_to_onnx(
        model_path=args.model_path,
        config_path=args.config_path,
        output_path=args.output_path,
        input_shape=input_shape
    )