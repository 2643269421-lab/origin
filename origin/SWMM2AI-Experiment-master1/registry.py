from typing import Dict, Type, Any
import torch.nn as nn

# 模型注册表
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """模型装饰器，用于注册模型类"""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model_class(model_type: str) -> Type[nn.Module]:
    """根据模型类型名称获取模型类"""
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"模型类型 '{model_type}' 未注册。可用的模型: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[model_type]

def create_model(model_type: str, **kwargs) -> nn.Module:
    """创建模型实例"""
    model_cls = get_model_class(model_type)
    return model_cls(**kwargs)