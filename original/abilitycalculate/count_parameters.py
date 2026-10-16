def count_parameters(model):
    """计算模型参数量 - 最安全的方法"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)")
    print(f"模型大小: {total_params * 2 / (1024 ** 2):.2f} MB (FP16)")

    return total_params, trainable_params