import math

import torch
import torch.nn.functional as F

try:
    from xfuser.model_executor.layers.usp import USP
except ImportError:
    USP = None

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

MEMORY_LAYOUT = {
    # flash模式:
    # 预处理: 输入 [batch_size, seq_len, num_heads, head_dim]
    # 后处理: 保持形状不变
    "flash": (
        lambda x: x,  # 保持形状
        lambda x: x,  # 保持形状
    ),
    # torch/vanilla模式:
    # 预处理: 交换序列和注意力头的维度 [B,S,A,D] -> [B,A,S,D]
    # 后处理: 交换回原始维度 [B,A,S,D] -> [B,S,A,D]
    "torch": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xdit": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    )
}


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    """
    执行QKV自注意力计算

    Args:
        q (torch.Tensor): 查询张量，形状 [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): 键张量，形状 [batch_size, seq_len_kv, num_heads, head_dim]
        v (torch.Tensor): 值张量，形状 [batch_size, seq_len_kv, num_heads, head_dim]
        mode (str): 注意力模式，可选 'flash', 'torch', 'vanilla'
        drop_rate (float): 注意力矩阵的dropout概率
        attn_mask (torch.Tensor): 注意力掩码，形状根据模式不同而变化
        causal (bool): 是否使用因果注意力（仅关注前面位置）

    Returns:
        torch.Tensor: 注意力输出，形状 [batch_size, seq_len, num_heads * head_dim]
    """
    # 获取预处理和后处理函数
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    # 应用预处理变换
    q = pre_attn_layout(q)  # 形状根据模式变化
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        # 使用PyTorch原生的scaled_dot_product_attention
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        assert flash_attn_func is not None, "flash_attn_func未定义"
        assert attn_mask is None, "不支持的注意力掩码"
        x: torch.Tensor = flash_attn_func(
            q, k, v, dropout_p=drop_rate, causal=causal, softmax_scale=None
        )  # type: ignore
    elif mode == "vanilla":
        # 手动实现注意力机制
        scale_factor = 1 / math.sqrt(q.size(-1))  # 缩放因子 1/sqrt(d_k)

        b, a, s, _ = q.shape  # 获取形状参数
        s1 = k.size(2)  # 键值序列长度

        # 初始化注意力偏置
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)

        # 处理因果掩码
        if causal:
            assert attn_mask is None, "因果掩码和注意力掩码不能同时使用"
            # 生成下三角因果掩码
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(q.dtype)

        # 处理自定义注意力掩码
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask  # 允许类似ALiBi的位置偏置

        # 计算注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * scale_factor  # [B,A,S,S1]
        attn += attn_bias

        # softmax和dropout
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)

        # 计算输出
        x = attn @ v  # [B,A,S,D]
    elif mode == "xdit":
        x: torch.Tensor = USP(q, k, v, dropout_p=drop_rate, is_causal=causal)
    else:
        raise NotImplementedError(f"不支持的注意力模式: {mode}")

    # 应用后处理变换
    x = post_attn_layout(x)  # 恢复原始维度顺序

    # 合并注意力头维度
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)  # [B,S,A*D]
    return out
