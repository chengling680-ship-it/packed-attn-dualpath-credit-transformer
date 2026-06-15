import torch

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class VarLenMarker:
    """
    专门为变长序列设计的掩码助手
    """
    @staticmethod
    #独立工具函数：只是挂在类名下的普通函数，和实例 / 类无关
    # 直用类名调用：类名.方法名()
    # 也能用实例调用：实例.方法名()
    def get_block_diagonal_mask(cu_seqlens, device):
        """
        生成块对角掩码，确保 Batch 内的不同序列互不干扰
        cu_seqlens: [0, 5, 12, ...] 累计长度
        """
        total_len = cu_seqlens[-1]
        # 1. 创建基础掩码：全为 True (屏蔽)
        mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        
        # 优化建议：虽然 for 循环在 Batch 较小时没问题，但如果序列非常多，
        # 可以保持现状，因为这通常只在每一层初始化一次。
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i+1]
            mask[start:end, start:end] = False # 同一序列内设为 False (可见)
            
        return mask 

class TriangularCausalMask():
    """
    适配变长拼接后的因果掩码：既不能看未来，也不能看别人。
    """
    def __init__(self, cu_seqlens, device="cpu"):
        total_len = cu_seqlens[-1].item() if torch.is_tensor(cu_seqlens) else cu_seqlens[-1]
        #.item()把 GPU 上的单值张量 → 普通 Python 数字（int/float）
        with torch.no_grad():#告诉 PyTorch 不要跟踪这个代码块中的梯度
            # 1. 基础的上三角掩码 (防止看未来)
            # diagonal=1 表示主对角线不屏蔽，对角线以上屏蔽
            basic_mask = torch.triu(
                torch.ones((total_len, total_len), dtype=torch.bool, device=device), 
                diagonal=1
            )
            
            # 2. 生成块对角掩码 (防止看别人)
            # 直接复用 VarLenMarker 的逻辑
            block_mask = VarLenMarker.get_block_diagonal_mask(cu_seqlens, device)
                
            # 3. 合并：(看未来 | 看别人) -> 只要满足其一就屏蔽
            # 结果中 True 代表屏蔽，False 代表可见
            self._mask = (basic_mask | block_mask)
            #|：在 PyTorch 中，对布尔张量（bool） 执行 按位或 / 逻辑或 运算
            #只要有一个是 True，结果就是 True

    @property
    def mask(self):
        return self._mask
