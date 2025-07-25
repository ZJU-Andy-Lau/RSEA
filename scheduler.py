from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List
class MultiStageOneCycleLR(_LRScheduler):
    """
    一个自定义的学习率调度器，它结合了线性预热、保持和余弦退火。

    该调度器将学习率在训练过程中分为三个阶段进行调整：
    1. 线性预热：在前 `warmup_steps` 步中，学习率从0线性增加到 `base_lr`。
    2. 保持阶段：在预热之后，学习率保持为 `base_lr`。
    3. 余弦退火：在最后的 `cooldown_steps` 步中，学习率按余弦曲线从 `base_lr` 衰减到0。

    Args:
        optimizer (Optimizer): 被包装的优化器。
        total_steps (int): 训练过程的总步数。
        warmup_ratio (float): 用于线性预热的步数占总步数的比例。
        cooldown_ratio (float): 用于余弦退火的步数占总步数的比例。
        last_epoch (int, optional): 最后一个周期的索引。默认为 -1。
        verbose (bool, optional): 如果为 True，则每次更新时打印一条信息。默认为 False。
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_ratio: float, cooldown_ratio: float, last_epoch: int = -1, verbose: bool = False):
        # 计算各个阶段的步数
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.cooldown_steps = int(total_steps * cooldown_ratio)
        
        # 计算保持阶段的步数
        self.constant_steps = total_steps - self.warmup_steps - self.cooldown_steps
        
        # 进行合法性检查
        if self.constant_steps < 0:
            raise ValueError("预热比例和退火比例的总和不能超过1。")

        # 计算退火阶段的起始步
        self.cooldown_start_step = self.warmup_steps + self.constant_steps

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        根据当前步数计算学习率。
        这是 _LRScheduler 的核心方法，由 step() 方法在内部调用。
        """
        # self.last_epoch 是由 _LRScheduler 基类管理的当前步数（从0开始）
        current_step = self.last_epoch

        # --- 阶段1: 线性预热 ---
        # 仅在 warmup_steps > 0 且当前步数在预热阶段内时执行
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            # 计算预热比例因子
            # +1 是为了确保在第0步时学习率不为0（如果需要），但通常从0开始更常见
            warmup_factor = float(current_step + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # --- 阶段3: 余弦退火 ---
        # 如果当前步数已经进入或超过了退火的起始步
        elif current_step >= self.cooldown_start_step:
            # 计算当前在退火阶段的进度
            step_in_cooldown = current_step - self.cooldown_start_step
            
            # 避免在 cooldown_steps 为 0 时出现除零错误
            if self.cooldown_steps == 0:
                cooldown_progress = 1.0
            else:
                # 确保进度在 [0, 1] 范围内
                cooldown_progress = min(float(step_in_cooldown) / float(self.cooldown_steps), 1.0)
            
            # 使用余弦退火公式计算衰减因子
            cooldown_factor = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
            
            return [base_lr * cooldown_factor for base_lr in self.base_lrs]

        # --- 阶段2: 保持阶段 ---
        # 如果不处于预热或退火阶段，则学习率保持为基础学习率
        else:
            return [base_lr for base_lr in self.base_lrs]