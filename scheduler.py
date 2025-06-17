import torch
import torch.nn
import torch.optim as optim
from torch.optim import lr_scheduler
class MultiStageOneCycleLR:
    def __init__(self, 
                 optimizer, 
                 max_lr, 
                 steps_per_epoch, 
                 n_epochs_per_stage,
                 gamma = None, 
                 gamma_factor = 1.0,
                 min_lr = None,
                 lr_decay = 1.0,
                 epoch_decay = 1.0, 
                 pct_start = 0.3,
                 pct_decay = 0.8,
                 summit_hold = 0.1,
                 cooldown = 0.1, 
                 anneal_strategy='cos'):
        """
        自定义多阶段 OneCycleLR 调度器
        
        Args:
            optimizer: 优化器
            max_lr: 初始阶段的最大学习率
            steps_per_epoch: 每个epoch的 step 数
            n_epochs_per_stage: 每个阶段的 epoch 数
            w: 每阶段的学习率衰减因子（如 0.1)
            pct_start: OneCycleLR 中学习率上升占比
            anneal_strategy: OneCycleLR 的学习率下降策略 ('cos' 或 'linear')
        """
        self.optimizer = optimizer
        self.base_max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs_per_stage = n_epochs_per_stage
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.epoch_decay = epoch_decay
        self.pct_start = pct_start
        self.pct_decay = pct_decay
        self.anneal_strategy = anneal_strategy
        self.summit_hold = summit_hold
        self.cooldown = cooldown
        self.gamma_factor = gamma_factor

        # 当前阶段和调度器
        self.current_stage = 0
        self.epoch_count = 0
        self.step_count = 0
        self.scheduler = None
        self.scheduler_type = 0
        self.cooldown_started = False
        self.manual_cooldown = False
        self.gamma = 0.3 ** (1. / (10 * self.steps_per_epoch)) if gamma is None else gamma
        if not self.min_lr is None:
            exp_steps = self.steps_per_epoch * self.n_epochs_per_stage * (1. - self.pct_start - self.summit_hold)
            self.gamma = (self.min_lr / self.base_max_lr) ** (1. / exp_steps)
        self._create_new_scheduler()

    def _create_new_scheduler(self):
        """初始化当前阶段的 OneCycleLR 调度器"""
        stage_max_lr = self.base_max_lr * (self.lr_decay ** self.current_stage)
        # print(f"Stage {self.current_stage + 1}: max_lr = {stage_max_lr:.6f} n_epochs = {self.n_epochs_per_stage} pct_start = {self.pct_start} gamma = {self.gamma} gamma_factor = {self.gamma_factor}")

        # 计算当前阶段的步数
        self.scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=stage_max_lr,
            steps_per_epoch= self.steps_per_epoch,
            epochs=self.n_epochs_per_stage,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=False,
            final_div_factor=40
        )
    
    def _create_exp_scheduler(self):

        # 计算当前阶段的步数
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,self.gamma)
    

    def step(self):
        """更新学习率"""
        if self.step_count < self.steps_per_epoch * self.n_epochs_per_stage * (self.pct_start) or self.step_count >= self.steps_per_epoch * self.n_epochs_per_stage * (self.pct_start + self.summit_hold) or self.manual_cooldown:
            self.scheduler.step()
        self.step_count += 1

        if self.scheduler_type == 0 and (self.step_count >= self.steps_per_epoch * self.n_epochs_per_stage * (self.pct_start + self.summit_hold) or self.manual_cooldown):
            self._create_exp_scheduler()
            self.scheduler_type = 1
            print('switch to exp scheduler')

        if self.scheduler_type == 1 and self.cooldown_started == False and self.step_count >= self.steps_per_epoch * self.n_epochs_per_stage * (1. - self.cooldown):
            self.scheduler.gamma = self.gamma ** 5
            self.cooldown_started = True
            print('start cooldown')

        if self.step_count % self.steps_per_epoch == 0:
            self.epoch_count += 1
            if self.scheduler_type == 1:
                self.scheduler.gamma = min(1.,self.scheduler.gamma * self.gamma_factor)
        if self.epoch_count >= self.n_epochs_per_stage:
            self.n_epochs_per_stage = int(self.n_epochs_per_stage * self.epoch_decay)
            self.pct_start *= self.pct_decay
            self.new_stage()
    
    def cool_down(self):
        self.manual_cooldown = True
        if not self.min_lr is None:
            exp_steps = self.steps_per_epoch * self.n_epochs_per_stage - self.step_count
            self.gamma = (self.min_lr / self.base_max_lr) ** (1. / exp_steps)

    def new_stage(self):
        """进入下一个阶段"""
        self.current_stage += 1
        self.step_count = 0
        self.epoch_count = 0
        self.scheduler_type = 0
        self._create_new_scheduler()

    def get_last_lr(self):
        """获取当前学习率"""
        return self.scheduler.get_last_lr()