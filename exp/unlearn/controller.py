"""模态混合系数 γ 的控制器。

- `static`：常数 γ（baseline 组）——**不需要 margin**，框架据此跳过多余前向。
- `ema_level`：`γ = σ(α·ŝ)`，`ŝ` 为 gap 的水平 EMA（含偏差校正；MAW 组）。
  `rho=0` 时退化为 `γ = σ(α·gap)`（纯瞬时水平信号）。

逻辑与旧 `MAW.py` 训练循环内的实现逐行一致。
"""
import math

STATIC_MODES = ("static",)
DYNAMIC_MODES = ("ema_level",)


class GammaController:
    def __init__(self, args):
        self.mode = args.gamma_mode
        self.alpha = args.alpha
        self.rho = args.rho
        self.bias_correct = args.bias_correct
        self.gamma_const = args.gamma
        self.gap_ema = 0.0
        self.level_ema = 0.0
        self.step = 0

    @property
    def needs_margin(self) -> bool:
        return self.mode in DYNAMIC_MODES

    def static_value(self) -> float:
        return float(self.gamma_const)

    def update(self, gap: float):
        """按当前 gap 更新 γ，返回 (gamma, info)。"""
        assert self.mode in DYNAMIC_MODES, f"{self.mode} takes no gap"
        self.step += 1
        gap_ema_prev = self.gap_ema
        s_hat = 0.0
        # ema_level
        self.level_ema = self.rho * self.level_ema + (1.0 - self.rho) * gap
        if self.bias_correct == "off" or self.rho >= 1.0:
            s_hat = float(gap) if self.rho >= 1.0 else self.level_ema
        else:
            denom = 1.0 - (self.rho ** self.step)
            s_hat = self.level_ema / denom if denom > 1e-8 else float(gap)
        gamma = 1.0 / (1.0 + math.exp(-(self.alpha * s_hat)))
        self.gap_ema = self.rho * self.gap_ema + (1.0 - self.rho) * gap
        info = {
            "s": gap - gap_ema_prev,
            "alpha_s": self.alpha * (gap - gap_ema_prev),
            "gap_ema": self.gap_ema,
            "level_ema": self.level_ema,
            "s_hat": s_hat,
        }
        return gamma, info

    def state(self) -> dict:
        return {"gap_ema_final": float(self.gap_ema), "rho": self.rho, "mode": self.mode}
