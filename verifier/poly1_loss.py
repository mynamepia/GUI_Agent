import torch
import torch.nn.functional as F


def poly1_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                    epsilon: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    """
    Poly-1 loss for binary classification (Leng et al., ICLR 2022).
    logits: (B,) sigmoid 적용 전 raw logit (QwenVerifier.forward() 출력 그대로)
    targets: (B,) 0/1 float label
    epsilon: >0이면 confidently-wrong 예측에 추가 벌점, 0이면 그냥 BCE와 동일
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)  # 정답 라벨에 대한 예측 확률
    loss = bce + epsilon * (1 - pt)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss