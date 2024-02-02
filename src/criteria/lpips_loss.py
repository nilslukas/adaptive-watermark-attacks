import lpips
import torch.nn as nn


class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.lpips = lpips.LPIPS(net='alex').cuda()

    def forward(self, x, y):
        # normalize to [-1, 1]
        x = (x * 2 - 1)
        y = (y * 2 - 1)
        return self.lpips(x, y)  # normalize to [-1, 1]
