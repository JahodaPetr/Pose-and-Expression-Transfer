import torch
from torch import nn

"""
LMCL - Loss explained in paper
"""


class CosFace(nn.Module):
    def __init__(self, s, m):
        super(CosFace, self).__init__()
        self.cos_sim = nn.CosineSimilarity()
        self.s = s
        self.m = m

    def forward(self, zs_d, zss_d, zd, zds):
        zs_d = torch.flatten(zs_d, start_dim=1)
        zss_d = torch.flatten(zss_d, start_dim=1)
        zd = torch.flatten(zd, start_dim=1)
        zds = torch.flatten(zds, start_dim=1)
        p1 = torch.exp(self.cos_dist(zs_d, zd))
        p2 = torch.exp(self.cos_dist(zss_d, zd))
        n1 = torch.exp(self.cos_dist(zs_d, zds))
        n2 = torch.exp(self.cos_dist(zss_d, zds))
        return (
            torch.sum(
                -(
                    torch.log((p1 / (p1 + n1 + n2)) + 1e-8)
                    + torch.log((p2 / (p2 + n1 + n2)) + 1e-8)
                )
            )
            / zs_d.shape[0]
        )

    def cos_dist(self, zi, zj):
        return self.s * ((self.cos_sim(zi, zj)) - self.m)
