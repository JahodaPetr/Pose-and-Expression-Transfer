import torch
from torch import nn
from paths_config import model_paths
from model_irse import Backbone

"""
ArcFace backbone weights are necessary to be put into the pretrained_models directory
Downloadable from https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view
"""


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print("Loading ResNet ArcFace")
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        self.facenet.load_state_dict(torch.load(model_paths["ir_se50"]))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            id_logs.append({"diff_target": float(diff_target)})
            loss += 1 - diff_target
            count += 1
        return loss / count, id_logs
