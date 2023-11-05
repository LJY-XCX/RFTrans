import torch
from torch import nn
from utils.utils import flow_to_rgb

class RGB2NormalModel(nn.Module):
    def __init__(self, r2f_model, f2n_model):
        super(RGB2NormalModel, self).__init__()
        self.r2f_model = r2f_model
        self.f2n_model = f2n_model

    def forward(self, rgb):
        flow = self.r2f_model(rgb)
        rgb_flow = flow_to_rgb(flow.detach().cpu())
        rgb_flow = rgb_flow.to(rgb.device)
        f2n_input = torch.zeros_like(rgb_flow, dtype=torch.float32)
        f2n_input[:, :2, :, :] = flow
        normal = self.f2n_model(f2n_input)
        return flow, rgb_flow, normal

    def flow_loss(self, predicted_flow, target_flow, criterion):
        return criterion(predicted_flow, target_flow)

    def normal_loss(self, predicted_normal, target_normal, criterion):
        return criterion(predicted_normal, target_normal)

    def to(self, device):
        self.r2f_model.to(device)
        self.f2n_model.to(device)
        return self

    def train(self, mode=True):
        self.r2f_model.train()
        self.f2n_model.train()

    def eval(self):
        self.r2f_model.eval()
        self.f2n_model.eval()

