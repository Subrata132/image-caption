import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(decoder_dim, attention_dim)
        self.u = nn.Linear(encoder_dim, attention_dim)
        self.a = nn.Linear(attention_dim, 1)

    def forward(self, image_features, hidden_state):
        u_hs = self.u(image_features)
        w_ah = self.w(hidden_state)
        combined = torch.tanh(u_hs + w_ah.unsqueeze(1))
        attention_score = self.a(combined)
        alpha = F.softmax(attention_score, dim=1)
        attention_weights = image_features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)
        return alpha, attention_weights


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        encoder_cnn = models.resnet50(pretrained=True)
        for parameter in encoder_cnn.parameters():
            parameter.requires_grad_(False)
        modules = list(encoder_cnn.children())[:-2]
        self.encoder_cnn = nn.Sequential(*modules)

    def forward(self, image):
        image_features = self.encoder_cnn(image)
        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(image_features.shape[0], -1, image_features.shape[-1])
        return image_features
