import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dims=[512, 256]):
        super().__init__()
        # Assumes embeddings are already averaged before input
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return F.normalize(self.mlp(x), p=2, dim=1)

class PostTower(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return F.normalize(self.mlp(x), p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, post_tower):
        super().__init__()
        self.user_tower = user_tower
        self.post_tower = post_tower
        
    def forward(self, user_features, post_features):
        user_emb = self.user_tower(user_features)
        post_emb = self.post_tower(post_features)
        return user_emb, post_emb

    def get_user_embeddings(self, user_features):
        return self.user_tower(user_features)
    
    def get_post_embeddings(self, post_features):
        return self.post_tower(post_features)