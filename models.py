import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from typing import Literal, Optional, Tuple, Dict, List
from cosine_center_loss import CosineCenterLoss 

CONV_MAPPING = {"GCN": GCNConv, "GAT": GATConv}

class MLP(torch.nn.Module):
    def __init__(self, input_dim: int = 1022, hidden_dim: int = 32, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index=None) -> tuple:
        #Latent embedding
        h = self.encoder(x)
        out = self.classifier(h)
        #I kept the same return signature as the GNN
        return out, None, h 

    def compute_loss(self, out: torch.Tensor, z: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(out[mask], y[mask])
    
    def get_optimizer_params(self, base_lr: float) -> List[Dict]:
        #Dummy method, for compatibility only
        return [{'params': self.parameters(), 'lr': base_lr}]

class GNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1022,
        hidden_dim: int = 32,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        type_layers: Literal["GCN", "GAT"] = "GCN",
        contrastive: bool = False,
        alpha: float = 0.7,
        center_lr: float = 0.05, # LR specifico per i centri
        classifier_type: Literal["linear", "mlp"] = "linear",
        classifier_hidden_dim: int = 32,
    ):
        super().__init__()
        self.dropout = dropout
        self.contrastive = contrastive
        self.alpha = alpha
        self.center_lr = center_lr

        if type_layers not in CONV_MAPPING:
            raise ValueError(f"Layer type not valid: {type_layers}. Choose one among {list(CONV_MAPPING.keys())}")

        ConvLayer = CONV_MAPPING[type_layers]

        # Encoder
        self.conv_layers = nn.ModuleList([ConvLayer(input_dim, hidden_dim)])
        for _ in range(num_layers-1):
            self.conv_layers.append(ConvLayer(hidden_dim, hidden_dim))

        self.center_criterion = None

        if self.contrastive:
            # Center Loss works directly on the embedding produced by the GNN encoder (h)
            self.center_criterion = CosineCenterLoss(num_classes, hidden_dim)

        # Classifier head
        if classifier_type == "linear":
            self.classifier = nn.Linear(hidden_dim, num_classes)
        elif classifier_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes)
            )
        else:
            raise ValueError("classifier_type must be one of: 'linear', 'mlp'")

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 1. Encoder pass
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        z = None
        h = x
        if self.contrastive:
            #CenterLoss works directly on the latent space embeddings
            h = F.normalize(x, p=2, dim=1) 
            z = h

        out = self.classifier(h)
        # 2. Classifier pass (Logits)
        if isinstance(out, tuple): 
            out = out[0]
        #here, z = h. this is done because in a previous iteration i tried another loss, so i left all three of these for compatibility
        return out, z, h

    def compute_loss(self, out: torch.Tensor, h: Optional[torch.Tensor], y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        ce_loss = F.cross_entropy(out[mask], y[mask])
        if not self.contrastive:
            return ce_loss
        aux_loss = self.center_criterion(h[mask], y[mask])
        return self.alpha * ce_loss + (1 - self.alpha) * aux_loss
    
    def get_optimizer_params(self, base_lr: float) -> List[Dict]:
        if self.contrastive:
            center_params = list(self.center_criterion.parameters())
            center_ids = set(map(id, center_params))
            base_params = [p for p in self.parameters() if id(p) not in center_ids]
        
            return [
                #The net uses global weight decay globale
                {'params': base_params, 'lr': base_lr},
                #The centers has no weight decay: they need to stay on the surface of the sphere, not fall into the center.
                {'params': center_params, 'lr': self.center_lr, 'weight_decay': 0.0}
            ]
        else:
            return [{'params': self.parameters(), 'lr': base_lr}]