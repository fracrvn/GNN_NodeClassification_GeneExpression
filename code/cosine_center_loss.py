import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineCenterLoss(nn.Module):
    """
    Cosine Center Loss.
    Optimized for small data/batch scenarios.
    It minimizes the cosine distance (1 - cosine_similarity) between 
    node embeddings and their assigned class centers.
    """
    def __init__(self, num_classes, feat_dim):
        super(CosineCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        #Trainable parameters: The class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        with torch.no_grad():
            self.centers.copy_(F.normalize(self.centers, p=2, dim=1))

    def forward(self, features, labels):
  
        #Spherical normalization: will force the model to separate the embeddings by rotating (and therefore clustering) them. 
        features = F.normalize(features, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)

        #Cosine Similarity Matrix
        #Similarity between every feature and every class center.
        #[num_features,2]
        similarity_matrix = torch.matmul(features, centers.t())

        #Select only the similarity for the correct class (Ground Truth)
        #Labels need to be [Batch, 1] to act as indices
        labels = labels.view(-1, 1).long()
        
        #Gather extracts values from similarity_matrix at indices specified by labels
        score_correct_class = similarity_matrix.gather(1, labels)

        #Maximize similarity => Minimize (1 - similarity)
        #Similarity range: [-1, 1]. Loss range: [0, 2]
        loss = 1.0 - score_correct_class
        return loss.mean()