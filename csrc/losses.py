import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, projections, labels):
        # projections: (2N, embed_dim), already L2-normalized
        # labels: (2N,)

        device = projections.device
        batch_size = projections.shape[0]  # 2N

        # all pairwise dot products, scaled by temperature
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature

        # mask removes the diagonal
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # positive mask is the P(i), finds all views from same label. 
        labels = labels.unsqueeze(0)  # (1, 2N)
        positive_mask = (labels == labels.T) & ~self_mask  # (2N, 2N)

        # for numerical stability, subtract the max from each row
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True).values

        # denominator: exp of all similarities except self
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * ~self_mask  # zero out diagonal
        denominator = exp_sim.sum(dim=1, keepdim=True)  # (2N, 1)

        # log probability for each pair
        log_prob = similarity_matrix - torch.log(denominator)

        # average over positives for each anchor
        # sum of log_prob where positive, divided by number of positives
        positive_log_prob = (log_prob * positive_mask).sum(dim=1)  # (2N,)
        num_positives = positive_mask.sum(dim=1)  # (2N,)

        loss = -positive_log_prob / num_positives
        loss = loss.mean()

        return loss
            


            
