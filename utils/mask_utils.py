import torch

def apply_mask(x, masks):
    '''
        @brief: Apply mask to the input
        @params:
            x       : tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
            masks   : list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    '''
    
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)