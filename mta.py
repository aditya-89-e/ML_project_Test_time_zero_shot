import torch
import torch.nn.functional as F 

def von_mises_fisher_kernel(mu, kappa, datapoints):
    """
    von Mises-Fisher kernel - the natural distribution for normalized embeddings
    on a hypersphere (like CLIP features). Equivalent to Gaussian for directional data.

    Args:
        mu: mode/center on the hypersphere (normalized)
        kappa: concentration parameter (higher = more peaked, analogous to 1/bandwidth^2)
        datapoints: normalized feature vectors
    """
    # Cosine similarity (dot product of normalized vectors)
    cos_sim = torch.sum(mu * datapoints, dim=-1)
    # vMF density (proportional to exp(kappa * cos_sim))
    density = torch.exp(kappa * (cos_sim - 1))  # subtract 1 for numerical stability
    return density


def solve_mta(model, inputs, args):
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            image_features, text_features, logit_scale = model(inputs, features=True)
    logits = image_features @ text_features.t() * logit_scale 
        
    lambda_y = args.lambda_y
    lambda_q = args.lambda_q
    max_iter = 5
    temperature = 1
    
    batch_size = image_features.shape[0]
    
    # Concentration parameter (kappa) for von Mises-Fisher kernel
    # For normalized vectors: ||a-b||^2 = 2(1-cos(a,b)), so kappa ~ 1/bandwidth^2
    dist = torch.cdist(image_features, image_features)
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = int(0.3 * (image_features.shape[0]-1))
    selected_distances = sorted_dist[:, 1:k+1]**2  # exclude the distance to the point itself
    mean_distance = torch.mean(selected_distances, dim=1)
    bandwidth = torch.sqrt(0.5 * mean_distance)
    kappa = 1.0 / (bandwidth**2 + 1e-8)  # concentration parameter 
    
    # Affinity matrix based on logits
    affinity_matrix = (logits/temperature).softmax(1) @ (logits/temperature).softmax(1).t()
    
    # Inlierness scores initialization: uniform
    y = torch.ones(batch_size, device=image_features.device)/batch_size
    
    # Mode initialization: original image embedding
    mode_init = image_features[0]
    mode = mode_init
    
    convergence = False
    th = 1e-6
    iter = 0
    
    while not convergence:
        
        ###################
        # Inlierness step #
        ###################
        
        density = von_mises_fisher_kernel(mode, kappa, image_features)
    
        convergence_inlierness = False
        i = 0
        while not convergence_inlierness:
            i+=1
            old_y = y
            weighted_affinity = affinity_matrix * y.unsqueeze(0)
            y = F.softmax(1/lambda_y * (density + lambda_q * torch.sum(weighted_affinity, dim=1)), dim=-1)

            if torch.norm(old_y - y)<th or i>= max_iter:
                convergence_inlierness = True
        
        #############
        # Mode step #
        #############
        
        convergence_mode = False
        i=0
        while not convergence_mode:
            i+=1
            old_mode = mode
            density = von_mises_fisher_kernel(mode, kappa, image_features)
            weighted_density = density *  y
            mode = torch.sum(weighted_density.unsqueeze(1)* image_features, dim=0)/torch.sum(weighted_density)
            mode /= mode.norm(p=2, dim=-1)
            
            if torch.norm(old_mode - mode)<th or i>= max_iter:
                convergence_mode = True
        
        iter +=1
        if iter >= max_iter:
            convergence = True
    
    output = mode.unsqueeze(0) @ text_features.t() * logit_scale
    return output
