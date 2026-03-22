import torch
import torch.nn.functional as F 

def gaussian_kernel(mu, bandwidth, datapoints):
    dist = torch.norm(datapoints - mu,dim=-1, p=2)
    density = torch.exp(-dist**2/(2*bandwidth**2))
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
    
    # Coarse-to-fine bandwidth schedule
    # Calculate base bandwidth from k-nearest neighbor distances
    dist = torch.cdist(image_features, image_features)
    sorted_dist, _ = torch.sort(dist, dim=1)
    k = int(0.3 * (image_features.shape[0]-1))
    selected_distances = sorted_dist[:, 1:k+1]**2  # exclude the distance to the point itself 
    mean_distance = torch.mean(selected_distances, dim=1)
    base_bandwidth = torch.sqrt(0.5 * mean_distance)
    
    # Bandwidth schedule: start with larger bandwidth, progressively reduce
    # Initial multiplier (coarse stage) and final multiplier (fine stage)
    # Read values from args when provided, preserve previous defaults otherwise
    bandwidth_init_scale = getattr(args, 'bandwidth_init_scale', 2.0)
    bandwidth_final_scale = getattr(args, 'bandwidth_final_scale', 0.8)
    
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
        # Coarse-to-fine bandwidth schedule: support multiple decay types.
        # By default use exponential (geometric) interpolation between init and final scales.
        # You can set `args.bandwidth_schedule` to one of: 'exponential' (default), 'linear', 'cosine', 'inverse'.
        decay_type = getattr(args, 'bandwidth_schedule', 'exponential')
        progress = min(iter / (max_iter - 1), 1.0) if max_iter > 1 else 1.0

        if decay_type == 'linear':
            # linear interpolation between init and final scales
            bandwidth_scale = bandwidth_init_scale + progress * (bandwidth_final_scale - bandwidth_init_scale)
        elif decay_type == 'exponential':
            # geometric interpolation (smooth exponential-like decay)
            # bandwidth_scale = init * (final/init) ** progress
            if bandwidth_init_scale == 0:
                bandwidth_scale = bandwidth_final_scale
            else:
                bandwidth_scale = bandwidth_init_scale * (bandwidth_final_scale / bandwidth_init_scale) ** progress
        elif decay_type == 'cosine':
            # cosine annealing between init and final
            import math
            bandwidth_scale = bandwidth_final_scale + 0.5 * (bandwidth_init_scale - bandwidth_final_scale) * (1 + math.cos(math.pi * progress))
        elif decay_type == 'inverse':
            # inverse schedule: quick initial change then slow approach to final
            # using 10 as a shape parameter (larger -> steeper initial drop)
            bandwidth_scale = bandwidth_final_scale + (bandwidth_init_scale - bandwidth_final_scale) * (1.0 / (1.0 + 10.0 * progress))
        else:
            # fallback to linear if unknown string
            bandwidth_scale = bandwidth_init_scale + progress * (bandwidth_final_scale - bandwidth_init_scale)

        bandwidth = base_bandwidth * bandwidth_scale
        
        ###################
        # Inlierness step #
        ###################
        
        density = gaussian_kernel(mode, bandwidth, image_features)
    
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
            density = gaussian_kernel(mode, bandwidth, image_features)
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
