def adaptive_haze_correction(img_tensor):
    if img_tensor is None or img_tensor.shape[1] != 3:
        return img_tensor
    device = img_tensor.device
    R, G, B = img_tensor[:, 0, ...], img_tensor[:, 1, ...], img_tensor[:, 2, ...]
    Luma = 0.299 * R + 0.587 * G + 0.114 * B
    Max_Val, _ = torch.max(img_tensor, dim=1)
    safe_limit_g = G * 1.20
    safe_limit_r = R * 1.20 + 0.02
    safe_blue_limit = torch.max(safe_limit_g, safe_limit_r)
    excess_blue = torch.relu(B - safe_blue_limit)
    mask_haze = torch.sigmoid((Luma - 0.30) * 10)
    mask_light = 1.0 - torch.sigmoid((Max_Val - 0.95) * 50)
    global_overflow = torch.mean(excess_blue)
    global_weight = torch.clamp(global_overflow * 100, 0.0, 1.0)
    final_mask = mask_haze * mask_light * global_weight  
    strength = 0.9  
    B_new = B - (excess_blue * final_mask * strength)   
    temp = img_tensor.clone()
    temp[:, 2] = B_new   
    Luma_temp = 0.299 * temp[:,0] + 0.587 * temp[:,1] + 0.114 * temp[:,2]  
    ratio = Luma / (Luma_temp + 1e-6)  
    ratio = torch.clamp(ratio, 0.5, 4.0)  
    output = temp * ratio.unsqueeze(1)
       
    return torch.clamp(output, 0.0, 1.0)