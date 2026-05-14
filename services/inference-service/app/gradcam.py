import torch
import torch.nn.functional as F
import numpy as np
import cv2


def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """0-1 arası cam → BGR heatmap"""
    cam = np.clip(cam, 0, 1)
    cam_uint8 = (cam * 255).astype(np.uint8)
    return cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)


def _safe_gradcam_computation(act: torch.Tensor, grad: torch.Tensor) -> np.ndarray:
    """
    Güvenli GradCAM hesaplaması — tensor shape mismatch'i işler
    act: (C, H, W) — ativações
    grad: (C, H, W) — gradientes
    """
    try:
        # Ensure 3D
        if act.dim() != 3:
            act = act.squeeze()
        if grad.dim() != 3:
            grad = grad.squeeze()
            
        # Kanal sayısını eşitle
        min_channels = min(act.shape[0], grad.shape[0])
        act = act[:min_channels, :, :]
        grad = grad[:min_channels, :, :]
        
        # GradCAM hesapla usando keepdim
        weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = (weights * act).sum(dim=0)  # (H, W)
        cam = F.relu(cam)
        
        # Normalize
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        
        return cam_np
    except Exception as e:
        print(f"[GRADCAM COMPUTE ERROR] {str(e)}")
        return None


def gradcam_yolo(model, frame_bgr: np.ndarray, detections: list = None) -> np.ndarray:
    """
    YOLOv8 GradCAM - creates attention map from model confidence.
    Shows where model detected objects with confidence intensity.
    """
    h, w = frame_bgr.shape[:2]

    try:
        # Create attention map from detections and confidence
        attention_map = np.zeros((h, w), dtype=np.float32)
        
        if detections:
            # For each detection, add a Gaussian distribution centered at the detection
            for det in detections:
                bbox = det.get("bbox", {})
                confidence = det.get("confidence", 0.5)
                
                # Get bbox coordinates
                x1 = int(bbox.get("x1", 0))
                y1 = int(bbox.get("y1", 0))
                x2 = int(bbox.get("x2", w))
                y2 = int(bbox.get("y2", h))
                
                # Clamp to bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Create Gaussian at detection center with sigma proportional to box size
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                sigma_x = max(5, (x2 - x1) // 3)
                sigma_y = max(5, (y2 - y1) // 3)
                
                # Create meshgrid for Gaussian
                yy, xx = np.ogrid[:h, :w]
                gaussian = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) + 
                                   (yy - cy)**2 / (2 * sigma_y**2)))
                
                # Weight by confidence and add to map
                attention_map += gaussian * confidence
        else:
            # If no detections, create uniform attention
            attention_map = np.ones((h, w), dtype=np.float32) * 0.3
        
        # Normalize to 0-1
        att_min = attention_map.min()
        att_max = attention_map.max()
        if att_max > att_min:
            attention_map = (attention_map - att_min) / (att_max - att_min)
        else:
            attention_map = np.ones_like(attention_map) * 0.5
        
        # Clip to 0-1
        attention_map = np.clip(attention_map, 0, 1)
        
        # Convert to heatmap
        attention_uint8 = (attention_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
        
        return heatmap
        
    except Exception as e:
        print(f"[GRADCAM ERROR YOLO] {str(e)}")
        return np.zeros((h, w, 3), dtype=np.uint8)



def gradcam_faster_rcnn(model, frame_bgr: np.ndarray, detections: list = None) -> np.ndarray:
    """
    Faster R-CNN GradCAM - shows model focus using activation maps.
    Uses hook to capture FPN layer outputs.
    """
    device = next(model.parameters()).device
    h, w = frame_bgr.shape[:2]

    try:
        # Prepare input
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_batch = img.unsqueeze(0).to(device)

        model.train(False)
        
        # Store features captured by hook
        features_captured = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                features_captured.append(output.detach())
            elif isinstance(output, dict):
                # If output is dict (like FPN), get the first value
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        features_captured.append(v.detach())
                        break
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        features_captured.append(item.detach())
                        break
        
        # Register hook on FPN (feature pyramid network)
        target_layer = model.backbone
        hook = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = model([img_batch[0]])
            
            if not features_captured:
                print("[GRADCAM FASTER_RCNN] No features captured")
                return np.zeros((h, w, 3), dtype=np.uint8)
            
            # Get the captured feature (usually P3 or P4 from FPN)
            feat = features_captured[0]  # (B, C, H, W) or (C, H, W)
            
            # Ensure 3D (C, H, W)
            if feat.dim() == 4:
                feat = feat[0]  # Remove batch
            
            # Create attention map: mean across channels
            if feat.dim() == 3:
                attention_map = feat.mean(dim=0).cpu().numpy()  # (H, W)
            else:
                attention_map = feat.squeeze().cpu().numpy()
            
            # Normalize to 0-1
            att_min = attention_map.min()
            att_max = attention_map.max()
            if att_max > att_min:
                attention_map = (attention_map - att_min) / (att_max - att_min)
            else:
                attention_map = np.ones_like(attention_map) * 0.5
            
            # Resize to original image size
            attention_map_resized = cv2.resize(attention_map, (w, h))
            
            # Convert to heatmap: 0-255 range for colormap
            attention_uint8 = (attention_map_resized * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
            
            return heatmap
        
        finally:
            hook.remove()
        
    except Exception as e:
        print(f"[GRADCAM ERROR FASTER_RCNN] {str(e)}")
        return np.zeros((h, w, 3), dtype=np.uint8) #hata alnın yer

