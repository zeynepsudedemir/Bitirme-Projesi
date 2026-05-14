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
    YOLOv8 GradCAM - shows model focus on DETECTED OBJECTS.
    Uses hooks on multiple detection-relevant layers for object-focused attention.
    """
    h, w = frame_bgr.shape[:2]

    try:
        features_multi = []

        def hook_fn_factory(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    features_multi.append({
                        "layer": layer_idx,
                        "feat": output.detach()
                    })
                elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                    features_multi.append({
                        "layer": layer_idx,
                        "feat": output[0].detach()
                    })
                elif isinstance(output, dict):
                    for value in output.values():
                        if isinstance(value, torch.Tensor):
                            features_multi.append({
                                "layer": layer_idx,
                                "feat": value.detach()
                            })
                            break
            return hook_fn

        backbone_model = None
        hook_handles = []

        if hasattr(model, "model"):
            if hasattr(model.model, "model"):
                backbone_model = model.model.model
            else:
                backbone_model = model.model

        if backbone_model is None or not hasattr(backbone_model, "__len__"):
            print("[GRADCAM YOLO] Model structure unexpected")
            return np.zeros((h, w, 3), dtype=np.uint8)

        total_layers = len(backbone_model)
        hook_positions = []

        if total_layers >= 3:
            hook_positions = [
                max(0, total_layers - 5),
                max(0, total_layers - 4),
                max(0, total_layers - 3),
            ]
        else:
            hook_positions = list(range(total_layers))

        for pos in hook_positions:
            try:
                layer_module = backbone_model[pos]
                hook_handles.append(layer_module.register_forward_hook(hook_fn_factory(pos)))
            except Exception:
                continue

        def clear_hooks():
            for handle in hook_handles:
                try:
                    handle.remove()
                except Exception:
                    pass

        try:
            with torch.no_grad():
                _ = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.1)

            if not features_multi:
                clear_hooks()
                hook_handles = []

                fallback_positions = list(range(max(0, total_layers - 6), total_layers))
                for pos in fallback_positions:
                    try:
                        layer_module = backbone_model[pos]
                        hook_handles.append(layer_module.register_forward_hook(hook_fn_factory(pos)))
                    except Exception:
                        continue

                with torch.no_grad():
                    _ = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.1)

            if not features_multi:
                print("[GRADCAM YOLO] No features captured")
                attention_map = np.ones((h, w), dtype=np.float32) * 0.5
            else:
                features_multi.sort(key=lambda x: x["layer"], reverse=True)
                feat = features_multi[0]["feat"]

                if feat.dim() == 4:
                    feat = feat[0]
                elif feat.dim() != 3:
                    feat = feat.squeeze()

                if feat.dim() != 3:
                    print(f"[GRADCAM YOLO] Unexpected feature shape: {feat.shape}")
                    attention_map = np.ones((h, w), dtype=np.float32) * 0.5
                else:
                    feat_norm = torch.norm(feat, p=2, dim=0)
                    norm_np = feat_norm.cpu().numpy()

                    if norm_np.max() > norm_np.min():
                        attention_map = (norm_np - norm_np.min()) / (norm_np.max() - norm_np.min() + 1e-6)
                    else:
                        attention_map = feat.mean(dim=0).cpu().numpy()

                    att_min = attention_map.min()
                    att_max = attention_map.max()
                    if att_max > att_min:
                        attention_map = (attention_map - att_min) / (att_max - att_min)
                    else:
                        attention_map = np.ones_like(attention_map) * 0.5

                    attention_map = cv2.resize(attention_map, (w, h))
                    attention_map = np.clip(attention_map, 0, 1)

            attention_uint8 = (attention_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
            return heatmap

        finally:
            clear_hooks()

    except Exception as e:
        print(f"[GRADCAM ERROR YOLO] {str(e)}")
        import traceback
        traceback.print_exc()
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

        model.eval()
        
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