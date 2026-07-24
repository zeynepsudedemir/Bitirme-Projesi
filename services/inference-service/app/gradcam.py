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
    act: (C, H, W) — aktivasyonlar
    grad: (C, H, W) — gradyanlar
    """
    try:
        if act.dim() != 3:
            act = act.squeeze()
        if grad.dim() != 3:
            grad = grad.squeeze()

        min_channels = min(act.shape[0], grad.shape[0])
        act = act[:min_channels]
        grad = grad[:min_channels]

        weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = F.relu((weights * act).sum(dim=0))       # (H, W)

        cam_np = cam.detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        return cam_np
    except Exception as e:
        print(f"[GRADCAM COMPUTE ERROR] {e}")
        return None


def _extract_first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _extract_first_tensor(item)
            if isinstance(tensor, torch.Tensor):
                return tensor
    if isinstance(output, dict):
        for v in output.values():
            tensor = _extract_first_tensor(v)
            if isinstance(tensor, torch.Tensor):
                return tensor
    return None


#YOLO için hook önbellek

_yolo_hook_positions_cache: dict = {}


def _get_yolo_hook_positions(backbone_model) -> list[int]:
    model_id = id(backbone_model)
    if model_id in _yolo_hook_positions_cache:
        return _yolo_hook_positions_cache[model_id]

    total_layers = len(backbone_model)
    if total_layers >= 3:
        positions = [
            max(0, total_layers - 5),
            max(0, total_layers - 4),
            max(0, total_layers - 3),
        ]
    else:
        positions = list(range(total_layers))

    _yolo_hook_positions_cache[model_id] = positions
    return positions


def gradcam_yolo(model, frame_bgr: np.ndarray, detections: list = None,
                 enable_gradcam: bool = False, top_k: int = 1, max_side: int = 960) -> np.ndarray:
    """
    YOLOv8/v5 GradCAM — standard GradCAM activation-gradient attention map.
    Hook pozisyonları önbelleklenir, her çağrıda model parse edilmez.
    """
    h, w = frame_bgr.shape[:2]

    try:
        #backbone
        backbone_model = None
        if hasattr(model, "model"):
            backbone_model = model.model.model if hasattr(model.model, "model") else model.model

        if backbone_model is None or not hasattr(backbone_model, "__len__"):
            print("[GRADCAM YOLO] Model yapısı beklenmedik")
            return np.zeros((h, w, 3), dtype=np.uint8)

        hook_positions = _get_yolo_hook_positions(backbone_model)

        #hook
        features_multi = []
        hook_handles = []
        grads = {}

        use_real_gradcam = bool(enable_gradcam)
        proc_frame = frame_bgr
        if max_side is not None:
            h0, w0 = frame_bgr.shape[:2]
            scale = min(1.0, float(max_side) / max(h0, w0))
            if scale < 1.0:
                proc_frame = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

        print(f"[GRADCAM YOLO] enabled={enable_gradcam} detections={bool(detections)} in_shape={frame_bgr.shape} proc_shape={proc_frame.shape}")

        def hook_fn_factory(layer_idx):
            def hook_fn(module, input, output):
                tensor = _extract_first_tensor(output)
                if tensor is None:
                    return

                features_multi.append({"layer": layer_idx, "feat": tensor})

                if tensor.requires_grad:
                    def _record_grad(grad, idx=layer_idx):
                        grads[idx] = grad.detach()
                    try:
                        tensor.register_hook(_record_grad)
                    except Exception:
                        pass
            return hook_fn

        for pos in hook_positions:
            try:
                hook_handles.append(backbone_model[pos].register_forward_hook(hook_fn_factory(pos)))
            except Exception:
                continue

        def clear_hooks():
            for h_ in hook_handles:
                try:
                    h_.remove()
                except Exception:
                    pass

        try:
            
            use_half = next(model.parameters()).dtype == torch.float16
            device = next(model.parameters()).device
            proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(proc_frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            if use_half:
                img_tensor = img_tensor.half()
            img_tensor.requires_grad_(use_real_gradcam)

            if use_real_gradcam:
                try:
                    outputs = model.model(img_tensor)
                    S = None

                    if isinstance(outputs, tuple) and len(outputs) > 1 and isinstance(outputs[1], dict):
                        scores = outputs[1].get("scores")
                        if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                            S = scores.view(-1).max()

                    if S is None and isinstance(outputs, torch.Tensor):
                        S = outputs.mean()
                    elif S is None:
                        first_tensor = _extract_first_tensor(outputs)
                        if isinstance(first_tensor, torch.Tensor):
                            S = first_tensor.mean()

                    if S is not None:
                        model.model.zero_grad(set_to_none=True)
                        S.backward(retain_graph=False)
                    else:
                        with torch.no_grad():
                            _ = model.predict(proc_frame, verbose=False, imgsz=640, conf=0.1, half=use_half)
                except Exception:
                    with torch.no_grad():
                        _ = model.predict(proc_frame, verbose=False, imgsz=640, conf=0.1, half=use_half)
            else:
                with torch.no_grad():
                    _ = model.predict(proc_frame, verbose=False, imgsz=640, conf=0.1, half=use_half)

            if not features_multi:
                clear_hooks()
                hook_handles.clear()
                total_layers = len(backbone_model)
                for pos in range(max(0, total_layers - 6), total_layers):
                    try:
                        hook_handles.append(backbone_model[pos].register_forward_hook(hook_fn_factory(pos)))
                    except Exception:
                        continue
                with torch.no_grad():
                    _ = model.predict(proc_frame, verbose=False, imgsz=640, conf=0.1, half=use_half)

            #Attention map hesabı
            if not features_multi:
                print("[GRADCAM YOLO] Feature yakalanamadı")
                attention_map = np.full((h, w), 0.5, dtype=np.float32)
            else:
                features_multi.sort(key=lambda x: x["layer"], reverse=True)
                best = features_multi[0]
                feat = best["feat"]
                grad = grads.get(best["layer"], None)

                if feat.dim() == 4:
                    feat = feat[0]
                if grad is not None and grad.dim() == 4:
                    grad = grad[0]
                elif grad is not None and grad.dim() == 5:
                    grad = grad[0]

                if feat.dim() != 3 or grad is None:
                    feat_norm = torch.norm(feat, p=2, dim=0).cpu().numpy()
                    mn, mx_ = feat_norm.min(), feat_norm.max()
                    if mx_ > mn:
                        attention_map = (feat_norm - mn) / (mx_ - mn + 1e-6)
                    else:
                        attention_map = feat.mean(dim=0).cpu().numpy()
                        att_mn, att_mx = attention_map.min(), attention_map.max()
                        if att_mx > att_mn:
                            attention_map = (attention_map - att_mn) / (att_mx - att_mn)
                else:
                    cam_np = _safe_gradcam_computation(feat, grad)
                    if cam_np is not None:
                        attention_map = cam_np
                    else:
                        attention_map = torch.norm(feat, p=2, dim=0).cpu().numpy()
                        mn, mx_ = attention_map.min(), attention_map.max()
                        if mx_ > mn:
                            attention_map = (attention_map - mn) / (mx_ - mn + 1e-6)
                        else:
                            attention_map = feat.mean(dim=0).cpu().numpy()
                            att_mn, att_mx = attention_map.min(), attention_map.max()
                            if att_mx > att_mn:
                                attention_map = (attention_map - att_mn) / (att_mx - att_mn)

                attention_map = cv2.resize(
                    attention_map.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
                attention_map = np.clip(attention_map, 0, 1)

            attention_uint8 = (attention_map * 255).astype(np.uint8)
            return cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)

        finally:
            clear_hooks()

    except Exception as e:
        print(f"[GRADCAM ERROR YOLO] {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((h, w, 3), dtype=np.uint8)


#Faster R-CNN backbone hook
_frcnn_hook_handle = None
_frcnn_features: list = []


def _ensure_frcnn_hook(model) -> None:
    """Model backbone'una kalıcı hook tak — sadece bir kez."""
    global _frcnn_hook_handle

    if _frcnn_hook_handle is not None:
        return  # zaten takılı

    def hook_fn(module, input, output):
        _frcnn_features.clear()
        if isinstance(output, torch.Tensor):
            _frcnn_features.append(output.detach())
        elif isinstance(output, dict):
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    _frcnn_features.append(v.detach())
                    break
        elif isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    _frcnn_features.append(item.detach())
                    break

    _frcnn_hook_handle = model.backbone.register_forward_hook(hook_fn)
    print("[GRADCAM FASTER_RCNN] Kalıcı backbone hook takıldı")


def gradcam_faster_rcnn(model, frame_bgr: np.ndarray, detections: list = None,
                        enable_gradcam: bool = False, top_k: int = 1, max_side: int = 960) -> np.ndarray:
    """
    Faster R-CNN GradCAM — FPN backbone activation haritası.
    Hook her çağrıda takılıp sökülmez; startup'ta bir kez register edilir.
    """
    device = next(model.parameters()).device
    h, w = frame_bgr.shape[:2]

    try:
        # Keep legacy persistent hook but also support one-shot gradcam via temporary hooks
        _ensure_frcnn_hook(model)

        # Decide whether to attempt real Grad-CAM: only if explicitly enabled
        use_real_gradcam = bool(enable_gradcam) and bool(detections)
        proc_frame = frame_bgr
        if max_side is not None:
            h0, w0 = frame_bgr.shape[:2]
            scale = min(1.0, float(max_side) / max(h0, w0))
            if scale < 1.0:
                proc_frame = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

        print(f"[GRADCAM FRCNN] enabled={enable_gradcam} detections={bool(detections)} in_shape={frame_bgr.shape} proc_shape={proc_frame.shape}")
        activations = {}
        grads = {}
        temp_handle = None

        def _temp_hook(module, input, output):
            tensor = None
            if isinstance(output, torch.Tensor):
                tensor = output
            elif isinstance(output, (list, tuple)) and output:
                for item in output:
                    if isinstance(item, torch.Tensor):
                        tensor = item
                        break
            elif isinstance(output, dict):
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break

            if tensor is None:
                return

            activations[0] = tensor

            def _record_grad(g):
                grads[0] = g.detach()

            try:
                tensor.register_hook(_record_grad)
            except Exception:
                pass

        if use_real_gradcam:
            try:
                temp_handle = model.backbone.register_forward_hook(_temp_hook)
            except Exception:
                temp_handle = None

        frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).to(device)

        model.eval()

        if use_real_gradcam:
            # run forward with grad tracking to get target score
            try:
                img_tensor.requires_grad_(True)
                outputs = model([img_tensor])

                # select a target score S: prefer highest detection score
                S = None
                try:
                    if isinstance(outputs, (list, tuple)) and outputs:
                        out0 = outputs[0]
                        if isinstance(out0, dict) and 'scores' in out0:
                            S = out0['scores'].max()
                        elif isinstance(out0, list) and out0 and 'scores' in out0[0]:
                            S = out0[0]['scores'].max()
                except Exception:
                    S = None

                if S is None:
                    # fallback to legacy no-grad path
                    with torch.no_grad():
                        _ = model([img_tensor])
                else:
                    model.zero_grad()
                    S.backward(retain_graph=True)
            except Exception:
                with torch.no_grad():
                    _ = model([img_tensor])
        else:
            with torch.no_grad():
                _ = model([img_tensor])

        # If real gradcam succeeded, try to use activations+grads
        if use_real_gradcam and activations and grads:
            try:
                act = activations[0]
                grad = grads.get(0, None)
                if grad is not None:
                    if act.dim() == 4:
                        act = act[0]
                    if grad.dim() == 4:
                        grad = grad[0]
                    cam_np = _safe_gradcam_computation(act, grad)
                    if cam_np is not None:
                        cam_resized = cv2.resize(cam_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                        return _apply_colormap(cam_resized)
            except Exception:
                pass

        # legacy fallback using persistent _frcnn_features

        if not _frcnn_features:
            print("[GRADCAM FASTER_RCNN] Feature yakalanamadı")
            return np.zeros((h, w, 3), dtype=np.uint8)

        feat = _frcnn_features[0]
        if feat.dim() == 4:
            feat = feat[0]  # batch boyutu kaldır

        if feat.dim() == 3:
            attention_map = feat.mean(dim=0).cpu().numpy()
        else:
            attention_map = feat.squeeze().cpu().numpy()

        att_min, att_max = attention_map.min(), attention_map.max()
        if att_max > att_min:
            attention_map = (attention_map - att_min) / (att_max - att_min)
        else:
            attention_map = np.full_like(attention_map, 0.5)

        attention_map_resized = cv2.resize(
            attention_map.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        attention_uint8 = (attention_map_resized * 255).astype(np.uint8)
        return cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)

    except Exception as e:
        print(f"[GRADCAM ERROR FASTER_RCNN] {e}")
        return np.zeros((h, w, 3), dtype=np.uint8)