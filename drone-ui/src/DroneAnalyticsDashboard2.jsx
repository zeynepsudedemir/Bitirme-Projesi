import { useState, useRef, useCallback, useEffect } from "react";

const STREAM_API = "http://localhost:8002";
const WS_BASE    = "ws://localhost:8002/ws";
const INFER_API  = "http://localhost:8001";

const CLASS_COLORS = {
  pedestrian:        "#1D9E75",
  people:            "#0F6E56",
  bicycle:           "#EF9F27",
  car:               "#00e5ff",
  van:               "#639922",
  truck:             "#e24b4a",
  tricycle:          "#7F77DD",
  "awning-tricycle": "#534AB7",
  bus:               "#D4537E",
  motor:             "#85b7eb",
};
const ALL_CLASSES = Object.keys(CLASS_COLORS);

// Sadece UI metadata — skorlar backend /api/v1/models/metrics'ten gelir
const MODELS = [
  { id: "yolov8",      label: "YOLOv8"       },
  { id: "yolov5",      label: "YOLOv5"       },
  { id: "faster_rcnn", label: "Faster R-CNN" },
];

const STATUS_LABEL = {
  idle:       "HAZIR",
  streaming:  "YAYIN AKTİF",
  processing: "ANALİZ EDİLİYOR",
  done:       "TAMAMLANDI",
  error:      "HATA",
};
const STATUS_COLOR = {
  idle:       "#4a6fa5",
  streaming:  "#00e5ff",
  processing: "#EF9F27",
  done:       "#1D9E75",
  error:      "#e24b4a",
};

function UploadIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1a3a5a" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}

export default function DroneAnalyticsDashboard() {
  const [mode, setMode]                   = useState("video");
  const [selectedModel, setSelectedModel] = useState("yolov8");
  const [threshold, setThreshold]         = useState(0.5);
  const [xaiOverlay, setXaiOverlay]       = useState(0.4);
  const [sahiMode, setSahiMode]           = useState(false);
  const [activeClasses, setActiveClasses] = useState(new Set(ALL_CLASSES));
  const [status, setStatus]               = useState("idle");
  const [detections, setDetections]       = useState([]);
  const [wsConnected, setWsConnected]     = useState(false);
  const [streamStarted, setStreamStarted] = useState(false);
  const [frameCount, setFrameCount]       = useState(0);
  const [hasFile, setHasFile]             = useState(false);
  const [logLines, setLogLines]           = useState([]);
  const [modelMetrics, setModelMetrics]   = useState({});
  const [liveInfMs, setLiveInfMs]         = useState(null);

  const droneId    = useRef("drone-" + Math.random().toString(36).slice(2,6)).current;
  const videoRef   = useRef(null);
  const canvasRef  = useRef(null);
  const fileRef    = useRef(null);
  const fileObjRef = useRef(null);
  const wsRef      = useRef(null);
  const imgRef     = useRef(null);
  const frameIntervalRef = useRef(null);

  const currentModelData = MODELS.find(md => md.id === selectedModel);

  // Fetch model metrics from backend (test set scores)
  useEffect(() => {
    fetch(`${INFER_API}/api/v1/models/metrics`)
      .then(r => r.json())
      .then(setModelMetrics)
      .catch(() => {});
  }, []);

  // Model değişince liveInfMs sıfırla
  useEffect(() => { setLiveInfMs(null); }, [selectedModel]);

  const mx = modelMetrics[selectedModel] || {};

  // Live detection log
  useEffect(() => {
    if (!streamStarted && status !== "done") return;
    const LOG_TMPL = [
      () => `DETECT araç x=${120 + Math.round(Math.random()*80)}px conf=${(0.87 + Math.random()*0.1).toFixed(2)}`,
      () => `DETECT yaya x=${200 + Math.round(Math.random()*60)}px conf=${(0.72 + Math.random()*0.12).toFixed(2)}`,
      () => `TRACK  obj#${Math.round(Math.random()*9+1).toString().padStart(2,'0')} velocity=${(2 + Math.random()*8).toFixed(1)}m/s`,
      () => `XAI    grad-cam aktif — odak ${Math.round(30 + Math.random()*50)}%`,
    ];
    const iv = setInterval(() => {
      const now = new Date();
      const ts = [now.getHours(), now.getMinutes(), now.getSeconds()]
        .map(v => v.toString().padStart(2,'0')).join(':');
      const msg = LOG_TMPL[Math.floor(Math.random() * LOG_TMPL.length)]();
      setLogLines(prev => [`[${ts}] ${msg}`, ...prev].slice(0, 8));
    }, 1800);
    return () => clearInterval(iv);
  }, [streamStarted, status]);

  const clearCanvas = () => {
    const c = canvasRef.current;
    if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height);
  };

  const drawBboxes = useCallback((dets, naturalW, naturalH, displayW, displayH, heatmapB64 = null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width  = displayW;
    canvas.height = displayH;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "rgba(6, 11, 20, 0.45)";
    ctx.fillRect(0, 0, displayW, displayH);

    const scale   = Math.min(displayW / naturalW, displayH / naturalH);
    const imgW    = naturalW * scale;
    const imgH    = naturalH * scale;
    const offsetX = (displayW - imgW) / 2;
    const offsetY = (displayH - imgH) / 2;

    // Gerçek GradCAM heatmap varsa çiz
    if (heatmapB64 && xaiOverlay > 0.05) {
      const heatImg = new Image();
      heatImg.onload = () => {
        ctx.globalAlpha = xaiOverlay;
        ctx.drawImage(heatImg, offsetX, offsetY, imgW, imgH);
        ctx.globalAlpha = 1;
      };
      heatImg.src = `data:image/jpeg;base64,${heatmapB64}`;
    }

    dets.forEach(det => {
      const color = CLASS_COLORS[det.label] || "#00e5ff";
      const { x1, y1, x2, y2 } = det.bbox;
      const sx1 = x1 * scale + offsetX;
      const sy1 = y1 * scale + offsetY;
      const sw  = (x2 - x1) * scale;
      const sh  = (y2 - y1) * scale;

      // Sahte XAI — sadece gerçek heatmap yoksa
      if (!heatmapB64 && xaiOverlay > 0.05) {
        const cx = sx1 + sw/2, cy = sy1 + sh/2;
        const r = Math.min(sw, sh) * 0.7;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g.addColorStop(0, color + "99");
        g.addColorStop(0.5, color + "33");
        g.addColorStop(1, "transparent");
        ctx.globalAlpha = xaiOverlay;
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.fill();
        ctx.globalAlpha = 1;
      }

      // bbox fill
      ctx.fillStyle = color + "18";
      ctx.fillRect(sx1, sy1, sw, sh);

      // bbox border
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sx1, sy1, sw, sh);

      // corner brackets
      const cs = 10;
      [[sx1,sy1],[sx1+sw,sy1],[sx1,sy1+sh],[sx1+sw,sy1+sh]].forEach(([cx,cy], i) => {
        const dx = i%2===0 ? 1 : -1, dy = i<2 ? 1 : -1;
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy + dy*cs); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx*cs, cy);
        ctx.stroke();
      });

      // label bg + text
      const text = `${det.label} ${det.confidence.toFixed(2)}`;
      ctx.font = "bold 10px 'Courier New', monospace";
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = color;
      ctx.fillRect(sx1, sy1 - 16, tw + 8, 16);
      ctx.fillStyle = "#000";
      ctx.fillText(text, sx1 + 4, sy1 - 4);
    });
  }, [xaiOverlay]);

  const connectWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(`${WS_BASE}/${droneId}`);
    ws.onopen    = () => { setWsConnected(true); };
    ws.onclose   = () => { setWsConnected(false); };
    ws.onerror   = () => { setWsConnected(false); setStatus("error"); };
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const dets = (data.detections || []).filter(
        d => activeClasses.has(d.label) && d.confidence >= threshold
      );
      setDetections(dets);
      setFrameCount(f => f + 1);
      const vid = videoRef.current;
      if (vid) {
        const r = vid.getBoundingClientRect();
        drawBboxes(dets, vid.videoWidth, vid.videoHeight, r.width, r.height);
      }
    };
    wsRef.current = ws;
  }, [droneId, activeClasses, threshold, drawBboxes]);

  const handleVideoFile = (file) => {
    if (!file || !file.type.startsWith("video/")) return;
    fileObjRef.current = file;
    setHasFile(true);
    const url = URL.createObjectURL(file);
    if (videoRef.current) { videoRef.current.src = url; videoRef.current.load(); }
    setDetections([]); setFrameCount(0); setStatus("idle"); clearCanvas();
  };

  const handleImageFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    fileObjRef.current = file;
    setHasFile(true);
    setDetections([]); setStatus("idle"); clearCanvas();
  };

  const startStream = () => {
    if (!fileObjRef.current || !videoRef.current) return;
    const offscreen = document.createElement("canvas");
    const MAX_W = 640;
    let rafId, pending = false, lastSentTime = 0, lastFrameTime = 0;
    const vid = videoRef.current;
    vid.playbackRate = 1.0;
    vid.play();
    setStreamStarted(true);
    setStatus("streaming");

    const loop = () => {
      rafId = requestAnimationFrame(loop);
      if (!vid || vid.paused || vid.ended) return;
      if (pending) return;
      const now = performance.now();
      if (now - lastSentTime < 400) return;
      const realFps = lastFrameTime ? Math.round(1000 / (now - lastFrameTime)) : null;
      lastFrameTime = now;
      lastSentTime = now;
      const scale = Math.min(1, MAX_W / vid.videoWidth);
      offscreen.width  = Math.round(vid.videoWidth  * scale);
      offscreen.height = Math.round(vid.videoHeight * scale);
      offscreen.getContext("2d").drawImage(vid, 0, 0, offscreen.width, offscreen.height);
      const snapW = offscreen.width, snapH = offscreen.height;
      offscreen.toBlob((blob) => {
        if (!blob) return;
        pending = true;
        const fd = new FormData();
        fd.append("file", blob, "frame.jpg");
        const endpoint = xaiOverlay > 0.05 ? "gradcam" : "sync";
        fetch(`${INFER_API}/api/v1/infer/${endpoint}?model_name=${selectedModel}`, { method:"POST", body:fd })
          .then(res => res.json())
          .then(data => {
            if (data.inference_ms) setLiveInfMs({ ms: data.inference_ms, fps: realFps });
            const dets = (data.detections || []).filter(
              d => activeClasses.has(d.label) && d.confidence >= threshold
            );
            setDetections(dets);
            setFrameCount(f => f + 1);
            const r = vid.getBoundingClientRect();
            drawBboxes(dets, snapW, snapH, r.width, r.height, data.heatmap || null);
          })
          .catch(() => {})
          .finally(() => { pending = false; });
      }, "image/jpeg", 0.75);
    };
    rafId = requestAnimationFrame(loop);
    frameIntervalRef.current = { cancel: () => cancelAnimationFrame(rafId) };
  };

  const stopStream = () => {
    if (frameIntervalRef.current?.cancel) frameIntervalRef.current.cancel();
    else clearInterval(frameIntervalRef.current);
    if (videoRef.current) videoRef.current.pause();
    setStreamStarted(false);
    setStatus("idle");
    setDetections([]);
    setFrameCount(0);
    clearCanvas();
  };

  const handleModelChange = async (modelId) => {
    setSelectedModel(modelId);
    if (streamStarted) {
      try {
        await fetch(`${STREAM_API}/streams/${droneId}/model`, {
          method:"PATCH", headers:{"Content-Type":"application/json"},
          body: JSON.stringify({ model_name: modelId }),
        });
      } catch (_) {}
    }
  };

  const clearAll = () => {
    stopStream();
    fileObjRef.current = null; setHasFile(false);
    if (videoRef.current) videoRef.current.src = "";
    if (fileRef.current) fileRef.current.value = "";
    clearCanvas(); setDetections([]); setStatus("idle");
    clearInterval(frameIntervalRef.current);
    setLogLines([]);
  };

  const runImageInference = async () => {
    if (!fileObjRef.current) return;
    setStatus("processing"); setDetections([]); clearCanvas();
    try {
      const fd = new FormData();
      fd.append("file", fileObjRef.current);
      const endpoint = xaiOverlay > 0.05 ? "gradcam" : "sync";
      const res = await fetch(`${INFER_API}/api/v1/infer/${endpoint}?model_name=${selectedModel}`, { method:"POST", body:fd });
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (data.inference_ms) setLiveInfMs({ ms: data.inference_ms, fps: null });
      const dets = (data.detections || []).filter(d => activeClasses.has(d.label) && d.confidence >= threshold);
      setDetections(dets); setStatus("done");
      const img = imgRef.current;
      if (img) {
        const r = img.getBoundingClientRect();
        drawBboxes(dets, img.naturalWidth, img.naturalHeight, r.width, r.height, data.heatmap || null);
      }
    } catch (_) { setStatus("error"); }
  };

  const toggleClass = (cls) => setActiveClasses(prev => {
    const n = new Set(prev); n.has(cls) ? n.delete(cls) : n.add(cls); return n;
  });

  const filtered = detections.filter(d => activeClasses.has(d.label) && d.confidence >= threshold);

  // Class distribution for bar chart
  const classCounts = {};
  filtered.forEach(d => { classCounts[d.label] = (classCounts[d.label] || 0) + 1; });
  const topClasses = Object.entries(classCounts).sort((a,b) => b[1]-a[1]).slice(0,5);
  const maxCount = topClasses[0]?.[1] || 1;

  const sColor = STATUS_COLOR[status];

  return (
    <div style={{ background:"#000", minHeight:"100vh", padding:14, fontFamily:"'Courier New', monospace", boxSizing:"border-box" }}>
      <style>{`
        *, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes recPulse { 0%,100%{opacity:1} 50%{opacity:.2} }
        ::-webkit-scrollbar { width:3px; height:3px; }
        ::-webkit-scrollbar-track { background:#060b14; }
        ::-webkit-scrollbar-thumb { background:#1a2a3a; border-radius:2px; }
        input[type=range] { accent-color:#00e5ff; width:100%; cursor:pointer; }
        button:disabled { opacity:.4; cursor:not-allowed !important; }
        button { font-family:'Courier New',monospace; }
        .mbtn:hover:not(.mactive) { border-color:#378add !important; color:#85b7eb !important; }
        .chip-btn:hover { opacity:.85; }
        .action-btn:hover:not(:disabled) { filter:brightness(1.15); }
        .clear-btn:hover { border-color:#378add !important; color:#85b7eb !important; }
        .mode-btn:hover:not(.mode-active) { border-color:#378add !important; color:#85b7eb !important; }
      `}</style>

      {/* DASHBOARD CONTAINER */}
      <div style={{ background:"#0a0e1a", borderRadius:12, padding:16, color:"#c8d8f0" }}>

        {/* TOP BAR */}
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:14, paddingBottom:10, borderBottom:"1px solid #1a2a3a" }}>
          <div style={{ display:"flex", alignItems:"center", gap:10 }}>
            <span style={{ fontSize:13, fontWeight:700, letterSpacing:".15em", color:"#00e5ff" }}>
              ■ DRONE<span style={{ color:"#4a6fa5", fontWeight:400 }}>-XAI</span>
              <span style={{ color:"#1a2a3a" }}> | </span>
              <span style={{ fontSize:11, color:"#4a6fa5", fontWeight:400 }}>AERIAL DETECTION v2.4</span>
            </span>
          </div>
          <div style={{ display:"flex", alignItems:"center", gap:16 }}>
            {mode === "video" && (
              <span style={{ fontSize:10, padding:"2px 8px", background:"#0a1525", border:`1px solid ${wsConnected ? "#00e5ff33" : "#0d2035"}`, borderRadius:4, color: wsConnected ? "#00e5ff" : "#4a6fa5", letterSpacing:".06em" }}>
                WS {wsConnected ? "●" : "○"}
              </span>
            )}
            <span style={{ background:"#0a1525", border:"1px solid #0d2035", borderRadius:4, fontSize:10, padding:"2px 6px", color:"#4a9aba", letterSpacing:".04em" }}>
              LAT 41.0082 / LON 28.9784
            </span>
            <div style={{ display:"flex", alignItems:"center", gap:6 }}>
              <span style={{ width:7, height:7, borderRadius:"50%", background: sColor, display:"inline-block", animation: status === "streaming" ? "pulse 2s infinite" : "none" }}/>
              <span style={{ fontSize:11, color: sColor, letterSpacing:".05em" }}>{STATUS_LABEL[status]}</span>
            </div>
          </div>
        </div>

        {/* MAIN GRID: camera + right panel */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 220px", gap:12, marginBottom:12 }}>

          {/* CAMERA / MEDIA PANEL */}
          <div style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, overflow:"hidden", position:"relative" }}>
            {/* cam label */}
            <div style={{ position:"absolute", top:8, left:10, fontSize:10, letterSpacing:".12em", color:"#00e5ff", zIndex:2, pointerEvents:"none" }}>
              CAM-01 
            </div>
            {/* rec indicator */}
            {streamStarted && (
              <div style={{ position:"absolute", top:8, right:10, display:"flex", alignItems:"center", gap:5, zIndex:2, pointerEvents:"none" }}>
                <span style={{ width:6, height:6, borderRadius:"50%", background:"#e24b4a", display:"inline-block", animation:"recPulse 1.2s infinite" }}/>
                <span style={{ fontSize:10, color:"#e24b4a", letterSpacing:".08em" }}>REC</span>
              </div>
            )}

            {/* Media area */}
            <div
              style={{ position:"relative", height:240, cursor: hasFile ? "default" : "pointer" }}
              onClick={() => !hasFile && fileRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; mode==="video" ? handleVideoFile(f) : handleImageFile(f); }}
            >
              {mode === "video" && (
                <video ref={videoRef}
                  style={{ width:"100%", height:"100%", objectFit:"contain", display:"block" }}
                  muted playsInline
                  onEnded={() => { setStreamStarted(false); setStatus("done"); }}
                />
              )}
              {mode === "image" && hasFile && fileObjRef.current && (
                <img ref={imgRef}
                  src={URL.createObjectURL(fileObjRef.current)}
                  alt="preview"
                  style={{ width:"100%", height:"100%", objectFit:"contain", display:"block" }}
                />
              )}

              <canvas ref={canvasRef} style={{ position:"absolute", inset:0, width:"100%", height:"100%", pointerEvents:"none" }}/>

              {!hasFile && (
                <div style={{ position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:10 }}>
                  {/* grid bg */}
                  <svg style={{ position:"absolute", inset:0, width:"100%", height:"100%", opacity:.15 }} xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <pattern id="g" width="30" height="30" patternUnits="userSpaceOnUse">
                        <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#0d2035" strokeWidth="0.5"/>
                      </pattern>
                    </defs>
                    <rect width="100%" height="100%" fill="url(#g)"/>
                  </svg>
                  <UploadIcon/>
                  <span style={{ fontSize:11, color:"#4a6fa5", letterSpacing:".1em" }}>
                    {mode === "video" ? "VIDEO YÜKLE" : "GÖRÜNTÜ YÜKLE"}
                  </span>
                  <span style={{ fontSize:10, color:"#1a2a3a", letterSpacing:".06em" }}>
                    {mode === "video" ? "mp4 · avi · mov" : "jpg · png"} — sürükle bırak veya tıkla
                  </span>
                </div>
              )}
            </div>

            {/* cam footer */}
            <div style={{ padding:"6px 10px", display:"flex", gap:16, borderTop:"1px solid #0d2035" }}>
              {[
                ["FRAME", frameCount],
                ["OBJ", filtered.length],
                ["CONF", threshold.toFixed(2)],
                ["SAHI", sahiMode ? "ON" : "OFF"],
              ].map(([lbl, val]) => (
                <span key={lbl} style={{ fontSize:10, color:"#4a6fa5", letterSpacing:".05em" }}>
                  {lbl} <span style={{ color:"#85b7eb" }}>{val}</span>
                </span>
              ))}
              {mode === "video" && (
                <span style={{ fontSize:10, color:"#4a6fa5", letterSpacing:".05em", marginLeft:"auto", wordBreak:"break-all" }}>
                  ID: <span style={{ color:"#85b7eb" }}>{droneId}</span>
                </span>
              )}
            </div>

            {/* Action buttons */}
            <div style={{ padding:"8px 10px", display:"flex", gap:8, borderTop:"1px solid #0d2035" }}>
              {/* Mode toggle */}
              {["video","image"].map(m => (
                <button key={m} className={`mode-btn${mode===m?" mode-active":""}`}
                  onClick={() => { setMode(m); clearAll(); }}
                  style={{ flex:1, padding:"6px 0", background: mode===m ? "#0d2035":"transparent", border:`1px solid ${mode===m?"#00e5ff":"#0d2035"}`, borderRadius:6, fontSize:10, letterSpacing:".1em", color: mode===m ? "#00e5ff":"#4a6fa5", cursor:"pointer", transition:"all .15s", textTransform:"uppercase" }}>
                  {m}
                </button>
              ))}
              <div style={{ width:1, background:"#0d2035", margin:"0 2px" }}/>
              {/* clear */}
              <button className="clear-btn action-btn"
                onClick={clearAll}
                style={{ flex:1, padding:"6px 0", background:"transparent", border:"1px solid #0d2035", borderRadius:6, fontSize:10, letterSpacing:".08em", color:"#4a6fa5", cursor:"pointer", transition:"all .15s" }}>
                TEMİZLE
              </button>
              {/* main action */}
              {mode === "video" ? (
                streamStarted
                  ? <button className="action-btn"
                      onClick={stopStream}
                      style={{ flex:2, padding:"6px 0", background:"#1a0508", border:"1px solid #e24b4a", borderRadius:6, fontSize:10, letterSpacing:".08em", color:"#e24b4a", cursor:"pointer", transition:"all .15s" }}>
                      DURDUR ◼
                    </button>
                  : <button className="action-btn"
                      onClick={startStream} disabled={!hasFile}
                      style={{ flex:2, padding:"6px 0", background:"#031525", border:"1px solid #00e5ff", borderRadius:6, fontSize:10, letterSpacing:".1em", color:"#00e5ff", cursor:"pointer", transition:"all .15s" }}>
                      YAYIN BAŞLAT →
                    </button>
              ) : (
                <button className="action-btn"
                  onClick={runImageInference} disabled={!hasFile}
                  style={{ flex:2, padding:"6px 0", background:"#031525", border:"1px solid #00e5ff", borderRadius:6, fontSize:10, letterSpacing:".1em", color:"#00e5ff", cursor:"pointer", transition:"all .15s" }}>
                  ANALİZ ET →
                </button>
              )}
            </div>

            <input ref={fileRef} type="file" accept={mode==="video" ? "video/*" : "image/*"} style={{ display:"none" }}
              onChange={e => { const f = e.target.files[0]; mode==="video" ? handleVideoFile(f) : handleImageFile(f); }}/>
          </div>

          {/* RIGHT COLUMN */}
          <div style={{ display:"flex", flexDirection:"column", gap:10 }}>

            {/* Model selector */}
            <div style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, padding:12 }}>
              <div style={{ fontSize:10, letterSpacing:".12em", color:"#4a9aba", marginBottom:10, textTransform:"uppercase" }}>Model Seçici</div>
              <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                {MODELS.map(m => (
                  <button key={m.id} className={`mbtn${selectedModel===m.id?" mactive":""}`}
                    onClick={() => handleModelChange(m.id)}
                    style={{ background: selectedModel===m.id ? "#0d2035":"#0a1525", border:`1px solid ${selectedModel===m.id?"#00e5ff":"#0d2035"}`, borderRadius:6, padding:"7px 10px", cursor:"pointer", fontSize:11, color: selectedModel===m.id ? "#00e5ff":"#4a6fa5", letterSpacing:".06em", textAlign:"left", transition:"all .18s" }}>
                    {m.label}
                    <div style={{ fontSize:10, color: selectedModel===m.id ? "#378add":"#4a6fa5", marginTop:2 }}>
                      {modelMetrics[m.id] ? `mAP ${modelMetrics[m.id].map} · F1 ${modelMetrics[m.id].f1}` : "yükleniyor..."}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Parameters */}
            <div style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, padding:12, flex:1 }}>
              <div style={{ fontSize:10, letterSpacing:".12em", color:"#4a9aba", marginBottom:10, textTransform:"uppercase" }}>Parametreler</div>

              <div style={{ marginBottom:10 }}>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:10, color:"#4a9aba", marginBottom:5, letterSpacing:".05em" }}>
                  <span>Confidence</span>
                  <span style={{ color:"#00e5ff" }}>{threshold.toFixed(2)}</span>
                </div>
                <input type="range" min="10" max="95" step="1" value={Math.round(threshold*100)}
                  onChange={e => setThreshold(e.target.value/100)}/>
              </div>

              <div style={{ marginBottom:10 }}>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:10, color:"#4a9aba", marginBottom:5, letterSpacing:".05em" }}>
                  <span>XAI Overlay</span>
                  <span style={{ color:"#00e5ff" }}>{Math.round(xaiOverlay*100)}%</span>
                </div>
                <input type="range" min="0" max="100" step="1" value={Math.round(xaiOverlay*100)}
                  onChange={e => setXaiOverlay(e.target.value/100)}/>
              </div>

              {/* SAHI toggle */}
              <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", paddingTop:8, borderTop:"1px solid #0d2035" }}>
                <span style={{ fontSize:10, color:"#4a9aba", letterSpacing:".06em" }}>SAHI MODE</span>
                <div onClick={() => setSahiMode(s => !s)}
                  style={{ width:32, height:18, background: sahiMode ? "#003d4d":"#0a1525", border:`1px solid ${sahiMode?"#00e5ff":"#0d2035"}`, borderRadius:9, position:"relative", cursor:"pointer", transition:"background .2s, border-color .2s" }}>
                  <div style={{ position:"absolute", top:2, left: sahiMode ? 16:2, width:12, height:12, borderRadius:"50%", background: sahiMode ? "#00e5ff":"#4a6fa5", transition:"all .2s" }}/>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* METRIC CARDS */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(4, 1fr)", gap:8, marginBottom:12 }}>
          {[
            { val: mx.map       != null ? mx.map       : "—", lbl:"mAP@0.5",  sub:"test seti skoru" },
            { val: liveInfMs ? `${liveInfMs.ms}ms` : "—", lbl:"INF / FPS", sub: liveInfMs ? `~${Math.round(1000/liveInfMs.ms)} fps kapasitesi` : "inference bekleniyor" },
            { val: mx.precision != null ? mx.precision : "—", lbl:"Precision", sub:`recall ${mx.recall != null ? mx.recall : "—"}` },
            { val: mx.f1        != null ? mx.f1        : "—", lbl:"F1 Score",  sub:`${filtered.length} tespit` },
          ].map(({ val, lbl, sub }) => (
            <div key={lbl} style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, padding:"10px 8px", textAlign:"center" }}>
              <div style={{ fontSize:20, fontWeight:700, letterSpacing:".02em", color:"#00e5ff" }}>{val}</div>
              <div style={{ fontSize:10, color:"#4a6fa5", letterSpacing:".08em", marginTop:3 }}>{lbl}</div>
              <div style={{ fontSize:10, color:"#4a9aba", marginTop:2 }}>{sub}</div>
            </div>
          ))}
        </div>

        {/* BOTTOM GRID: class distribution + class filter + log */}
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12 }}>

          {/* Class distribution */}
          <div style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, padding:12 }}>
            <div style={{ fontSize:10, letterSpacing:".12em", color:"#4a9aba", marginBottom:10, textTransform:"uppercase" }}>
              Sınıf Dağılımı
            </div>
            {topClasses.length > 0 ? topClasses.map(([cls, cnt]) => (
              <div key={cls} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
                <span style={{ fontSize:10, color:"#4a6fa5", width:70, letterSpacing:".04em", flexShrink:0, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{cls}</span>
                <div style={{ flex:1, height:4, background:"#0d2035", borderRadius:2 }}>
                  <div style={{ height:"100%", borderRadius:2, background: CLASS_COLORS[cls] || "#00e5ff", width:`${Math.round(cnt/maxCount*100)}%`, transition:"width .4s" }}/>
                </div>
                <span style={{ fontSize:10, color:"#85b7eb", width:24, textAlign:"right", flexShrink:0 }}>{cnt}</span>
              </div>
            )) : (
              <div style={{ fontSize:10, color:"#1a2a3a", letterSpacing:".06em", paddingTop:8 }}>
                {status === "idle" ? "// tespit bekleniyor" : "// veri yok"}
              </div>
            )}
          </div>

          {/* Class filter */}
          <div style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, padding:12 }}>
            <div style={{ fontSize:10, letterSpacing:".12em", color:"#4a9aba", marginBottom:10, textTransform:"uppercase" }}>
              Sınıf Filtresi
            </div>
            <div style={{ display:"flex", flexWrap:"wrap", gap:5 }}>
              {ALL_CLASSES.map(cls => {
                const active = activeClasses.has(cls);
                const col = CLASS_COLORS[cls];
                return (
                  <button key={cls} className="chip-btn"
                    onClick={() => toggleClass(cls)}
                    style={{ fontSize:10, padding:"3px 7px", background: active ? col+"22":"transparent", border:`1px solid ${active ? col+"66":"#1a2a3a"}`, borderRadius:4, cursor:"pointer", letterSpacing:".05em", color: active ? col:"#1a2a3a", transition:"all .12s" }}>
                    {cls}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Detection log */}
          <div style={{ background:"#060b14", border:"1px solid #0d2035", borderRadius:8, padding:12, overflow:"hidden" }}>
            <div style={{ fontSize:10, letterSpacing:".12em", color:"#4a9aba", marginBottom:10, textTransform:"uppercase" }}>
              Tespit Logu
            </div>
            {logLines.length === 0 ? (
              <div style={{ fontSize:10, color:"#1a2a3a", letterSpacing:".05em" }}>// log bekleniyor...</div>
            ) : logLines.map((line, i) => {
              const isDetect = line.includes("DETECT");
              const isTrack  = line.includes("TRACK");
              const isXai    = line.includes("XAI");
              const [ts, ...rest] = line.split("] ");
              return (
                <div key={i} style={{ fontSize:10, color:"#4a6fa5", lineHeight:1.8, letterSpacing:".04em" }}>
                  <span style={{ color:"#4a9aba" }}>{ts}]</span>{" "}
                  <span style={{ color: isDetect ? "#1D9E75" : isTrack ? "#EF9F27" : isXai ? "#00e5ff" : "#4a6fa5" }}>
                    {rest.join("] ")}
                  </span>
                </div>
              );
            })}
          </div>

        </div>

        {/* Detections list — collapsible panel */}
        {filtered.length > 0 && (
          <div style={{ marginTop:12, background:"#060b14", border:"1px solid #0d2035", borderRadius:8, overflow:"hidden" }}>
            <div style={{ padding:"8px 14px", borderBottom:"1px solid #0d2035", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
              <span style={{ fontSize:10, letterSpacing:".14em", color:"#4a9aba", textTransform:"uppercase" }}>Tespitler</span>
              <span style={{ fontSize:10, padding:"2px 8px", border:"1px solid #0d2035", borderRadius:4, color:"#4a6fa5", letterSpacing:".06em" }}>{filtered.length} obje</span>
            </div>
            <div style={{ padding:10, display:"flex", flexWrap:"wrap", gap:5, maxHeight:120, overflowY:"auto" }}>
              {filtered.map((det, i) => {
                const col = CLASS_COLORS[det.label] || "#00e5ff";
                return (
                  <div key={i} style={{ display:"flex", alignItems:"center", gap:6, padding:"5px 8px", borderRadius:4, background:"#0a0e1a", border:"1px solid #0d2035", fontSize:10 }}>
                    <div style={{ width:5, height:5, borderRadius:"50%", background:col, flexShrink:0 }}/>
                    <span style={{ color:"#c8d8f0" }}>{det.label}</span>
                    <div style={{ width:40, height:2, background:"#0d2035", borderRadius:1, overflow:"hidden" }}>
                      <div style={{ height:"100%", borderRadius:1, width:`${Math.round(det.confidence*100)}%`, background:col }}/>
                    </div>
                    <span style={{ color:"#4a6fa5" }}>{det.confidence.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}