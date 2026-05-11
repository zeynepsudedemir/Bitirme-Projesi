import { useState, useRef, useCallback, useEffect } from "react";

const STREAM_API  = "http://localhost:8002";
const WS_BASE     = "ws://localhost:8002/ws";
const INFER_API   = "http://localhost:8001";

const CLASS_COLORS = {
  pedestrian:        "#1D9E75",
  people:            "#0F6E56",
  bicycle:           "#BA7517",
  car:               "#378ADD",
  van:               "#639922",
  truck:             "#D85A30",
  tricycle:          "#7F77DD",
  "awning-tricycle": "#534AB7",
  bus:               "#D4537E",
  motor:             "#E24B4A",
};
const ALL_CLASSES = Object.keys(CLASS_COLORS);

const MODELS = [
  { id: "yolov8",      label: "YOLOv8",       tag: "v8", desc: "Hızlı · Genel amaç",     color: "#534AB7" },
  { id: "yolov5",      label: "YOLOv5",       tag: "v5", desc: "Dengeli · Stabil",        color: "#185FA5" },
  { id: "faster_rcnn", label: "Faster R-CNN", tag: "FR", desc: "Yüksek doğruluk · Yavaş", color: "#993C1D" },
];

const s = {
  root: { display:"grid", gridTemplateColumns:"272px 1fr", gridTemplateRows:"52px 1fr", height:"100vh", fontFamily:"'IBM Plex Mono','Courier New',monospace", background:"#0a0c0e", color:"#c8cdd4", overflow:"hidden" },
  topbar: { gridColumn:"1/-1", borderBottom:"1px solid #1e2328", display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 20px", background:"#0a0c0e" },
  topLeft: { display:"flex", alignItems:"center", gap:10 },
  logoMark: { width:22, height:22, border:"1px solid #1D9E75", borderRadius:3, display:"flex", alignItems:"center", justifyContent:"center" },
  logoInner: { width:8, height:8, background:"#1D9E75", borderRadius:1 },
  logoText: { fontSize:12, letterSpacing:"0.12em", color:"#8a9199" },
  statusPill: { fontSize:10, letterSpacing:"0.1em", padding:"3px 10px", borderRadius:2, border:"1px solid", background:"transparent" },
  sidebar: { borderRight:"1px solid #1e2328", padding:"16px 14px", display:"flex", flexDirection:"column", gap:20, overflowY:"auto", background:"#0d0f12" },
  sectionLabel: { fontSize:9, letterSpacing:"0.18em", color:"#4a5260", textTransform:"uppercase", marginBottom:8 },
  modelCard: { display:"flex", alignItems:"center", gap:10, padding:"9px 10px", border:"1px solid", borderRadius:4, cursor:"pointer", background:"transparent", marginBottom:5, width:"100%", textAlign:"left", fontFamily:"'IBM Plex Mono',monospace", transition:"all 0.15s" },
  modelTag: { width:30, height:30, borderRadius:3, display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, flexShrink:0 },
  modelName: { fontSize:12, color:"#c8cdd4" },
  modelDesc: { fontSize:10, color:"#4a5260", marginTop:1 },
  activeDot: { width:5, height:5, borderRadius:"50%", background:"#1D9E75", marginLeft:"auto", flexShrink:0 },
  sliderLabel: { display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 },
  sliderVal: { fontSize:12, color:"#1D9E75" },
  chips: { display:"flex", flexWrap:"wrap", gap:5 },
  chip: { fontSize:10, padding:"3px 8px", border:"1px solid", borderRadius:2, cursor:"pointer", letterSpacing:"0.06em", transition:"all 0.12s", background:"transparent", fontFamily:"'IBM Plex Mono',monospace" },
  main: { display:"flex", flexDirection:"column", gap:12, padding:16, overflow:"hidden" },
  videoWrap: { position:"relative", background:"#0d0f12", border:"1px solid #1e2328", borderRadius:6, overflow:"hidden", flexShrink:0, height:280 },
  overlayCanvas: { position:"absolute", inset:0, width:"100%", height:"100%", pointerEvents:"none" },
  dropOverlay: { position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:8, cursor:"pointer" },
  dropTitle: { fontSize:12, color:"#4a5260", letterSpacing:"0.06em" },
  dropSub: { fontSize:10, color:"#2a3038" },
  actionRow: { display:"flex", gap:8, flexShrink:0 },
  btn:        { flex:1, padding:"9px 0", border:"1px solid #1e2328", borderRadius:3, background:"transparent", color:"#6a7480", fontSize:11, letterSpacing:"0.08em", cursor:"pointer", fontFamily:"'IBM Plex Mono',monospace" },
  btnPrimary: { flex:2, padding:"9px 0", border:"1px solid #1D9E75", borderRadius:3, background:"#0a1f18", color:"#1D9E75", fontSize:11, letterSpacing:"0.1em", cursor:"pointer", fontFamily:"'IBM Plex Mono',monospace" },
  btnDanger:  { flex:2, padding:"9px 0", border:"1px solid #A32D2D", borderRadius:3, background:"#1a0a0a", color:"#E24B4A", fontSize:11, letterSpacing:"0.08em", cursor:"pointer", fontFamily:"'IBM Plex Mono',monospace" },
  resultsBox: { border:"1px solid #1e2328", borderRadius:6, display:"flex", flexDirection:"column", flex:1, minHeight:0, background:"#0d0f12", overflow:"hidden" },
  resultsHeader: { padding:"9px 14px", borderBottom:"1px solid #1e2328", display:"flex", alignItems:"center", justifyContent:"space-between", flexShrink:0 },
  resultsLabel: { fontSize:9, letterSpacing:"0.16em", color:"#4a5260", textTransform:"uppercase" },
  countTag: { fontSize:10, padding:"2px 8px", border:"1px solid #1e2328", borderRadius:2, color:"#4a5260", letterSpacing:"0.06em" },
  resultsList: { padding:10, display:"flex", flexDirection:"column", gap:5, overflowY:"auto", flex:1 },
  detRow: { display:"flex", alignItems:"center", gap:8, padding:"6px 10px", borderRadius:3, background:"#0a0c0e", border:"1px solid #1e2328", fontSize:11 },
  detDot: { width:6, height:6, borderRadius:"50%", flexShrink:0 },
  detLabel: { flex:1, color:"#c8cdd4" },
  detConf: { fontSize:10, color:"#4a5260" },
  barWrap: { width:50, height:2, background:"#1e2328", borderRadius:1, overflow:"hidden" },
  bar: { height:"100%", borderRadius:1 },
  emptyState: { padding:24, textAlign:"center", color:"#2a3038", fontSize:11, letterSpacing:"0.06em" },
  wsTag: { fontSize:9, padding:"2px 8px", borderRadius:2, border:"1px solid", letterSpacing:"0.08em" },
  modeRow: { display:"flex", gap:6, marginBottom:4 },
  modeBtn: { flex:1, padding:"6px 0", border:"1px solid #1e2328", borderRadius:3, fontSize:10, letterSpacing:"0.08em", cursor:"pointer", fontFamily:"'IBM Plex Mono',monospace", transition:"all 0.12s" },
};

const STATUS_PILL = {
  idle:       { color:"#4a5260", borderColor:"#1e2328" },
  streaming:  { color:"#1D9E75", borderColor:"#0F6E56" },
  processing: { color:"#BA7517", borderColor:"#633806" },
  done:       { color:"#378ADD", borderColor:"#185FA5" },
  error:      { color:"#E24B4A", borderColor:"#A32D2D" },
};
const STATUS_LABEL = {
  idle:       "// hazır",
  streaming:  "// yayın aktif",
  processing: "// analiz ediliyor",
  done:       "// tamamlandı",
  error:      "// hata",
};

function UploadIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2a3038" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
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
  const [activeClasses, setActiveClasses] = useState(new Set(ALL_CLASSES));
  const [status, setStatus]               = useState("idle");
  const [detections, setDetections]       = useState([]);
  const [wsConnected, setWsConnected]     = useState(false);
  const [streamStarted, setStreamStarted] = useState(false);
  const [frameCount, setFrameCount]       = useState(0);
  const [hasFile, setHasFile]             = useState(false);

  const droneId    = useRef("drone-" + Math.random().toString(36).slice(2,6)).current;
  const videoRef   = useRef(null);
  const canvasRef  = useRef(null);
  const fileRef    = useRef(null);
  const fileObjRef = useRef(null);
  const wsRef      = useRef(null);
  const imgRef     = useRef(null);
  const frameIntervalRef = useRef(null);

  const clearCanvas = () => {
    const c = canvasRef.current;
    if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height);
  };

// drawBboxes başına ekle — önceki frame'den kalan bbox'ları yarı saydam göster
  const drawBboxes = useCallback((dets, naturalW, naturalH, displayW, displayH) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width  = displayW;
    canvas.height = displayH;
    const ctx = canvas.getContext("2d");

    // Tamamen silmek yerine hafif karart — önceki frame izleri solar
    ctx.fillStyle = "rgba(10, 12, 14, 0.55)";
    ctx.fillRect(0, 0, displayW, displayH);

    const scale   = Math.min(displayW / naturalW, displayH / naturalH);
    const imgW    = naturalW * scale;
    const imgH    = naturalH * scale;
    const offsetX = (displayW - imgW) / 2;
    const offsetY = (displayH - imgH) / 2;

    dets.forEach(det => {
      const color = CLASS_COLORS[det.label] || "#888";
      const { x1, y1, x2, y2 } = det.bbox;
      const sx1 = x1 * scale + offsetX;
      const sy1 = y1 * scale + offsetY;
      const sw  = (x2 - x1) * scale;
      const sh  = (y2 - y1) * scale;

      ctx.strokeStyle = color;
      ctx.lineWidth   = 1.5;
      ctx.strokeRect(sx1, sy1, sw, sh);

      const text = `${det.label} ${det.confidence.toFixed(2)}`;
      ctx.font = "10px 'IBM Plex Mono', monospace";
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = color + "cc";
      ctx.fillRect(sx1, sy1 - 16, tw + 8, 16);
      ctx.fillStyle = "#0a0c0e";
      ctx.fillText(text, sx1 + 4, sy1 - 4);
    });
  }, []);

  // WebSocket bağlantısı
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
    let rafId;
    let pending = false;
    let lastSentTime = 0;

    const vid = videoRef.current;
    vid.playbackRate = 1.0; // normal hız
    vid.play();
    setStreamStarted(true);
    setStatus("streaming");

    const loop = () => {
      rafId = requestAnimationFrame(loop);
      if (!vid || vid.paused || vid.ended) return;
      if (pending) return; // önceki bitmeden gönderme

      const now = performance.now();

      if (now - lastSentTime < 400) return; // en fazla ~2.5 istek/sn

      lastSentTime = now;

      const scale = Math.min(1, MAX_W / vid.videoWidth);
      console.log(`video: ${vid.videoWidth}x${vid.videoHeight} → gönderilen: ${offscreen.width}x${offscreen.height}`);
      offscreen.width  = Math.round(vid.videoWidth  * scale);
      offscreen.height = Math.round(vid.videoHeight * scale);
      offscreen.getContext("2d").drawImage(vid, 0, 0, offscreen.width, offscreen.height);

      const snapW = offscreen.width;
      const snapH = offscreen.height;

      offscreen.toBlob((blob) => {
        if (!blob) return;
        pending = true;
        const fd = new FormData();
        fd.append("file", blob, "frame.jpg");
        fd.append("model_name", selectedModel);

        const t0 = performance.now();

        fetch(`${INFER_API}/api/v1/infer/sync`, { method:"POST", body:fd })
          .then(res => res.json())
          .then(data => {
            console.log(`toplam: ${(performance.now()-t0).toFixed(0)}ms`); 
            const dets = (data.detections || []).filter(
              d => activeClasses.has(d.label) && d.confidence >= threshold
            );
            setDetections(dets);
            setFrameCount(f => f + 1);
            const r = vid.getBoundingClientRect();
            drawBboxes(dets, snapW, snapH, r.width, r.height);
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
  };

  const runImageInference = async () => {
    if (!fileObjRef.current) return;
    setStatus("processing"); setDetections([]); clearCanvas();
    try {
      const fd = new FormData();
      fd.append("file", fileObjRef.current);
      fd.append("model_name", selectedModel);
      const res  = await fetch(`${INFER_API}/api/v1/infer/sync`, { method:"POST", body:fd });
      if (!res.ok) throw new Error();
      const data = await res.json();
      const dets = (data.detections || []).filter(d => activeClasses.has(d.label) && d.confidence >= threshold);
      setDetections(dets); setStatus("done");
      const img = imgRef.current;
      if (img) {
        const r = img.getBoundingClientRect();
        drawBboxes(dets, img.naturalWidth, img.naturalHeight, r.width, r.height);
      }
    } catch (_) { setStatus("error"); }
  };

  const toggleClass = (cls) => setActiveClasses(prev => { const n = new Set(prev); n.has(cls) ? n.delete(cls) : n.add(cls); return n; });

  const filtered = detections.filter(d => activeClasses.has(d.label) && d.confidence >= threshold);

  return (
    <div style={s.root}>
      <style>{`
        @keyframes spin{to{transform:rotate(360deg)}}
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:3px}
        ::-webkit-scrollbar-track{background:#0a0c0e}
        ::-webkit-scrollbar-thumb{background:#1e2328;border-radius:2px}
        input[type=range]{accent-color:#1D9E75;width:100%}
        button:disabled{opacity:0.4;cursor:not-allowed}
      `}</style>

      {/* Topbar */}
      <div style={s.topbar}>
        <div style={s.topLeft}>
          <div style={s.logoMark}><div style={s.logoInner}/></div>
          <span style={s.logoText}>DRONE ANALYTICS</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          {mode === "video" && (
            <span style={{ ...s.wsTag, color: wsConnected ? "#1D9E75" : "#4a5260", borderColor: wsConnected ? "#0F6E56" : "#1e2328" }}>
              {wsConnected ? "ws ●" : "ws ○"}
            </span>
          )}
          <span style={{ ...s.statusPill, ...STATUS_PILL[status] }}>{STATUS_LABEL[status]}</span>
        </div>
      </div>

      {/* Sidebar */}
      <div style={s.sidebar}>

        <div>
          <div style={s.sectionLabel}>mod</div>
          <div style={s.modeRow}>
            {["video","image"].map(m => (
              <button key={m} style={{ ...s.modeBtn, background: mode===m ? "#0a1f18":"transparent", color: mode===m ? "#1D9E75":"#4a5260", borderColor: mode===m ? "#1D9E75":"#1e2328" }}
                onClick={() => { setMode(m); clearAll(); }}>{m}</button>
            ))}
          </div>
        </div>

        <div>
          <div style={s.sectionLabel}>model seçimi</div>
          {MODELS.map(m => (
            <button key={m.id} style={{ ...s.modelCard, borderColor: selectedModel===m.id ? "#1D9E75":"#1e2328", background: selectedModel===m.id ? "#0a1f18":"transparent" }}
              onClick={() => handleModelChange(m.id)}>
              <div style={{ ...s.modelTag, background: m.color+"22", color: m.color }}>{m.tag}</div>
              <div style={{ flex:1 }}>
                <div style={s.modelName}>{m.label}</div>
                <div style={s.modelDesc}>{m.desc}</div>
              </div>
              {selectedModel === m.id && <div style={s.activeDot}/>}
            </button>
          ))}
        </div>

        <div>
          <div style={s.sectionLabel}>güven eşiği</div>
          <div style={s.sliderLabel}>
            <span style={{ fontSize:10, color:"#4a5260" }}>min. confidence</span>
            <span style={s.sliderVal}>{threshold.toFixed(2)}</span>
          </div>
          <input type="range" min="0" max="100" step="1" value={Math.round(threshold*100)} onChange={e => setThreshold(e.target.value/100)}/>
        </div>

        <div>
          <div style={s.sectionLabel}>sınıf filtresi</div>
          <div style={s.chips}>
            {ALL_CLASSES.map(cls => (
              <button key={cls} style={{ ...s.chip, background: activeClasses.has(cls) ? CLASS_COLORS[cls]+"22":"transparent", color: activeClasses.has(cls) ? CLASS_COLORS[cls]:"#2a3038", borderColor: activeClasses.has(cls) ? CLASS_COLORS[cls]+"66":"#1e2328" }}
                onClick={() => toggleClass(cls)}>{cls}</button>
            ))}
          </div>
        </div>

        {mode === "video" && streamStarted && (
          <div>
            <div style={s.sectionLabel}>istatistik</div>
            <div style={{ fontSize:10, color:"#4a5260", lineHeight:1.9 }}>
              <div>frame: <span style={{ color:"#c8cdd4" }}>{frameCount}</span></div>
              <div style={{ wordBreak:"break-all" }}>id: <span style={{ color:"#c8cdd4" }}>{droneId}</span></div>
            </div>
          </div>
        )}

      </div>

      {/* Main */}
      <div style={s.main}>

        {/* Medya alanı */}
        <div style={s.videoWrap} onClick={() => !hasFile && fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; mode==="video" ? handleVideoFile(f) : handleImageFile(f); }}>

          {mode === "video" && (
            <video ref={videoRef} style={{ width:"100%", height:"100%", objectFit:"contain", display:"block" }}
              muted playsInline onEnded={() => { setStreamStarted(false); setStatus("done"); }}/>
          )}
          {mode === "image" && hasFile && fileObjRef.current && (
            <img ref={imgRef} src={URL.createObjectURL(fileObjRef.current)} alt="preview"
              style={{ width:"100%", height:"100%", objectFit:"contain", display:"block" }}/>
          )}

          <canvas ref={canvasRef} style={s.overlayCanvas}/>

          {!hasFile && (
            <div style={s.dropOverlay}>
              <UploadIcon/>
              <span style={s.dropTitle}>{mode==="video" ? "video yükle" : "görüntü yükle"}</span>
              <span style={s.dropSub}>{mode==="video" ? "mp4 · avi · mov" : "jpg · png"} — sürükle bırak veya tıkla</span>
            </div>
          )}
        </div>

        <input ref={fileRef} type="file" accept={mode==="video" ? "video/*" : "image/*"} style={{ display:"none" }}
          onChange={e => { const f = e.target.files[0]; mode==="video" ? handleVideoFile(f) : handleImageFile(f); }}/>

        <div style={s.actionRow}>
          <button style={s.btn} onClick={clearAll}>temizle</button>
          {mode === "video" ? (
            streamStarted
              ? <button style={s.btnDanger} onClick={stopStream}>durdur ◼</button>
              : <button style={s.btnPrimary} onClick={startStream} disabled={!hasFile}>yayını başlat →</button>
          ) : (
            <button style={s.btnPrimary} onClick={runImageInference} disabled={!hasFile}>analiz et →</button>
          )}
        </div>

        {/* Tespitler */}
        <div style={s.resultsBox}>
          <div style={s.resultsHeader}>
            <span style={s.resultsLabel}>tespitler</span>
            <span style={s.countTag}>{filtered.length > 0 ? `${filtered.length} obje` : "—"}</span>
          </div>
          <div style={s.resultsList}>
            {filtered.length === 0 && (
              <div style={s.emptyState}>
                {mode==="video" && !streamStarted ? "// video yükleyip yayını başlat"
                  : mode==="image" && status==="idle" ? "// görüntü yükleyip analiz et"
                  : "// tespit bekleniyor..."}
              </div>
            )}
            {filtered.map((det, i) => {
              const color = CLASS_COLORS[det.label] || "#888";
              return (
                <div key={i} style={s.detRow}>
                  <div style={{ ...s.detDot, background:color }}/>
                  <span style={s.detLabel}>{det.label}</span>
                  <div style={s.barWrap}><div style={{ ...s.bar, width:`${Math.round(det.confidence*100)}%`, background:color }}/></div>
                  <span style={s.detConf}>{det.confidence.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}