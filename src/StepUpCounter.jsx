import React, { useState, useRef, useEffect, useCallback } from “react”;

// ─── MediaPipe config ─────────────────────────────────────────────────────────
const MODEL_URL =
“https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task”;
// WASM files are copied to public/wasm at build time (see vite.config.js)
const WASM_PATH = “/wasm”;

// ─── Pose detection constants ─────────────────────────────────────────────────
const ANGLE_UP   = 160;
const ANGLE_DOWN = 130;

function angle3(A, B, C) {
const BAx = A.x - B.x, BAy = A.y - B.y;
const BCx = C.x - B.x, BCy = C.y - B.y;
const dot = BAx * BCx + BAy * BCy;
const mag = Math.hypot(BAx, BAy) * Math.hypot(BCx, BCy);
if (mag === 0) return 0;
return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI;
}

function getPoseState(lms) {
if (!lms || lms.length < 33) return null;
const lA = angle3(lms[23], lms[25], lms[27]);
const rA = angle3(lms[24], lms[26], lms[28]);
if (Math.max(lA, rA) > ANGLE_UP && Math.min(lA, rA) > ANGLE_UP) return “UP”;
if (Math.min(lA, rA) < ANGLE_DOWN) return “DOWN”;
return “TRANSITION”;
}

const LEG_PAIRS = [
[23,25],[25,27],[27,29],[27,31],
[24,26],[26,28],[28,30],[28,32],
[23,24],
];
const KEY_PTS = [23,24,25,26,27,28];

function drawSkeleton(canvas, video, landmarkSets) {
if (!canvas || !video) return;
canvas.width  = video.videoWidth  || 640;
canvas.height = video.videoHeight || 480;
const ctx = canvas.getContext(“2d”);
ctx.clearRect(0, 0, canvas.width, canvas.height);
if (!landmarkSets?.length) return;
const lms = landmarkSets[0];
const W = canvas.width, H = canvas.height;
ctx.strokeStyle = “rgba(0,255,150,0.9)”;
ctx.lineWidth = 3;
for (const [a,b] of LEG_PAIRS) {
if (!lms[a] || !lms[b]) continue;
ctx.beginPath();
ctx.moveTo(lms[a].x * W, lms[a].y * H);
ctx.lineTo(lms[b].x * W, lms[b].y * H);
ctx.stroke();
}
for (const i of KEY_PTS) {
if (!lms[i]) continue;
ctx.beginPath();
ctx.arc(lms[i].x * W, lms[i].y * H, 6, 0, Math.PI * 2);
ctx.fillStyle = i <= 24 ? “#FF6B6B” : “#00FF96”;
ctx.fill();
}
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function StepUpCounter() {
const videoRef     = useRef(null);
const canvasRef    = useRef(null);
const fileInputRef = useRef(null);
const landmarkerRef= useRef(null);
const rafRef       = useRef(null);
const lastTimeRef  = useRef(-1);
const poseStateRef = useRef(“UP”);
const pendingRef   = useRef(false);
const countRef     = useRef(0);

const [count,      setCount]      = useState(0);
const [videoSrc,   setVideoSrc]   = useState(null);
const [mpStatus,   setMpStatus]   = useState(“loading”); // loading|ready|error
const [mpMsg,      setMpMsg]      = useState(“Initialising…”);
const [appPhase,   setAppPhase]   = useState(“idle”);    // idle|ready|running|done
const [poseLabel,  setPoseLabel]  = useState(”—”);
const [kneeAngle,  setKneeAngle]  = useState(null);
const [showTip,    setShowTip]    = useState(true);

// ── Load MediaPipe ──────────────────────────────────────────────────────────
useEffect(() => {
let cancelled = false;
(async () => {
try {
const { PoseLandmarker, FilesetResolver } = await import(”@mediapipe/tasks-vision”);
if (cancelled) return;
setMpMsg(“Downloading model (~10 MB)…”);
const resolver = await FilesetResolver.forVisionTasks(WASM_PATH);
const pl = await PoseLandmarker.createFromOptions(resolver, {
baseOptions: { modelAssetPath: MODEL_URL, delegate: “CPU” },
runningMode: “VIDEO”,
numPoses: 1,
});
if (cancelled) return;
landmarkerRef.current = pl;
setMpStatus(“ready”);
setMpMsg(””);
} catch (err) {
if (!cancelled) {
console.error(err);
setMpStatus(“error”);
setMpMsg(“Failed: “ + err.message);
}
}
})();
return () => { cancelled = true; };
}, []);

// ── File selection ──────────────────────────────────────────────────────────
function pickVideo() { fileInputRef.current?.click(); }

function handleFile(e) {
const file = e.target.files?.[0];
if (!file) return;
cancelAnimationFrame(rafRef.current);
countRef.current = 0; poseStateRef.current = “UP”; pendingRef.current = false; lastTimeRef.current = -1;
setCount(0); setPoseLabel(”—”); setKneeAngle(null); setAppPhase(“ready”);
setVideoSrc(URL.createObjectURL(file));
}

// ── Frame loop ──────────────────────────────────────────────────────────────
const processFrame = useCallback(() => {
const video = videoRef.current;
const lmk   = landmarkerRef.current;
if (!video || !lmk) return;
if (video.paused || video.ended) { if (video.ended) setAppPhase(“done”); return; }
if (video.currentTime === lastTimeRef.current) {
rafRef.current = requestAnimationFrame(processFrame); return;
}
lastTimeRef.current = video.currentTime;
const result = lmk.detectForVideo(video, performance.now());
const lms    = result?.landmarks?.[0];
drawSkeleton(canvasRef.current, video, result?.landmarks ?? []);

```
if (lms) {
  const lA = angle3(lms[23], lms[25], lms[27]);
  const rA = angle3(lms[24], lms[26], lms[28]);
  setKneeAngle(((lA + rA) / 2).toFixed(1));
  const state = getPoseState(lms);
  setPoseLabel(state ?? "TRANSITION");
  if (state === "DOWN" && poseStateRef.current === "UP") {
    poseStateRef.current = "DOWN"; pendingRef.current = true;
  } else if (state === "UP" && poseStateRef.current === "DOWN" && pendingRef.current) {
    poseStateRef.current = "UP"; pendingRef.current = false;
    countRef.current += 1; setCount(countRef.current);
  }
}
rafRef.current = requestAnimationFrame(processFrame);
```

}, []);

function onPlay()  { setAppPhase(“running”); rafRef.current = requestAnimationFrame(processFrame); }
function onPause() { cancelAnimationFrame(rafRef.current); }
function onEnded() { cancelAnimationFrame(rafRef.current); setAppPhase(“done”); }

function reset() {
cancelAnimationFrame(rafRef.current);
countRef.current = 0; poseStateRef.current = “UP”; pendingRef.current = false;
setCount(0); setPoseLabel(”—”); setKneeAngle(null);
if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.pause(); }
setAppPhase(videoSrc ? “ready” : “idle”);
}

// ── Derived ─────────────────────────────────────────────────────────────────
const stateColor  = { UP:”#00FF96”, DOWN:”#FF6B6B”, TRANSITION:”#FFD700”, “—”:”#444” }[poseLabel] ?? “#444”;
const isBlocked   = mpStatus !== “ready”;
const statusLabel = mpStatus === “loading” ? “⏳ Loading AI…”
: mpStatus === “error”   ? “❌ Error”
: appPhase === “running” ? “▶ Analysing”
: appPhase === “done”    ? “✅ Done”
: “✅ Ready”;
const statusColor = mpStatus === “error” ? “#FF6B6B” : appPhase === “running” ? “#00FF96” : “#00FF96”;

// ─── Render ──────────────────────────────────────────────────────────────────
return (
<div style={css.root}>
{/* Background glow */}
<div style={css.glow} />

```
  {/* iOS install banner */}
  {showTip && (
    <div style={css.tipBanner}>
      <span style={css.tipText}>
        📲 <strong>Add to Home Screen:</strong> tap Share → "Add to Home Screen" in Safari
      </span>
      <button style={css.tipClose} onClick={() => setShowTip(false)}>✕</button>
    </div>
  )}

  {/* Header */}
  <div style={css.header}>
    <div style={css.logoRow}>
      <span style={css.logoIcon}>⬆</span>
      <div>
        <div style={css.appTitle}>STEP‑UP COUNTER</div>
        <div style={css.appSub}>AI Rep Tracker · MediaPipe Pose</div>
      </div>
    </div>
    <div style={{ ...css.pill, color: statusColor }}>
      {statusLabel}
    </div>
  </div>

  {/* Video box */}
  <div style={css.videoBox}>
    {videoSrc ? (
      <div style={{ position:"relative", width:"100%" }}>
        <video
          ref={videoRef}
          src={videoSrc}
          style={css.video}
          controls
          playsInline
          webkit-playsinline="true"
          crossOrigin="anonymous"
          onPlay={onPlay}
          onPause={onPause}
          onEnded={onEnded}
        />
        <canvas ref={canvasRef} style={css.canvas} />
      </div>
    ) : (
      <div style={css.emptyState}>
        <div style={{ fontSize:52 }}>🎬</div>
        <div style={css.emptyText}>Upload a step-up workout video</div>
        <div style={css.emptyHint}>MP4 · MOV · WebM</div>
      </div>
    )}
  </div>

  {/* Choose video button */}
  <button
    style={{ ...css.chooseBtn, opacity: isBlocked ? 0.45 : 1 }}
    disabled={isBlocked}
    onClick={pickVideo}
  >
    📂 &nbsp; Choose Video
  </button>
  <input
    ref={fileInputRef}
    type="file"
    accept="video/*"
    style={{ display:"none" }}
    onChange={handleFile}
  />

  {/* Status / loading message */}
  {mpMsg && (
    <div style={{ ...css.statusMsg, color: mpStatus === "error" ? "#FF6B6B" : "#FFD700" }}>
      {mpMsg}
    </div>
  )}

  {/* Done banner */}
  {appPhase === "done" && (
    <div style={css.doneBanner}>
      ✅ Done — {count} step-up{count !== 1 ? "s" : ""} detected
    </div>
  )}

  {/* Stats row */}
  <div style={css.statsRow}>
    {/* Rep counter */}
    <div style={{ ...css.card, ...css.repCard }}>
      <div style={css.repLabel}>STEP‑UPS</div>
      <div style={css.repNum}>{count}</div>
      <div style={css.repSub}>reps</div>
    </div>

    {/* State + Angle */}
    <div style={{ flex:1, display:"flex", flexDirection:"column", gap:10 }}>
      <div style={css.card}>
        <div style={css.cardLabel}>STATE</div>
        <div style={{ ...css.cardVal, color: stateColor }}>{poseLabel}</div>
        <div style={css.cardHint}>
          { poseLabel==="UP"         ? "Top of rep"
          : poseLabel==="DOWN"       ? "Step engaged"
          : poseLabel==="TRANSITION" ? "Mid-movement"
          : "Waiting for video…" }
        </div>
      </div>
      <div style={css.card}>
        <div style={css.cardLabel}>KNEE ANGLE</div>
        <div style={{ ...css.cardVal, color:"#FFD700" }}>
          {kneeAngle ? `${kneeAngle}°` : "—"}
        </div>
        <div style={css.cardHint}>UP &gt;{ANGLE_UP}° · DOWN &lt;{ANGLE_DOWN}°</div>
      </div>
    </div>
  </div>

  {/* How it works */}
  <div style={css.infoCard}>
    <div style={css.infoTitle}>HOW IT WORKS</div>
    <div style={css.infoBody}>
      MediaPipe tracks 33 body landmarks. Hip → knee → ankle angles detect position.
      A <span style={{ color:"#ccc" }}>DOWN → UP</span> cycle = +1 rep.
    </div>
    <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
      {[["#00FF96","Legs straight → UP"],["#FF6B6B","Knee bent → DOWN"],["#fff","DOWN → UP = +1 rep"]
      ].map(([c,l]) => (
        <div key={l} style={{ display:"flex", alignItems:"center", gap:8 }}>
          <span style={{ width:9, height:9, borderRadius:"50%", background:c, flexShrink:0, display:"inline-block" }} />
          <span style={css.legendText}>{l}</span>
        </div>
      ))}
    </div>
  </div>

  {/* Reset */}
  {(count > 0 || appPhase === "done") && (
    <button style={css.resetBtn} onClick={reset}>↺ &nbsp; Reset Counter</button>
  )}

  {/* Bottom safe area */}
  <div style={{ height:32 }} />
</div>
```

);
}

// ─── Styles ────────────────────────────────────────────────────────────────────
const FF = “‘Courier New’, Courier, monospace”;

const css = {
root: {
minHeight: “100vh”,
backgroundColor: “#08080F”,
color: “#E0E0F0”,
fontFamily: FF,
position: “relative”,
overflowX: “hidden”,
padding: “12px 16px”,
paddingTop: “max(12px, env(safe-area-inset-top))”,
paddingBottom: “max(16px, env(safe-area-inset-bottom))”,
},
glow: {
position: “fixed”, top:”-20%”, left:”-10%”,
width:“65vw”, height:“65vw”, borderRadius:“50%”,
background:“radial-gradient(circle, rgba(0,255,150,0.07) 0%, transparent 70%)”,
pointerEvents:“none”, zIndex:0,
},

// Install tip
tipBanner: {
display:“flex”, alignItems:“flex-start”, justifyContent:“space-between”,
background:“rgba(0,255,150,0.07)”, border:“1px solid rgba(0,255,150,0.2)”,
borderRadius:10, padding:“10px 12px”, marginBottom:14,
position:“relative”, zIndex:1,
},
tipText:  { fontSize:12, color:”#888”, lineHeight:“18px”, flex:1 },
tipClose: {
background:“none”, border:“none”, color:”#555”, fontSize:16,
cursor:“pointer”, padding:“0 0 0 10px”, lineHeight:1,
},

// Header
header: {
display:“flex”, alignItems:“center”, justifyContent:“space-between”,
marginBottom:14, position:“relative”, zIndex:1,
},
logoRow:  { display:“flex”, alignItems:“center”, gap:10 },
logoIcon: { fontSize:28, lineHeight:1 },
appTitle: { fontSize:16, fontWeight:“bold”, letterSpacing:3, color:”#00FF96” },
appSub:   { fontSize:10, letterSpacing:1.5, color:”#555”, marginTop:2 },
pill: {
fontSize:11, letterSpacing:0.5,
background:“rgba(0,255,150,0.08)”, border:“1px solid rgba(0,255,150,0.2)”,
borderRadius:20, padding:“5px 11px”,
},

// Video
videoBox: {
background:”#10101A”, borderRadius:12, border:“1px solid #1E1E30”,
overflow:“hidden”, minHeight:200,
display:“flex”, alignItems:“center”, justifyContent:“center”,
marginBottom:12, position:“relative”, zIndex:1,
},
video:  { width:“100%”, display:“block”, borderRadius:12 },
canvas: { position:“absolute”, top:0, left:0, width:“100%”, height:“100%”, pointerEvents:“none” },
emptyState: { textAlign:“center”, padding:“40px 20px” },
emptyText:  { color:”#555”, fontSize:14, marginTop:10, marginBottom:6 },
emptyHint:  { color:”#333”, fontSize:11, letterSpacing:1 },

// Buttons
chooseBtn: {
width:“100%”, padding:“14px 0”,
background:“linear-gradient(135deg,#00FF96,#00C97A)”,
color:”#08080F”, border:“none”, borderRadius:10,
fontSize:14, fontWeight:“bold”, letterSpacing:2,
cursor:“pointer”, fontFamily:FF, marginBottom:10,
position:“relative”, zIndex:1,
},
statusMsg: { textAlign:“center”, fontSize:12, letterSpacing:0.5, marginBottom:8 },
doneBanner: {
background:“rgba(0,255,150,0.08)”, border:“1px solid rgba(0,255,150,0.2)”,
borderRadius:10, padding:“12px 16px”, marginBottom:12,
color:”#00FF96”, fontSize:13, textAlign:“center”, letterSpacing:0.5,
position:“relative”, zIndex:1,
},

// Stats
statsRow: {
display:“flex”, gap:10, marginBottom:12,
position:“relative”, zIndex:1,
},
card: {
background:”#10101A”, border:“1px solid #1E1E30”,
borderRadius:12, padding:“14px 14px”,
},
repCard: {
background:“linear-gradient(135deg,rgba(0,255,150,0.09),rgba(0,255,150,0.02))”,
border:“1px solid rgba(0,255,150,0.2)”,
display:“flex”, flexDirection:“column”,
alignItems:“center”, justifyContent:“center”,
minWidth:110,
},
repLabel: { fontSize:10, letterSpacing:3, color:”#00FF96”, marginBottom:6, opacity:0.8 },
repNum:   { fontSize:64, fontWeight:“bold”, color:”#00FF96”, lineHeight:“1”,
textShadow:“0 0 30px rgba(0,255,150,0.5)” },
repSub:   { fontSize:11, color:”#444”, letterSpacing:2, marginTop:4 },
cardLabel:{ fontSize:10, letterSpacing:2.5, color:”#555”, marginBottom:5 },
cardVal:  { fontSize:22, fontWeight:“bold”, letterSpacing:1, marginBottom:3 },
cardHint: { fontSize:10, color:”#333”, letterSpacing:0.3, lineHeight:“14px” },

// Info card
infoCard: {
background:”#0C0C14”, border:“1px solid #1A1A28”,
borderRadius:12, padding:16, marginBottom:12,
position:“relative”, zIndex:1,
},
infoTitle:  { fontSize:10, letterSpacing:3, color:”#444”, marginBottom:10 },
infoBody:   { fontSize:12, color:”#555”, lineHeight:“20px”, marginBottom:12 },
legendText: { fontSize:12, color:”#666” },

// Reset
resetBtn: {
width:“100%”, padding:“12px 0”,
background:“transparent”, border:“1px solid #2A2A3A”,
borderRadius:10, color:”#555”, fontSize:12, letterSpacing:2,
cursor:“pointer”, fontFamily:FF,
position:“relative”, zIndex:1,
},
};