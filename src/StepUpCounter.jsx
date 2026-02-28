import React, { useState, useRef, useEffect, useCallback } from “react”

const MODEL_URL = “https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task”
const CDN = “https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14”

const ANGLE_UP = 160
const ANGLE_DOWN = 130

function angle3(A, B, C) {
const BAx = A.x - B.x, BAy = A.y - B.y
const BCx = C.x - B.x, BCy = C.y - B.y
const dot = BAx * BCx + BAy * BCy
const mag = Math.hypot(BAx, BAy) * Math.hypot(BCx, BCy)
if (mag === 0) return 0
return (Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180) / Math.PI
}

function getPoseState(lms) {
if (!lms || lms.length < 33) return null
const lA = angle3(lms[23], lms[25], lms[27])
const rA = angle3(lms[24], lms[26], lms[28])
if (Math.max(lA, rA) > ANGLE_UP && Math.min(lA, rA) > ANGLE_UP) return “UP”
if (Math.min(lA, rA) < ANGLE_DOWN) return “DOWN”
return “TRANSITION”
}

const LEG_PAIRS = [[23,25],[25,27],[27,29],[27,31],[24,26],[26,28],[28,30],[28,32],[23,24]]
const KEY_PTS = [23,24,25,26,27,28]

function drawSkeleton(canvas, video, landmarkSets) {
if (!canvas || !video) return
canvas.width = video.videoWidth || 640
canvas.height = video.videoHeight || 480
const ctx = canvas.getContext(“2d”)
ctx.clearRect(0, 0, canvas.width, canvas.height)
if (!landmarkSets || !landmarkSets.length) return
const lms = landmarkSets[0]
const W = canvas.width, H = canvas.height
ctx.strokeStyle = “rgba(0,255,150,0.9)”
ctx.lineWidth = 3
for (const [a,b] of LEG_PAIRS) {
if (!lms[a] || !lms[b]) continue
ctx.beginPath()
ctx.moveTo(lms[a].x * W, lms[a].y * H)
ctx.lineTo(lms[b].x * W, lms[b].y * H)
ctx.stroke()
}
for (const i of KEY_PTS) {
if (!lms[i]) continue
ctx.beginPath()
ctx.arc(lms[i].x * W, lms[i].y * H, 6, 0, Math.PI * 2)
ctx.fillStyle = i <= 24 ? “#FF6B6B” : “#00FF96”
ctx.fill()
}
}

function loadMediaPipe() {
if (window.__mpPromise) return window.__mpPromise
window.__mpPromise = new Promise((resolve, reject) => {
const s = document.createElement(“script”)
s.type = “module”
s.textContent = `import * as v from "${CDN}/vision_bundle.mjs" window.__mp = v window.dispatchEvent(new Event("__mpReady"))`
s.onerror = reject
document.head.appendChild(s)
window.addEventListener(”__mpReady”, () => resolve(window.__mp), { once: true })
setTimeout(() => reject(new Error(“MediaPipe load timeout”)), 30000)
})
return window.__mpPromise
}

export default function StepUpCounter() {
const videoRef = useRef(null)
const canvasRef = useRef(null)
const fileInputRef = useRef(null)
const landmarkerRef = useRef(null)
const rafRef = useRef(null)
const lastTimeRef = useRef(-1)
const poseStateRef = useRef(“UP”)
const pendingRef = useRef(false)
const countRef = useRef(0)

const [count, setCount] = useState(0)
const [videoSrc, setVideoSrc] = useState(null)
const [mpStatus, setMpStatus] = useState(“loading”)
const [mpMsg, setMpMsg] = useState(“Initialising…”)
const [appPhase, setAppPhase] = useState(“idle”)
const [poseLabel, setPoseLabel] = useState(”–”)
const [kneeAngle, setKneeAngle] = useState(null)
const [showTip, setShowTip] = useState(true)

useEffect(() => {
let cancelled = false
async function init() {
try {
setMpMsg(“Loading MediaPipe…”)
const mp = await loadMediaPipe()
if (cancelled) return
const { PoseLandmarker, FilesetResolver } = mp
setMpMsg(“Downloading model (~10MB)…”)
const resolver = await FilesetResolver.forVisionTasks(CDN + “/wasm”)
const pl = await PoseLandmarker.createFromOptions(resolver, {
baseOptions: { modelAssetPath: MODEL_URL, delegate: “CPU” },
runningMode: “VIDEO”,
numPoses: 1
})
if (cancelled) return
landmarkerRef.current = pl
setMpStatus(“ready”)
setMpMsg(””)
} catch(err) {
if (!cancelled) {
setMpStatus(“error”)
setMpMsg(“Error: “ + err.message)
}
}
}
init()
return () => { cancelled = true }
}, [])

function handleFile(e) {
const file = e.target.files && e.target.files[0]
if (!file) return
cancelAnimationFrame(rafRef.current)
countRef.current = 0
poseStateRef.current = “UP”
pendingRef.current = false
lastTimeRef.current = -1
setCount(0)
setPoseLabel(”–”)
setKneeAngle(null)
setAppPhase(“ready”)
setVideoSrc(URL.createObjectURL(file))
}

const processFrame = useCallback(() => {
const video = videoRef.current
const lmk = landmarkerRef.current
if (!video || !lmk) return
if (video.paused || video.ended) {
if (video.ended) setAppPhase(“done”)
return
}
if (video.currentTime === lastTimeRef.current) {
rafRef.current = requestAnimationFrame(processFrame)
return
}
lastTimeRef.current = video.currentTime
const result = lmk.detectForVideo(video, performance.now())
const lms = result && result.landmarks && result.landmarks[0]
drawSkeleton(canvasRef.current, video, result ? result.landmarks : [])
if (lms) {
const lA = angle3(lms[23], lms[25], lms[27])
const rA = angle3(lms[24], lms[26], lms[28])
setKneeAngle(((lA + rA) / 2).toFixed(1))
const state = getPoseState(lms)
setPoseLabel(state || “TRANSITION”)
if (state === “DOWN” && poseStateRef.current === “UP”) {
poseStateRef.current = “DOWN”
pendingRef.current = true
} else if (state === “UP” && poseStateRef.current === “DOWN” && pendingRef.current) {
poseStateRef.current = “UP”
pendingRef.current = false
countRef.current += 1
setCount(countRef.current)
}
}
rafRef.current = requestAnimationFrame(processFrame)
}, [])

function onPlay() { setAppPhase(“running”); rafRef.current = requestAnimationFrame(processFrame) }
function onPause() { cancelAnimationFrame(rafRef.current) }
function onEnded() { cancelAnimationFrame(rafRef.current); setAppPhase(“done”) }

function reset() {
cancelAnimationFrame(rafRef.current)
countRef.current = 0
poseStateRef.current = “UP”
pendingRef.current = false
setCount(0)
setPoseLabel(”–”)
setKneeAngle(null)
if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.pause() }
setAppPhase(videoSrc ? “ready” : “idle”)
}

const stateColor = poseLabel === “UP” ? “#00FF96” : poseLabel === “DOWN” ? “#FF6B6B” : poseLabel === “TRANSITION” ? “#FFD700” : “#444”
const isBlocked = mpStatus !== “ready”

const FF = “Courier New, Courier, monospace”

return (
<div style={{ minHeight:“100vh”, backgroundColor:”#08080F”, color:”#E0E0F0”, fontFamily:FF, padding:“12px 16px”, paddingTop:“max(12px, env(safe-area-inset-top))”, paddingBottom:“max(16px, env(safe-area-inset-bottom))”, overflowX:“hidden” }}>

```
  <div style={{ position:"fixed", top:"-20%", left:"-10%", width:"65vw", height:"65vw", borderRadius:"50%", background:"radial-gradient(circle, rgba(0,255,150,0.07) 0%, transparent 70%)", pointerEvents:"none" }} />

  {showTip && (
    <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", background:"rgba(0,255,150,0.07)", border:"1px solid rgba(0,255,150,0.2)", borderRadius:10, padding:"10px 12px", marginBottom:14 }}>
      <span style={{ fontSize:12, color:"#888", lineHeight:"18px", flex:1 }}>
        Add to Home Screen: tap Share then Add to Home Screen in Safari
      </span>
      <button onClick={() => setShowTip(false)} style={{ background:"none", border:"none", color:"#555", fontSize:16, cursor:"pointer", padding:"0 0 0 10px" }}>x</button>
    </div>
  )}

  <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:14 }}>
    <div style={{ display:"flex", alignItems:"center", gap:10 }}>
      <span style={{ fontSize:28 }}>up</span>
      <div>
        <div style={{ fontSize:16, fontWeight:"bold", letterSpacing:3, color:"#00FF96" }}>STEP-UP COUNTER</div>
        <div style={{ fontSize:10, letterSpacing:1.5, color:"#555", marginTop:2 }}>AI Rep Tracker</div>
      </div>
    </div>
    <div style={{ fontSize:11, background:"rgba(0,255,150,0.08)", border:"1px solid rgba(0,255,150,0.2)", borderRadius:20, padding:"5px 11px", color: mpStatus === "error" ? "#FF6B6B" : "#00FF96" }}>
      {mpStatus === "loading" ? "Loading..." : mpStatus === "error" ? "Error" : appPhase === "running" ? "Analysing" : "Ready"}
    </div>
  </div>

  <div style={{ background:"#10101A", borderRadius:12, border:"1px solid #1E1E30", overflow:"hidden", minHeight:200, display:"flex", alignItems:"center", justifyContent:"center", marginBottom:12, position:"relative" }}>
    {videoSrc ? (
      <div style={{ position:"relative", width:"100%" }}>
        <video ref={videoRef} src={videoSrc} style={{ width:"100%", display:"block" }} controls playsInline crossOrigin="anonymous" onPlay={onPlay} onPause={onPause} onEnded={onEnded} />
        <canvas ref={canvasRef} style={{ position:"absolute", top:0, left:0, width:"100%", height:"100%", pointerEvents:"none" }} />
      </div>
    ) : (
      <div style={{ textAlign:"center", padding:"40px 20px" }}>
        <div style={{ fontSize:52 }}>video</div>
        <div style={{ color:"#555", fontSize:14, marginTop:10 }}>Upload a step-up workout video</div>
        <div style={{ color:"#333", fontSize:11, marginTop:6 }}>MP4, MOV, WebM</div>
      </div>
    )}
  </div>

  <button onClick={() => fileInputRef.current && fileInputRef.current.click()} disabled={isBlocked} style={{ width:"100%", padding:"14px 0", background:"linear-gradient(135deg,#00FF96,#00C97A)", color:"#08080F", border:"none", borderRadius:10, fontSize:14, fontWeight:"bold", letterSpacing:2, cursor: isBlocked ? "not-allowed" : "pointer", fontFamily:FF, marginBottom:10, opacity: isBlocked ? 0.4 : 1 }}>
    Choose Video
  </button>
  <input ref={fileInputRef} type="file" accept="video/*" style={{ display:"none" }} onChange={handleFile} />

  {mpMsg && <div style={{ textAlign:"center", fontSize:12, color: mpStatus === "error" ? "#FF6B6B" : "#FFD700", marginBottom:8 }}>{mpMsg}</div>}

  {appPhase === "done" && (
    <div style={{ background:"rgba(0,255,150,0.08)", border:"1px solid rgba(0,255,150,0.2)", borderRadius:10, padding:"12px 16px", marginBottom:12, color:"#00FF96", fontSize:13, textAlign:"center" }}>
      Done - {count} step-up{count !== 1 ? "s" : ""} detected
    </div>
  )}

  <div style={{ display:"flex", gap:10, marginBottom:12 }}>
    <div style={{ background:"linear-gradient(135deg,rgba(0,255,150,0.09),rgba(0,255,150,0.02))", border:"1px solid rgba(0,255,150,0.2)", borderRadius:12, padding:"14px", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", minWidth:110 }}>
      <div style={{ fontSize:10, letterSpacing:3, color:"#00FF96", marginBottom:6, opacity:0.8 }}>STEP-UPS</div>
      <div style={{ fontSize:64, fontWeight:"bold", color:"#00FF96", lineHeight:1 }}>{count}</div>
      <div style={{ fontSize:11, color:"#444", letterSpacing:2, marginTop:4 }}>reps</div>
    </div>
    <div style={{ flex:1, display:"flex", flexDirection:"column", gap:10 }}>
      <div style={{ background:"#10101A", border:"1px solid #1E1E30", borderRadius:12, padding:14 }}>
        <div style={{ fontSize:10, letterSpacing:2.5, color:"#555", marginBottom:5 }}>STATE</div>
        <div style={{ fontSize:22, fontWeight:"bold", color:stateColor }}>{poseLabel}</div>
        <div style={{ fontSize:10, color:"#333" }}>
          {poseLabel === "UP" ? "Top of rep" : poseLabel === "DOWN" ? "Step engaged" : poseLabel === "TRANSITION" ? "Mid-movement" : "Waiting..."}
        </div>
      </div>
      <div style={{ background:"#10101A", border:"1px solid #1E1E30", borderRadius:12, padding:14 }}>
        <div style={{ fontSize:10, letterSpacing:2.5, color:"#555", marginBottom:5 }}>KNEE ANGLE</div>
        <div style={{ fontSize:22, fontWeight:"bold", color:"#FFD700" }}>{kneeAngle ? kneeAngle + "deg" : "--"}</div>
        <div style={{ fontSize:10, color:"#333" }}>UP over 160, DOWN under 130</div>
      </div>
    </div>
  </div>

  <div style={{ background:"#0C0C14", border:"1px solid #1A1A28", borderRadius:12, padding:16, marginBottom:12 }}>
    <div style={{ fontSize:10, letterSpacing:3, color:"#444", marginBottom:10 }}>HOW IT WORKS</div>
    <div style={{ fontSize:12, color:"#555", lineHeight:"20px", marginBottom:12 }}>
      MediaPipe tracks 33 body landmarks. Hip to knee to ankle angles detect position. DOWN to UP cycle counts as one rep.
    </div>
    {[["#00FF96","Legs straight = UP"],["#FF6B6B","Knee bent = DOWN"],["#fff","DOWN to UP = plus 1 rep"]].map(([c,l]) => (
      <div key={l} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:6 }}>
        <span style={{ width:9, height:9, borderRadius:"50%", background:c, display:"inline-block", flexShrink:0 }} />
        <span style={{ fontSize:12, color:"#666" }}>{l}</span>
      </div>
    ))}
  </div>

  {(count > 0 || appPhase === "done") && (
    <button onClick={reset} style={{ width:"100%", padding:"12px 0", background:"transparent", border:"1px solid #2A2A3A", borderRadius:10, color:"#555", fontSize:12, letterSpacing:2, cursor:"pointer", fontFamily:FF }}>
      Reset Counter
    </button>
  )}

  <div style={{ height:32 }} />
</div>
```

)
}