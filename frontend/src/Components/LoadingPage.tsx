import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { WS_BASE } from '../config';
import { AnalysisResult } from '../types';

const STAGE_MAP: Record<string, { message: string; progress: number }> = {
  MODEL_STARTED:  { message: "consulting AIs...",      progress: 10 },
  MODEL_DONE_ALL: { message: "extracting claims...",   progress: 25 },
  embed_claims:   { message: "embedding claims...",    progress: 35 },
  cluster_claims: { message: "clustering...",          progress: 45 },
  compute_umap:   { message: "mapping 3D space...",    progress: 55 },
  nli_verify:     { message: "verifying sources...",   progress: 70 },
  score_clusters: { message: "scoring trust...",       progress: 85 },
  DONE:           { message: "complete!",              progress: 100 },
};

export default function LoadingPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const jobId = new URLSearchParams(location.search).get('job');

  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("gathering information...");
  const [fatalError, setFatalError] = useState<string | null>(null);
  const modelDoneCount = useRef(0);

  useEffect(() => {
    if (!jobId) {
      // No job id — fall back to a simple animation and navigate to arena
      const timer = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(timer);
            setTimeout(() => navigate('/arena'), 500);
            return 100;
          }
          return prev + 2;
        });
      }, 120);
      return () => clearInterval(timer);
    }

    const ws = new WebSocket(`${WS_BASE}/ws/analysis/${jobId}`);

    ws.onmessage = (e) => {
      let event: { type: string; payload?: Record<string, unknown> };
      try {
        event = JSON.parse(e.data);
      } catch {
        return;
      }

      const { type, payload } = event;

      if (type === 'MODEL_STARTED') {
        setMessage(STAGE_MAP.MODEL_STARTED.message);
        setProgress(STAGE_MAP.MODEL_STARTED.progress);
      } else if (type === 'MODEL_DONE') {
        modelDoneCount.current += 1;
        if (modelDoneCount.current >= 5) {
          setMessage(STAGE_MAP.MODEL_DONE_ALL.message);
          setProgress(STAGE_MAP.MODEL_DONE_ALL.progress);
        }
      } else if (type === 'STAGE_PROGRESS' && payload) {
        const stage = payload.stage as string;
        if (STAGE_MAP[stage]) {
          setMessage(STAGE_MAP[stage].message);
          setProgress(STAGE_MAP[stage].progress);
        }
      } else if (type === 'DONE') {
        setMessage(STAGE_MAP.DONE.message);
        setProgress(100);
        const result = (payload as { result: AnalysisResult }).result;
        sessionStorage.setItem(`result_${jobId}`, JSON.stringify(result));
        setTimeout(() => navigate(`/arena?job=${jobId}`), 800);
      } else if (type === 'FATAL_ERROR') {
        const errMsg = payload ? (payload.message as string) ?? 'Analysis failed' : 'Analysis failed';
        setFatalError(errMsg);
      }
    };

    ws.onerror = () => setFatalError('Connection to server failed.');

    return () => ws.close();
  }, [jobId, navigate]);

  const percentage = progress;
  const messageIndex = Math.min(
    Math.floor((progress / 100) * 5),
    4
  );

  // --- INDIVIDUAL CIRCLE MATH ---
  const baseRadius = 150;

  // Layer 1 (Outer Thick Smoke)
  const r1 = baseRadius + 8;
  const circ1 = 2 * Math.PI * r1;
  const offset1 = circ1 - ((percentage / 100) * circ1);

  // Layer 2 (Inner Thin Smoke)
  const r2 = baseRadius - 6;
  const circ2 = 2 * Math.PI * r2;
  const offset2 = circ2 - ((percentage / 100) * circ2);

  // Layer 3 (Core Energy Line)
  const r3 = baseRadius;
  const circ3 = 2 * Math.PI * r3;
  const offset3 = circ3 - ((percentage / 100) * circ3);

  if (fatalError) {
    return (
      <div className="relative flex flex-col items-center justify-center w-screen h-screen bg-[#0A0E1A] overflow-hidden px-4 text-[#EBF0FF]">
        <p className="text-[#FF4757] text-xl font-bold mb-4">Analysis Failed</p>
        <p className="text-[#90A2B3] mb-8">{fatalError}</p>
        <button
          onClick={() => navigate('/')}
          className="px-6 py-3 bg-[#121825] border border-[#2C3A50] rounded-xl text-[#EBF0FF] hover:bg-[#1A2335] transition-colors"
        >
          Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="relative flex flex-col items-center justify-center w-screen h-screen bg-[#0A0E1A] overflow-hidden px-4">

      {/* Ambient Background Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#5273FB] opacity-[0.04] blur-[120px] rounded-full pointer-events-none" />

      {/* Main Content Wrapper */}
      <div className="relative flex flex-col items-center w-[817px] z-10 mt-10">

        {/* Dynamic Loading Text */}
        <div className="flex items-center justify-center h-[40px] mb-2">
          <motion.div
            key={messageIndex}
            initial={{ opacity: 0.4 }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
          >
            <p className="text-[#A9BDE8] tracking-[0.3em] font-mono text-[13px] uppercase">
              {message}
            </p>
          </motion.div>
        </div>

        {/* --- ETHEREAL FLUID ENERGY RING --- */}
        <div className="relative flex items-center justify-center w-[400px] h-[400px] mb-8">

          {/* Centered Percentage Display */}
          <div className="absolute inset-0 flex items-center justify-center z-20">
            <motion.div className="flex items-baseline justify-center">
              <p
                className="text-transparent bg-clip-text bg-gradient-to-b from-[#FFFFFF] to-[#A9BDE8]"
                style={{ fontFamily: 'Inter', fontSize: '72px', fontWeight: 300, letterSpacing: '-0.02em', filter: 'drop-shadow(0 0 15px rgba(169,189,232,0.3))' }}
              >
                {percentage}
              </p>
              <span className="text-[32px] opacity-50 font-light ml-2">%</span>
            </motion.div>
          </div>

          {/* SVG Complex Filters & Fluid Rings */}
          <svg width="400" height="400" viewBox="0 0 400 400" className="absolute inset-0 z-10 pointer-events-none overflow-visible">
            <defs>
              <filter id="fluidSmoke" x="-50%" y="-50%" width="200%" height="200%">
                <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="4" result="noise">
                  <animate attributeName="baseFrequency" values="0.015; 0.02; 0.015" dur="15s" repeatCount="indefinite" />
                </feTurbulence>
                <feDisplacementMap in="SourceGraphic" in2="noise" scale="35" xChannelSelector="R" yChannelSelector="G" result="displaced" />
                <feGaussianBlur in="displaced" stdDeviation="4" result="blurred" />
                <feMerge>
                  <feMergeNode in="blurred" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>

              <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>

              <linearGradient id="bluePlasma" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#FFFFFF" />
                <stop offset="50%" stopColor="#A9BDE8" />
                <stop offset="100%" stopColor="#5273FB" />
              </linearGradient>
            </defs>

            {/* Faint background track */}
            <circle cx="200" cy="200" r={baseRadius} fill="none" stroke="#1A2335" strokeWidth="1" opacity="0.3" />

            {/* LAYER 1: Thick, displaced outer smoke (Rotates Clockwise) */}
            <motion.g animate={{ rotate: 360 }} transition={{ duration: 40, repeat: Infinity, ease: "linear" }} style={{ originX: '50%', originY: '50%' }}>
              <motion.circle
                cx="200" cy="200" r={r1}
                fill="none" stroke="#3464c9" strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circ1}
                animate={{ strokeDashoffset: offset1 }}
                transition={{ type: "tween", ease: "easeOut", duration: 0.5 }}
                filter="url(#fluidSmoke)"
                opacity={0.5}
                style={{ rotate: "-90deg", originX: '50%', originY: '50%' }}
              />
            </motion.g>

            {/* LAYER 2: Thinner, displaced inner smoke (Rotates Counter-Clockwise) */}
            <motion.g animate={{ rotate: -360 }} transition={{ duration: 30, repeat: Infinity, ease: "linear" }} style={{ originX: '50%', originY: '50%' }}>
              <motion.circle
                cx="200" cy="200" r={r2}
                fill="none" stroke="#739efa" strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray={circ2}
                animate={{ strokeDashoffset: offset2 }}
                transition={{ type: "tween", ease: "easeOut", duration: 0.5 }}
                filter="url(#fluidSmoke)"
                opacity={0.6}
                style={{ rotate: "-90deg", originX: '50%', originY: '50%' }}
              />
            </motion.g>

            {/* LAYER 3: The solid, bright core energy line */}
            <motion.circle
              cx="200" cy="200" r={r3}
              fill="none" stroke="#EBF0FF" strokeWidth="2"
              strokeLinecap="round"
              strokeDasharray={circ3}
              animate={{ strokeDashoffset: offset3 }}
              transition={{ type: "tween", ease: "easeOut", duration: 0.5 }}
              filter="url(#softGlow)"
              style={{ rotate: "-90deg", originX: '50%', originY: '50%' }}
            />
          </svg>
        </div>

        {/* The Line - The Anchor for the layout */}
        <div className="mb-8 w-[817px] h-px bg-gradient-to-r from-transparent via-[#2C3A50] to-transparent" />

        {/* Graph Controls */}
        <div className="flex flex-col items-center space-y-5 w-full">
          <h2 className="text-[#a9bde8] text-[18px] uppercase tracking-[0.2em] font-bold">
            Graph Controls
          </h2>

          <div className="flex flex-col items-start leading-relaxed space-y-2">
            <p className="text-[#EBF0FF] text-[15px] font-['Inter'] font-normal">
              <span className="font-bold text-[#a9bde8]">click</span> on a node for more details
            </p>
            <p className="text-[#EBF0FF] text-[15px] font-['Inter'] font-normal">
              <span className="font-bold text-[#a9bde8]">pinch</span> or <span className="font-bold text-[#a9bde8]">scroll</span> to zoom
            </p>
            <p className="text-[#EBF0FF] text-[15px] font-['Inter'] font-normal">
              <span className="font-bold text-[#a9bde8]">click track pad with one finger, move with another</span> to pan
            </p>
            <p className="text-[#EBF0FF] text-[15px] font-['Inter'] font-normal">
              <span className="font-bold text-[#a9bde8]">hold shift while panning</span> to rotate
            </p>
          </div>
        </div>
      </div>

      {/* Home Button */}
      <div className="absolute bottom-10 left-10 z-50">
        <button
          onClick={() => navigate('/')}
          className="relative flex items-center justify-center transition-all hover:brightness-125 active:scale-95 bg-[#121825]/60 backdrop-blur-xl border border-[#2C3A50]/50 shadow-xl cursor-pointer group"
          style={{ width: '66px', height: '66px', borderRadius: '26px' }}
        >
          <span className="absolute inset-0 transition-opacity duration-300 rounded-full opacity-0 blur-md bg-[#A9BDE8]/20 group-hover:opacity-100" />
          <svg
            xmlns="http://www.w3.org/2000/svg" width="41" height="41" viewBox="0 0 41 41" fill="none"
            className="relative z-10 transition-colors duration-200"
          >
            <path
              d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.9419 5.125 35.0729 5.125 34.1667V15.3751Z"
              className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]"
              strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
