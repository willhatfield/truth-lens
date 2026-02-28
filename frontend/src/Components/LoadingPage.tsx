import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

const loadingMessages = [
  "gathering information...",
  "consulting AIs...",
  "parsing through information...",
  "creating visuals...",
  "making everything pretty..."
];

export default function LoadingPage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const startTimeout = setTimeout(() => {
      const timer = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 50) {
            clearInterval(timer);
            setTimeout(() => navigate('/arena'), 1000);
            return 50;
          }
          return prev + 1;
        });
      }, 120);
      return () => clearInterval(timer);
    }, 500);
    return () => clearTimeout(startTimeout);
  }, [navigate]);

  const percentage = Math.round((progress / 50) * 100);
  const messageIndex = Math.min(
    Math.floor((progress / 50) * loadingMessages.length),
    loadingMessages.length - 1
  );

  // SVG Circle Math for the drawing animation
  const radius = 150;
  const circumference = 2 * Math.PI * radius;
  const drawLength = (percentage / 100) * circumference;
  const offset = circumference - drawLength;

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
              {loadingMessages[messageIndex]}
            </p>
          </motion.div>
        </div>

        {/* --- ETHEREAL FLUID ENERGY RING --- */}
        <div className="relative flex items-center justify-center w-[400px] h-[400px] mb-8">
          
          {/* Centered Percentage Display */}
          <div className="absolute flex flex-col items-center justify-center z-20">
            <motion.p 
              className="text-transparent bg-clip-text bg-gradient-to-b from-[#FFFFFF] to-[#A9BDE8]"
              style={{ fontFamily: 'Inter', fontSize: '72px', fontWeight: 300, letterSpacing: '-0.02em', filter: 'drop-shadow(0 0 15px rgba(169,189,232,0.3))' }}
            >
              {percentage}<span className="text-[32px] ml-1 opacity-50 font-light">%</span>
            </motion.p>
          </div>

          {/* SVG Complex Filters & Fluid Rings */}
          <svg width="400" height="400" viewBox="0 0 400 400" className="absolute inset-0 z-10 pointer-events-none overflow-visible">
            <defs>
              {/* The magical SVG filter that turns lines into organic smoke/fluid */}
              <filter id="fluidSmoke" x="-50%" y="-50%" width="200%" height="200%">
                <feTurbulence type="fractalNoise" baseFrequency="0.015" numOctaves="4" result="noise">
                  {/* Animates the noise over time so the smoke continuously morphs */}
                  <animate attributeName="baseFrequency" values="0.015; 0.02; 0.015" dur="15s" repeatCount="indefinite" />
                </feTurbulence>
                <feDisplacementMap in="SourceGraphic" in2="noise" scale="35" xChannelSelector="R" yChannelSelector="G" result="displaced" />
                <feGaussianBlur in="displaced" stdDeviation="4" result="blurred" />
                <feMerge>
                  <feMergeNode in="blurred" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>

              {/* Softer glow for the inner core line */}
              <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>

              {/* Gradient map for the rings */}
              <linearGradient id="bluePlasma" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#FFFFFF" />
                <stop offset="50%" stopColor="#A9BDE8" />
                <stop offset="100%" stopColor="#5273FB" />
              </linearGradient>
            </defs>

            {/* Faint background track */}
            <circle cx="200" cy="200" r={radius} fill="none" stroke="#1A2335" strokeWidth="1" opacity="0.3" />

            {/* LAYER 1: Thick, displaced outer smoke (Rotates Clockwise) */}
            {/* Using Accent Text (#A9BDE8) so the outer aura has a rich blue glow */}
            <motion.g animate={{ rotate: 360 }} transition={{ duration: 40, repeat: Infinity, ease: "linear" }} style={{ originX: '50%', originY: '50%' }}>
              <motion.circle
                cx="200" cy="200" r={radius + 8}
                fill="none" stroke="#A9BDE8" strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circumference}
                animate={{ strokeDashoffset: offset }}
                transition={{ type: "tween", ease: "easeOut", duration: 0.5 }}
                filter="url(#fluidSmoke)"
                opacity={0.5}
                style={{ rotate: "-90deg", originX: '50%', originY: '50%' }} 
              />
            </motion.g>

            {/* LAYER 2: Thinner, displaced inner smoke (Rotates Counter-Clockwise) */}
            {/* Using Code/Technical (#CCD8FF) for a brighter mid-layer */}
            <motion.g animate={{ rotate: -360 }} transition={{ duration: 30, repeat: Infinity, ease: "linear" }} style={{ originX: '50%', originY: '50%' }}>
              <motion.circle
                cx="200" cy="200" r={radius - 6}
                fill="none" stroke="#CCD8FF" strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray={circumference}
                animate={{ strokeDashoffset: offset }}
                transition={{ type: "tween", ease: "easeOut", duration: 0.5 }}
                filter="url(#fluidSmoke)"
                opacity={0.6}
                style={{ rotate: "-90deg", originX: '50%', originY: '50%' }}
              />
            </motion.g>

            {/* LAYER 3: The solid, bright core energy line */}
            {/* Using Primary Text (#EBF0FF) for the sharpest, hottest core */}
            <motion.circle
              cx="200" cy="200" r={radius}
              fill="none" stroke="#EBF0FF" strokeWidth="2"
              strokeLinecap="round"
              strokeDasharray={circumference}
              animate={{ strokeDashoffset: offset }}
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