import { motion } from 'framer-motion';
import { 
  FileText, Database, 
  Network, ShieldCheck, ShieldAlert, 
  CheckCircle2, Trash2 
} from 'lucide-react';
import React from 'react';

interface PipelineProps {
  selectedModels: string[];
}

const MODEL_COLORS: Record<string, string> = {
  'GPT-4 (OpenAI)': '#10A37F',
  'Gemini (Google)': '#428F54', 
  'Claude (Anthropic)': '#E8825A',
  'Llama 3 (Meta)': '#A8555F',
  'Kimi (Moonshot)': '#5273FB'
};

// --- EXPANDED COORDINATE SYSTEM (1200x650) ---
// X-axis centers
const X = { input: 100, models: 300, extract: 500, cluster: 700, verify: 900, answer: 1100 };
// Y-axis centers
const Y = { r1: 120, r2: 210, center: 300, r4: 390, r5: 480, cTop: 165, cBot: 435, trash: 580 };

// --- BOUNDING BOX CALCULATIONS ---
// Used to stop the SVG lines exactly on the edge of the HTML nodes so they don't bleed inside
const P = {
  inR: 124,         // Input Right edge
  modL: 245,        // Models Left edge
  modR: 355,        // Models Right edge
  extL: 480,        // Extract Left edge
  extR: 520,        // Extract Right edge
  cluL: 660,        // Cluster Left edge
  cluR: 740,        // Cluster Right edge
  verL: 876,        // Verify Left edge
  verR: 924,        // Verify Right edge
  ansL: 1030,       // Answer Left edge
  verBot: 459,      // Verify Bottom edge
  trashTop: 548     // Trash Top edge
};

// Generates smooth Sankey-style S-curves with guaranteed horizontal entry/exit
const createSankeyPath = (x1: number, y1: number, x2: number, y2: number) => {
  if (y1 === y2) return `M ${x1} ${y1} L ${x2} ${y2}`; 
  const cpOffset = (x2 - x1) * 0.45; // Controls the curve steepness
  return `M ${x1} ${y1} C ${x1 + cpOffset} ${y1}, ${x2 - cpOffset} ${y2}, ${x2} ${y2}`;
};

export default function Pipeline({ selectedModels }: PipelineProps) {
  
  const getX = (val: number) => `${(val / 1200) * 100}%`;
  const getY = (val: number) => `${(val / 650) * 100}%`;

  const FlowRiver = ({ x1, y1, x2, y2, color, isRejected = false, isActive = true, isDrop = false }: any) => {
    if (!isActive) return null;
    
    // Drop paths use a straight vertical line, everything else uses the S-curve
    const path = isDrop 
      ? `M ${x1} ${y1} L ${x1} ${y2}` 
      : createSankeyPath(x1, y1, x2, y2);
    
    const strokeColor = isRejected ? '#FF4757' : color;
    
    return (
      <g>
        {/* Outer faint glow volume */}
        <path d={path} stroke={strokeColor} strokeWidth="16" fill="none" opacity={isRejected ? 0.05 : 0.15} />
        {/* Solid core line */}
        <path d={path} stroke={strokeColor} strokeWidth="2" fill="none" opacity={isRejected ? 0.2 : 0.4} />
        {/* Animated dashes */}
        <motion.path 
          d={path} 
          stroke={strokeColor} 
          strokeWidth={isRejected ? 2 : 4} 
          fill="none" 
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 6px ${strokeColor})` }}
          strokeDasharray="6 35"
          animate={{ strokeDashoffset: [41, 0] }}
          transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
          opacity={isRejected ? 0.4 : 1}
        />
      </g>
    );
  };

  const StageHeader = ({ x, title, subtitle }: { x: number, title: string, subtitle: string }) => (
    <div className="absolute top-6 -translate-x-1/2 flex flex-col items-center z-20" style={{ left: getX(x) }}>
      <span className="text-[#EBF0FF] text-[11px] font-bold tracking-[0.15em] uppercase">{title}</span>
      <span className="text-[#588983] text-[9px] font-medium tracking-widest uppercase mt-1">{subtitle}</span>
    </div>
  );

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-8 bg-[#0A0E1A] overflow-hidden">
      
      <div className="relative w-full max-w-[1400px] aspect-[1200/650] bg-[#121825] border border-[#2C3A50] rounded-3xl shadow-2xl overflow-hidden">
        
        <StageHeader x={X.input} title="Input" subtitle="User Prompt" />
        <StageHeader x={X.models} title="5 Models" subtitle="Parallel Gen" />
        <StageHeader x={X.extract} title="Extraction" subtitle="Atomic Claims" />
        <StageHeader x={X.cluster} title="Clustering" subtitle="Semantic NLI" />
        <StageHeader x={X.verify} title="Verification" subtitle="Filter Gate" />
        <StageHeader x={X.answer} title="Synthesis" subtitle="Safe Answer" />

        {/* --- SVG DRAWING LAYER (Flows perfectly from edge to edge) --- */}
        <svg viewBox="0 0 1200 650" className="absolute inset-0 w-full h-full z-10 pointer-events-none">
          
          {/* 1. INPUT TO MODELS */}
          <FlowRiver x1={P.inR} y1={Y.center} x2={P.modL} y2={Y.r1} color={MODEL_COLORS['GPT-4 (OpenAI)']} isActive={selectedModels.includes('GPT-4 (OpenAI)')} />
          <FlowRiver x1={P.inR} y1={Y.center} x2={P.modL} y2={Y.r2} color={MODEL_COLORS['Gemini (Google)']} isActive={selectedModels.includes('Gemini (Google)')} />
          <FlowRiver x1={P.inR} y1={Y.center} x2={P.modL} y2={Y.center} color={MODEL_COLORS['Claude (Anthropic)']} isActive={selectedModels.includes('Claude (Anthropic)')} />
          <FlowRiver x1={P.inR} y1={Y.center} x2={P.modL} y2={Y.r4} color={MODEL_COLORS['Llama 3 (Meta)']} isActive={selectedModels.includes('Llama 3 (Meta)')} />
          <FlowRiver x1={P.inR} y1={Y.center} x2={P.modL} y2={Y.r5} color={MODEL_COLORS['Kimi (Moonshot)']} isActive={selectedModels.includes('Kimi (Moonshot)')} />

          {/* 2. MODELS TO EXTRACTION */}
          <FlowRiver x1={P.modR} y1={Y.r1} x2={P.extL} y2={Y.r1} color={MODEL_COLORS['GPT-4 (OpenAI)']} isActive={selectedModels.includes('GPT-4 (OpenAI)')} />
          <FlowRiver x1={P.modR} y1={Y.r2} x2={P.extL} y2={Y.r2} color={MODEL_COLORS['Gemini (Google)']} isActive={selectedModels.includes('Gemini (Google)')} />
          <FlowRiver x1={P.modR} y1={Y.center} x2={P.extL} y2={Y.center} color={MODEL_COLORS['Claude (Anthropic)']} isActive={selectedModels.includes('Claude (Anthropic)')} />
          <FlowRiver x1={P.modR} y1={Y.r4} x2={P.extL} y2={Y.r4} color={MODEL_COLORS['Llama 3 (Meta)']} isActive={selectedModels.includes('Llama 3 (Meta)')} />
          <FlowRiver x1={P.modR} y1={Y.r5} x2={P.extL} y2={Y.r5} color={MODEL_COLORS['Kimi (Moonshot)']} isActive={selectedModels.includes('Kimi (Moonshot)')} />

          {/* 3. EXTRACTION TO CLUSTERING (Re-routed to avoid overlaps) */}
          <FlowRiver x1={P.extR} y1={Y.r1} x2={P.cluL} y2={Y.cTop} color={MODEL_COLORS['GPT-4 (OpenAI)']} isActive={selectedModels.includes('GPT-4 (OpenAI)')} />
          <FlowRiver x1={P.extR} y1={Y.r2} x2={P.cluL} y2={Y.cTop} color={MODEL_COLORS['Gemini (Google)']} isActive={selectedModels.includes('Gemini (Google)')} />
          
          <FlowRiver x1={P.extR} y1={Y.center} x2={P.cluL} y2={Y.center} color={MODEL_COLORS['Claude (Anthropic)']} isActive={selectedModels.includes('Claude (Anthropic)')} />
          <FlowRiver x1={P.extR} y1={Y.r5} x2={P.cluL} y2={Y.center} color={MODEL_COLORS['Kimi (Moonshot)']} isActive={selectedModels.includes('Kimi (Moonshot)')} />

          {/* Llama's Hallucination is now completely isolated at the bottom */}
          <FlowRiver x1={P.extR} y1={Y.r4} x2={P.cluL} y2={Y.cBot} color={MODEL_COLORS['Llama 3 (Meta)']} isActive={selectedModels.includes('Llama 3 (Meta)')} />

          {/* 4. CLUSTERING TO VERIFICATION */}
          <FlowRiver x1={P.cluR} y1={Y.cTop} x2={P.verL} y2={Y.cTop} color="#00D68F" />
          <FlowRiver x1={P.cluR} y1={Y.center} x2={P.verL} y2={Y.center} color="#00D68F" />
          <FlowRiver x1={P.cluR} y1={Y.cBot} x2={P.verL} y2={Y.cBot} color="#FFB020" />

          {/* 5. VERIFICATION TO SYNTHESIS OR TRASH */}
          <FlowRiver x1={P.verR} y1={Y.cTop} x2={P.ansL} y2={Y.center} color="#00D68F" />
          <FlowRiver x1={P.verR} y1={Y.center} x2={P.ansL} y2={Y.center} color="#00D68F" />
          
          {/* Drop down to filtered (Perfect vertical drop, zero overlap) */}
          <FlowRiver x1={X.verify} y1={P.verBot} x2={X.verify} y2={P.trashTop} color="#FF4757" isRejected={true} isDrop={true} />
        </svg>

        {/* --- HTML NODES LAYER (Solid blocks that sit over the lines) --- */}
        
        {/* INPUT */}
        <div className="absolute -translate-x-1/2 -translate-y-1/2 bg-[#1E2738] p-4 rounded-2xl border border-[#3A4B66] z-20 shadow-[0_0_15px_rgba(0,0,0,0.5)]" style={{ left: getX(X.input), top: getY(Y.center) }}>
          <FileText className="w-6 h-6 text-[#A9BDE8]" />
        </div>

        {/* MODELS & EXTRACTION */}
        {[
          { y: Y.r1, name: 'GPT-4', modelKey: 'GPT-4 (OpenAI)' },
          { y: Y.r2, name: 'Gemini', modelKey: 'Gemini (Google)' },
          { y: Y.center, name: 'Claude', modelKey: 'Claude (Anthropic)' },
          { y: Y.r4, name: 'Llama 3', modelKey: 'Llama 3 (Meta)' },
          { y: Y.r5, name: 'Mistral', modelKey: 'Kimi (Moonshot)' }
        ].map((m) => {
          const isActive = selectedModels.includes(m.modelKey);
          return (
            <div key={m.name} className={`transition-opacity duration-300 ${isActive ? 'opacity-100' : 'opacity-20 grayscale'}`}>
              <div className="absolute -translate-x-1/2 -translate-y-1/2 px-6 py-2 bg-[#1E2738] rounded-full border border-[#3A4B66] z-20 shadow-lg w-[110px] flex justify-center" style={{ left: getX(X.models), top: getY(m.y) }}>
                <span className="text-[12px] font-bold" style={{ color: MODEL_COLORS[m.modelKey] }}>{m.name}</span>
              </div>
              <div className="absolute -translate-x-1/2 -translate-y-1/2 bg-[#1E2738] p-2.5 rounded-full border border-[#3A4B66] z-20 shadow-lg w-10 flex justify-center" style={{ left: getX(X.extract), top: getY(m.y) }}>
                <Database className="w-4 h-4 text-[#90A2B3]" />
              </div>
            </div>
          );
        })}

        {/* CLUSTERING NODES */}
        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center w-20 h-20 bg-[#1E2738] rounded-full border border-[#3A4B66] z-20 shadow-lg" style={{ left: getX(X.cluster), top: getY(Y.cTop) }}>
          <Network className="w-5 h-5 text-[#A9BDE8] mb-1" />
          <span className="text-[9px] text-[#EBF0FF] font-bold">Consensus</span>
        </div>
        
        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center w-20 h-20 bg-[#1E2738] rounded-full border border-[#3A4B66] z-20 shadow-lg" style={{ left: getX(X.cluster), top: getY(Y.center) }}>
          <Network className="w-5 h-5 text-[#A9BDE8] mb-1" />
          <span className="text-[9px] text-[#EBF0FF] font-bold">Consensus</span>
        </div>

        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center w-20 h-20 bg-[#1E2738] rounded-full border border-[#FFB020]/40 z-20 shadow-lg" style={{ left: getX(X.cluster), top: getY(Y.cBot) }}>
          <span className="text-[10px] text-[#FFB020] font-bold text-center leading-tight">Solo<br/>Claim</span>
        </div>

        {/* VERIFICATION GATES */}
        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex items-center justify-center w-12 h-12 bg-[#161D2E] rounded-xl border border-[#00D68F]/60 z-20 shadow-[0_0_20px_rgba(0,214,143,0.15)]" style={{ left: getX(X.verify), top: getY(Y.cTop) }}>
          <ShieldCheck className="w-6 h-6 text-[#00D68F]" />
        </div>
        
        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex items-center justify-center w-12 h-12 bg-[#161D2E] rounded-xl border border-[#00D68F]/60 z-20 shadow-[0_0_20px_rgba(0,214,143,0.15)]" style={{ left: getX(X.verify), top: getY(Y.center) }}>
          <ShieldCheck className="w-6 h-6 text-[#00D68F]" />
        </div>

        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex items-center justify-center w-12 h-12 bg-[#161D2E] rounded-xl border border-[#FF4757]/60 z-20 shadow-[0_0_20px_rgba(255,71,87,0.15)]" style={{ left: getX(X.verify), top: getY(Y.cBot) }}>
          <ShieldAlert className="w-6 h-6 text-[#FF4757]" />
        </div>

        {/* REJECTED BUCKET */}
        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center w-20 h-16 bg-[#1A1215] rounded-xl border border-[#FF4757]/30 z-20 shadow-lg" style={{ left: getX(X.verify), top: getY(Y.trash) }}>
          <Trash2 className="w-5 h-5 text-[#FF4757]" />
          <span className="text-[9px] text-[#FF4757] font-bold tracking-widest mt-1.5">FILTERED</span>
        </div>

        {/* SAFE ANSWER */}
        <div className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center w-[140px] h-[140px] bg-[#121825] rounded-[2rem] border-2 border-[#00D68F]/50 z-20 shadow-[0_0_40px_rgba(0,214,143,0.2)] overflow-hidden" style={{ left: getX(X.answer), top: getY(Y.center) }}>
          <div className="absolute inset-0 bg-gradient-to-br from-[#00D68F]/20 to-transparent z-0" />
          <CheckCircle2 className="w-10 h-10 text-[#00D68F] mb-3 z-10" />
          <span className="text-[13px] font-bold text-[#EBF0FF] uppercase tracking-widest text-center leading-tight z-10">Verified<br/>Answer</span>
        </div>

      </div>
    </div>
  );
}