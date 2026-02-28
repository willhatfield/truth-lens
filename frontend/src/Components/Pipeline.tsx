import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, ShieldAlert, FileText, Cpu, Database, CheckCircle2 } from 'lucide-react';

interface PipelineProps {
  selectedModels: string[];
}

const MODEL_COLORS = {
  'GPT-4': '#10A37F',
  'Gemini': '#428F54', 
  'Claude': '#E8825A',
  'Llama': '#A855F7',
  'Mistral': '#F59E0B'
};

// SVG Coordinate System (1000x500)
const COL_X = { input: 60, models: 240, extract: 420, cluster: 600, verify: 780, answer: 940 };
const ROW_Y = { r1: 50, r2: 150, center: 250, r4: 350, r5: 450, c_top: 100, c_bot: 400 };

const PATHS = [
  { id: 'gpt4', color: MODEL_COLORS['GPT-4'], points: [ [COL_X.input, ROW_Y.center], [COL_X.models, ROW_Y.r1], [COL_X.extract, ROW_Y.r1], [COL_X.cluster, ROW_Y.c_top], [COL_X.verify, ROW_Y.c_top], [COL_X.answer, ROW_Y.center] ] },
  { id: 'gemini', color: MODEL_COLORS['Gemini'], points: [ [COL_X.input, ROW_Y.center], [COL_X.models, ROW_Y.r2], [COL_X.extract, ROW_Y.r2], [COL_X.cluster, ROW_Y.c_top], [COL_X.verify, ROW_Y.c_top], [COL_X.answer, ROW_Y.center] ] },
  { id: 'claude', color: MODEL_COLORS['Claude'], points: [ [COL_X.input, ROW_Y.center], [COL_X.models, ROW_Y.center], [COL_X.extract, ROW_Y.center], [COL_X.cluster, ROW_Y.center], [COL_X.verify, ROW_Y.center] ] },
  { id: 'llama', color: MODEL_COLORS['Llama'], points: [ [COL_X.input, ROW_Y.center], [COL_X.models, ROW_Y.r4], [COL_X.extract, ROW_Y.r4], [COL_X.cluster, ROW_Y.c_bot], [COL_X.verify, ROW_Y.c_bot], [COL_X.answer, ROW_Y.center] ] },
  { id: 'mistral', color: MODEL_COLORS['Mistral'], points: [ [COL_X.input, ROW_Y.center], [COL_X.models, ROW_Y.r5], [COL_X.extract, ROW_Y.r5], [COL_X.cluster, ROW_Y.c_bot], [COL_X.verify, ROW_Y.c_bot], [COL_X.answer, ROW_Y.center] ] },
];

// Helper to generate smooth bezier curves instead of harsh straight lines
const makeSmoothCurve = (points: number[][]) => {
  let d = `M ${points[0][0]},${points[0][1]} `;
  for (let i = 1; i < points.length; i++) {
    const [x1, y1] = points[i - 1];
    const [x2, y2] = points[i];
    // Control points halfway between X coordinates to ensure horizontal entry/exit
    const ctrlX = (x1 + x2) / 2;
    d += `C ${ctrlX},${y1} ${ctrlX},${y2} ${x2},${y2} `;
  }
  return d;
};

const FlowingStream = ({ path }: { path: typeof PATHS[0] }) => {
  const duration = (path.points.length - 1) * 1.5;

  return (
    <>
      <path d={makeSmoothCurve(path.points)} stroke="#2C3A50" strokeWidth="2" fill="none" />
      {[0, 0.33, 0.66].map((delayFactor, i) => (
        <motion.circle
          key={`${path.id}-${i}`}
          r="4"
          fill={path.color}
          style={{ filter: `drop-shadow(0 0 6px ${path.color})`, offsetPath: `path('${makeSmoothCurve(path.points)}')` }}
          animate={{ offsetDistance: ["0%", "100%"] }}
          transition={{
            duration: duration,
            ease: "linear",
            repeat: Infinity,
            delay: delayFactor * duration
          }}
        />
      ))}
    </>
  );
};

export default function Pipeline({ selectedModels }: PipelineProps) {
  const getPos = (val: number, isX: boolean) => `${(val / (isX ? 1000 : 500)) * 100}%`;

  const BaseNode = ({ x, y, children, className = "" }: { x: number, y: number, children: React.ReactNode, className?: string }) => (
    <div 
      className={`absolute flex items-center justify-center -translate-x-1/2 -translate-y-1/2 bg-[#1A2335] border border-[#2C3A50] shadow-lg z-10 ${className}`}
      style={{ left: getPos(x, true), top: getPos(y, false) }}
    >
      {children}
    </div>
  );

  return (
    <div className="flex flex-col items-center w-full h-full p-8 overflow-hidden bg-[#0A0E1A]">

      <div className="relative w-full max-w-6xl flex-1 bg-[#121825] border border-[#2C3A50] rounded-2xl shadow-2xl p-4 overflow-hidden">
        
        <div className="absolute top-6 left-0 right-0 h-10 flex text-[#90A2B3] text-[10px] font-bold tracking-widest uppercase z-20">
          <div className="absolute -translate-x-1/2 text-center" style={{ left: getPos(COL_X.input, true) }}>Input</div>
          <div className="absolute -translate-x-1/2 text-center" style={{ left: getPos(COL_X.models, true) }}>5 Models</div>
          <div className="absolute -translate-x-1/2 text-center w-24" style={{ left: getPos(COL_X.extract, true) }}>Claim Extraction</div>
          <div className="absolute -translate-x-1/2 text-center" style={{ left: getPos(COL_X.cluster, true) }}>Clustering</div>
          <div className="absolute -translate-x-1/2 text-center" style={{ left: getPos(COL_X.verify, true) }}>Verification</div>
          <div className="absolute -translate-x-1/2 text-center w-24" style={{ left: getPos(COL_X.answer, true) }}>Safe Answer</div>
        </div>

        <svg viewBox="0 0 1000 500" className="absolute inset-0 w-full h-full pointer-events-none z-0">
          {PATHS.map(path => {
            const isActive = selectedModels.some(sm => sm.includes(path.id === 'gpt4' ? 'GPT-4' : path.id === 'gemini' ? 'Gemini' : path.id === 'claude' ? 'Claude' : path.id === 'llama' ? 'Llama' : 'Kimi'));
            return isActive ? <FlowingStream key={path.id} path={path} /> : null;
          })}
        </svg>

        {/* Input */}
        <BaseNode x={COL_X.input} y={ROW_Y.center} className="w-12 h-12 rounded-xl">
          <FileText className="w-5 h-5 text-[#90A2B3]" />
        </BaseNode>

        {/* Models & Extraction */}
        {[
          { y: ROW_Y.r1, name: 'GPT-4', color: MODEL_COLORS['GPT-4'] },
          { y: ROW_Y.r2, name: 'Gemini', color: MODEL_COLORS['Gemini'] },
          { y: ROW_Y.center, name: 'Claude', color: MODEL_COLORS['Claude'] },
          { y: ROW_Y.r4, name: 'Llama 3', color: MODEL_COLORS['Llama'] },
          { y: ROW_Y.r5, name: 'Mistral', color: MODEL_COLORS['Mistral'] }
        ].map((m) => (
          <div key={m.name}>
            <BaseNode x={COL_X.models} y={m.y} className="w-28 h-10 rounded-xl">
              <span className="text-xs font-bold" style={{ color: m.color }}>{m.name}</span>
            </BaseNode>
            <BaseNode x={COL_X.extract} y={m.y} className="w-8 h-8 rounded-md bg-[#0A0E1A]">
              <Database className="w-3 h-3 text-[#90A2B3]" />
            </BaseNode>
          </div>
        ))}

        {/* Clustering */}
        <BaseNode x={COL_X.cluster} y={ROW_Y.c_top} className="w-14 h-14 rounded-full bg-[#1A2335] border-[#5E6E81]">
          <span className="text-[10px] text-[#EBF0FF] font-bold text-center leading-tight">Agreed<br/>Cluster</span>
        </BaseNode>
        <BaseNode x={COL_X.cluster} y={ROW_Y.center} className="w-12 h-12 rounded-full bg-[#0A0E1A] opacity-70">
          <span className="text-[9px] text-[#90A2B3] font-bold text-center leading-tight">Solo<br/>Claim</span>
        </BaseNode>
        <BaseNode x={COL_X.cluster} y={ROW_Y.c_bot} className="w-14 h-14 rounded-full bg-[#1A2335] border-[#5E6E81]">
          <span className="text-[10px] text-[#EBF0FF] font-bold text-center leading-tight">Agreed<br/>Cluster</span>
        </BaseNode>

        {/* Verification */}
        <BaseNode x={COL_X.verify} y={ROW_Y.c_top} className="w-10 h-10 rounded-xl border-[#00D68F] bg-[#00D68F]/10">
          <Shield className="w-4 h-4 text-[#00D68F]" />
        </BaseNode>
        <BaseNode x={COL_X.verify} y={ROW_Y.center} className="w-10 h-10 rounded-xl border-[#FF4757] bg-[#FF4757]/10">
          <ShieldAlert className="w-4 h-4 text-[#FF4757]" />
        </BaseNode>
        <BaseNode x={COL_X.verify} y={ROW_Y.c_bot} className="w-10 h-10 rounded-xl border-[#00D68F] bg-[#00D68F]/10">
          <Shield className="w-4 h-4 text-[#00D68F]" />
        </BaseNode>

        {/* Safe Answer */}
        <BaseNode x={COL_X.answer} y={ROW_Y.center} className="w-24 h-24 rounded-2xl border-[#00D68F] bg-[#00D68F]/5 shadow-[0_0_30px_rgba(0,214,143,0.15)] flex-col gap-2">
          <CheckCircle2 className="w-8 h-8 text-[#00D68F]" />
          <span className="text-[10px] font-bold text-[#EBF0FF] uppercase tracking-wider text-center">Verified<br/>Output</span>
        </BaseNode>

      </div>
    </div>
  );
}