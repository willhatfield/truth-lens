import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, ArrowUp } from 'lucide-react';
import { AnalysisResult } from '../types';

const TRUST_COLORS: Record<string, string> = {
  VerifiedSafe: '#00D68F',
  CautionUnverified: '#FFB020',
  Rejected: '#FF4757',
};

const MODEL_COLORS: Record<string, string> = {
  'GPT-4 (OpenAI)': '#10A37F',
  'Gemini (Google)': '#428F54',
  'Claude (Anthropic)': '#E8825A',
  'Llama 3 (Meta)': '#A8555F',
  'Kimi (Moonshot)': '#5273FB',
};

const MODELS = Object.keys(MODEL_COLORS);

interface HeatmapProps {
  selectedModels: string[];
  result: AnalysisResult | null;
}

const generateScatterClaims = () => {
  const baseClaims = [
    { text: "Intermittent fasting reduces insulin resistance by 15-30%.", consensus: 4, confidence: 92, status: 'VerifiedSafe' },
    { text: "OpenAI was founded in December 2015.", consensus: 5, confidence: 98, status: 'VerifiedSafe' },
    { text: "Quantum entanglement allows information transfer faster than light.", consensus: 2, confidence: 15, status: 'Rejected' },
    { text: "The Paris Agreement aims to limit global warming to 1.5°C.", consensus: 5, confidence: 85, status: 'VerifiedSafe' },
    { text: "Python is the most popular language for machine learning.", consensus: 3, confidence: 70, status: 'CautionUnverified' },
    { text: "Humans only use 10% of their brains.", consensus: 1, confidence: 5, status: 'Rejected' },
    { text: "Mars has two moons: Phobos and Deimos.", consensus: 5, confidence: 95, status: 'VerifiedSafe' },
    { text: "Artificial General Intelligence will be achieved by 2027.", consensus: 2, confidence: 45, status: 'CautionUnverified' },
    { text: "Cold fusion has been successfully demonstrated in a lab.", consensus: 1, confidence: 10, status: 'Rejected' },
    { text: "Vitamin C cures the common cold.", consensus: 2, confidence: 25, status: 'Rejected' },
    { text: "The speed of light is exactly 299,792,458 meters per second.", consensus: 4, confidence: 99, status: 'VerifiedSafe' },
  ];

  return baseClaims.map((claim, idx) => {
    const shuffledModels = [...MODELS].sort(() => 0.5 - Math.random());
    const agreedModels = shuffledModels.slice(0, claim.consensus);
    
    const jitterX = (Math.random() - 0.5) * 0.4;
    const jitterY = (Math.random() - 0.5) * 6;

    return {
      id: `scatter-claim-${idx}`,
      ...claim,
      plotX: Math.min(Math.max(claim.consensus + jitterX, 1), 5), 
      plotY: Math.min(Math.max(claim.confidence + jitterY, 5), 95), 
      agreedModels
    };
  });
};

export default function Heatmap({ selectedModels, result }: HeatmapProps) {
  const claims = useMemo(() => {
    if (!result) return generateScatterClaims();
    const { cluster_scores, clusters } = result;
    return cluster_scores.map((score, idx) => {
      const cluster = clusters.find(c => c.cluster_id === score.cluster_id);
      const verdict = score.verdict;
      const status = verdict === 'SAFE' ? 'VerifiedSafe' : verdict === 'CAUTION' ? 'CautionUnverified' : 'Rejected';
      const jitterX = (Math.random() - 0.5) * 0.3;
      const jitterY = (Math.random() - 0.5) * 4;
      return {
        id: `cluster-${idx}`,
        text: cluster?.representative_text ?? score.cluster_id,
        consensus: score.agreement.count,
        confidence: Math.round(score.trust_score),
        status,
        plotX: Math.min(Math.max(score.agreement.count + jitterX, 1), 5),
        plotY: Math.min(Math.max(score.trust_score + jitterY, 5), 95),
        agreedModels: score.agreement.models_supporting,
      };
    });
  }, [result]);
  const [hoveredClaim, setHoveredClaim] = useState<string | null>(null);

  const visibleClaims = claims.filter(claim => 
    claim.agreedModels.some(model => selectedModels.includes(model))
  );

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-8 overflow-hidden bg-[#0A0E1A]">
      
      {/* Changed background to a pure dark #0A0A0E to kill the blue tint */}
      <div className="relative w-full max-w-5xl aspect-[16/9] rounded-2xl border border-[#2C3A50]/50 bg-[#0A0A0E] shadow-2xl p-16 pl-20 pb-20">
        
        {/* --- AXES LABELS --- */}
        <div className="absolute top-8 bottom-20 left-6 flex flex-col items-center justify-center">
          <ArrowUp className="w-4 h-4 text-[#588983] mb-2" />
          <div className="w-px h-full bg-gradient-to-t from-transparent to-[#588983]/80" />
          <span className="text-[#A9BDE8] text-[10px] font-bold tracking-[0.2em] uppercase -rotate-90 absolute whitespace-nowrap" style={{ top: '50%' }}>
            Verification Confidence
          </span>
        </div>

        <div className="absolute bottom-6 left-20 right-16 flex items-center justify-center">
          <span className="text-[#A9BDE8] text-[10px] font-bold tracking-[0.2em] uppercase absolute" style={{ left: '50%', transform: 'translateX(-50%)' }}>
            Model Consensus
          </span>
          <div className="w-full h-px bg-gradient-to-r from-transparent to-[#588983]/80" />
          <ArrowRight className="w-4 h-4 text-[#588983] ml-2" />
        </div>

        <div className="relative w-full h-full rounded-xl overflow-visible">
          
          {/* --- QUADRANT BACKGROUND GRADIENTS (High Intensity) --- */}
          <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none opacity-70 mix-blend-screen">
            {/* Top Right: Safe Zone */}
            <div className="absolute top-0 right-0 w-[70%] h-[70%] bg-[radial-gradient(ellipse_at_top_right,_#00D68F50_0%,_transparent_60%)]" />
            {/* Bottom Left: Danger Zone */}
            <div className="absolute bottom-0 left-0 w-[70%] h-[70%] bg-[radial-gradient(ellipse_at_bottom_left,_#FF475750_0%,_transparent_60%)]" />
            {/* Top Left: Niche */}
            <div className="absolute top-0 left-0 w-[50%] h-[50%] bg-[radial-gradient(ellipse_at_top_left,_#FFB02030_0%,_transparent_60%)]" />
            {/* Bottom Right: Contested */}
            <div className="absolute bottom-0 right-0 w-[50%] h-[50%] bg-[radial-gradient(ellipse_at_bottom_right,_#FFB02030_0%,_transparent_60%)]" />
          </div>

          {/* --- GLOWING EDGE BORDERS --- */}
          {/* Top & Right Edge Glow (Green) */}
          <div className="absolute inset-0 rounded-xl border-t-2 border-r-2 border-[#00D68F]/40 shadow-[inset_-10px_10px_20px_rgba(0,214,143,0.1)] pointer-events-none" style={{ maskImage: 'linear-gradient(to bottom left, black 20%, transparent 60%)', WebkitMaskImage: 'linear-gradient(to bottom left, black 20%, transparent 60%)' }} />
          {/* Bottom & Left Edge Glow (Red) */}
          <div className="absolute inset-0 rounded-xl border-b-2 border-l-2 border-[#FF4757]/40 shadow-[inset_10px_-10px_20px_rgba(255,71,87,0.1)] pointer-events-none" style={{ maskImage: 'linear-gradient(to top right, black 20%, transparent 60%)', WebkitMaskImage: 'linear-gradient(to top right, black 20%, transparent 60%)' }} />

          {/* --- ENVIRONMENTAL TYPOGRAPHY --- */}
          <span className="absolute top-5 right-5 text-[#00D68F]/40 font-black text-2xl tracking-widest uppercase pointer-events-none select-none">Verified Consensus</span>
          <span className="absolute bottom-5 left-5 text-[#FF4757]/40 font-black text-2xl tracking-widest uppercase pointer-events-none select-none">Rejected Hallucinations</span>
          <span className="absolute top-5 left-5 text-[#FFB020]/30 font-bold text-lg tracking-widest uppercase pointer-events-none select-none">Niche Claims</span>
          <span className="absolute bottom-5 right-5 text-[#FFB020]/30 font-bold text-lg tracking-widest uppercase pointer-events-none select-none">Highly Contested</span>

          {/* Grid Lines (Neutralized to reduce blue) */}
          <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 rounded-xl border border-[#3A4B66]/30 pointer-events-none">
            {[...Array(4)].map((_, i) => (
              <div key={`v-${i}`} className="absolute top-0 bottom-0 border-r border-[#3A4B66]/30" style={{ left: `${(i + 1) * 20}%` }}>
                <span className="absolute -bottom-8 left-0 -translate-x-1/2 text-[#588983] text-xs font-mono">{i + 1}</span>
              </div>
            ))}
            <div className="absolute top-0 bottom-0 right-0">
                <span className="absolute -bottom-8 left-0 -translate-x-1/2 text-[#588983] text-xs font-mono">5</span>
            </div>
            
            {[...Array(3)].map((_, i) => (
              <div key={`h-${i}`} className="absolute left-0 right-0 border-b border-[#3A4B66]/30" style={{ top: `${(i + 1) * 25}%` }}>
                <span className="absolute -left-8 top-0 -translate-y-1/2 text-[#588983] text-xs font-mono">{75 - (i * 25)}%</span>
              </div>
            ))}
          </div>

          {/* --- AMBIENT FLOATING DATA POINTS --- */}
          {visibleClaims.map((claim, i) => {
            const leftPos = `${((claim.plotX - 1) / 4) * 100}%`;
            const bottomPos = `${claim.plotY}%`;
            const isHovered = hoveredClaim === claim.id;
            const color = TRUST_COLORS[claim.status];

            const showBelow = claim.plotY > 70;

            return (
              <motion.div
                key={claim.id}
                className={`absolute ${isHovered ? 'z-50' : 'z-10'}`}
                style={{ left: leftPos, bottom: bottomPos }}
                onMouseEnter={() => setHoveredClaim(claim.id)}
                onMouseLeave={() => setHoveredClaim(null)}
                animate={{ y: [0, -8, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: i * 0.2 }}
              >
                <motion.div 
                  initial={{ scale: 0 }}
                  animate={{ scale: isHovered ? 1.2 : 1 }}
                  className="flex items-center justify-center w-7 h-7 -ml-3.5 -mb-3.5 rounded-full cursor-pointer shadow-[0_0_15px_rgba(0,0,0,0.8)] bg-[#1A1215] text-white text-[11px] font-bold"
                  style={{ border: `2px solid ${color}`, boxShadow: isHovered ? `0 0 20px ${color}80` : 'none' }}
                >
                  {claim.consensus}
                </motion.div>

                <AnimatePresence>
                  {isHovered && (
                    <motion.div
                      initial={{ opacity: 0, y: showBelow ? -10 : 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: showBelow ? -5 : 5, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className={`absolute left-1/2 -translate-x-1/2 w-80 p-5 rounded-xl bg-[#1A2335]/95 backdrop-blur-md border border-[#2C3A50] shadow-2xl pointer-events-none 
                      ${showBelow ? 'top-10' : 'bottom-10'}`} 
                    >
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-bold text-[#90A2B3] uppercase tracking-wider">Claim</span>
                        <span className="text-sm font-bold px-2.5 py-1 rounded-md" style={{ backgroundColor: `${color}20`, color: color }}>
                          {claim.confidence}% Verified
                        </span>
                      </div>
                      
                      <p className="text-[#EBF0FF] text-[15px] leading-snug mb-4">
                        "{claim.text}"
                      </p>
                      
                      <div className="pt-3 border-t border-[#2C3A50]/50 flex flex-wrap gap-2">
                        {claim.agreedModels.map(model => {
                          const isModelActive = selectedModels.includes(model);
                          return (
                            <span 
                              key={model} 
                              className={`text-[10px] uppercase tracking-widest font-bold px-2 py-1 rounded transition-opacity ${isModelActive ? 'opacity-100' : 'opacity-30 line-through'}`}
                              style={{ backgroundColor: `${MODEL_COLORS[model]}20`, color: MODEL_COLORS[model], border: `1px solid ${MODEL_COLORS[model]}40` }}
                            >
                              {model.split(' ')[0]}
                            </span>
                          );
                        })}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}