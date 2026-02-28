import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// TruthLens Palette
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
}

// Generate enriched mock data for the scatter plot
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
  ];

  return baseClaims.map((claim, idx) => {
    // Randomly assign which specific models agreed based on the consensus count
    const shuffledModels = [...MODELS].sort(() => 0.5 - Math.random());
    const agreedModels = shuffledModels.slice(0, claim.consensus);
    
    // Add a tiny bit of random jitter so points don't perfectly overlap
    const jitterX = (Math.random() - 0.5) * 0.5;
    const jitterY = (Math.random() - 0.5) * 5;

    return {
      id: `scatter-claim-${idx}`,
      ...claim,
      plotX: Math.min(Math.max(claim.consensus + jitterX, 1), 5), // Keep between 1 and 5
      plotY: Math.min(Math.max(claim.confidence + jitterY, 5), 95), // Keep between 5 and 95
      agreedModels
    };
  });
};

export default function Heatmap({ selectedModels }: HeatmapProps) {
  const claims = useMemo(() => generateScatterClaims(), []);
  const [hoveredClaim, setHoveredClaim] = useState<string | null>(null);

  // Filter claims to only show those where at least one of the agreeing models is currently selected in the sidebar
  const visibleClaims = claims.filter(claim => 
    claim.agreedModels.some(model => selectedModels.includes(model))
  );

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-8 overflow-hidden">
      
      {/* Main Plot Area (Made taller and wider) */}
      <div className="relative w-full max-w-6xl h-[600px] rounded-2xl border border-[#2C3A50] bg-[#121825] shadow-2xl p-16">
        
        {/* Y-Axis Label */}
        <div className="absolute top-0 bottom-0 left-0 flex items-center justify-center w-16">
          <span className="text-[#90A2B3] text-sm font-bold tracking-widest uppercase -rotate-90 whitespace-nowrap">
            Verification Confidence
          </span>
        </div>

        {/* X-Axis Label */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-center h-16">
          <span className="text-[#90A2B3] text-sm font-bold tracking-widest uppercase">
            Model Consensus (1 to 5)
          </span>
        </div>

        {/* The Plot Container */}
        <div className="relative w-full h-full rounded-xl overflow-visible">
          
          {/* Gradient Background */}
          <div 
            className="absolute inset-0 rounded-xl opacity-30 pointer-events-none"
            style={{
              background: 'linear-gradient(to top right, #FF4757 0%, #FFB020 40%, #00D68F 100%)',
              filter: 'blur(10px)'
            }}
          />

          {/* Grid Lines */}
          <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 rounded-xl border border-[#2C3A50]/50 pointer-events-none">
            {[...Array(3)].map((_, i) => (
              <div key={`v-${i}`} className="absolute top-0 bottom-0 border-r border-[#2C3A50]/30" style={{ left: `${(i + 1) * 25}%` }} />
            ))}
            {[...Array(3)].map((_, i) => (
              <div key={`h-${i}`} className="absolute left-0 right-0 border-b border-[#2C3A50]/30" style={{ top: `${(i + 1) * 25}%` }} />
            ))}
          </div>

          {/* Data Points */}
          {visibleClaims.map((claim) => {
            const leftPos = `${((claim.plotX - 1) / 4) * 100}%`;
            const bottomPos = `${claim.plotY}%`;
            const isHovered = hoveredClaim === claim.id;
            const color = TRUST_COLORS[claim.status];

            // DETERMINISTIC FLIP: If the node is in the top 30% of the chart, show tooltip below
            const showBelow = claim.plotY > 70;

            return (
                <div
                key={claim.id}
                className={`absolute ${isHovered ? 'z-50' : 'z-10'}`}
                style={{ left: leftPos, bottom: bottomPos }}
                onMouseEnter={() => setHoveredClaim(claim.id)}
                onMouseLeave={() => setHoveredClaim(null)}
                >
                <motion.div 
                    initial={{ scale: 0 }}
                    animate={{ scale: isHovered ? 1.5 : 1 }}
                    className="w-4 h-4 -ml-2 -mb-2 rounded-full cursor-pointer shadow-[0_0_12px_rgba(0,0,0,0.6)]"
                    style={{ backgroundColor: color, border: `2px solid ${isHovered ? '#fff' : '#121825'}` }}
                />

                <AnimatePresence>
                    {isHovered && (
                    <motion.div
                        // ANIMATION ADJUSTMENT: Slide up if below, slide down if above
                        initial={{ opacity: 0, y: showBelow ? -10 : 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: showBelow ? -5 : 5, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className={`absolute left-1/2 -translate-x-1/2 w-80 p-5 rounded-xl bg-[#1A2335]/95 backdrop-blur-md border border-[#2C3A50] shadow-2xl pointer-events-none 
                        ${showBelow ? 'top-8' : 'bottom-8'}`} // CLASS ADJUSTMENT
                    >
                        {/* Tooltip Content */}
                        <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-bold text-[#90A2B3] uppercase tracking-wider">Claim</span>
                        <span className="text-sm font-bold px-2.5 py-1 rounded-md" style={{ backgroundColor: `${color}20`, color: color }}>
                            {claim.confidence}% Verified
                        </span>
                        </div>
                        
                        <p className="text-[#EBF0FF] text-base leading-snug mb-5">
                        "{claim.text}"
                        </p>
                        
                        <div className="flex flex-wrap gap-2">
                        {claim.agreedModels.map(model => {
                            const isModelActive = selectedModels.includes(model);
                            return (
                            <span 
                                key={model} 
                                className={`text-xs font-medium px-2.5 py-1 rounded-md transition-opacity ${isModelActive ? 'opacity-100' : 'opacity-30 line-through'}`}
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
                </div>
            );
            })}
        </div>
      </div>

    </div>
  );
}