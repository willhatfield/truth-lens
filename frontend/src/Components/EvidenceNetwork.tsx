import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Link as LinkIcon, AlertCircle, CheckCircle2, XCircle, ExternalLink } from 'lucide-react';

interface EvidenceNetworkProps {
  selectedModels: string[];
}

// --- MOCK DATA ---
const CLAIMS = [
  { id: 'c1', type: 'claim', label: 'Fasting reduces insulin resistance by 15-30%', trust: 'VerifiedSafe' },
  { id: 'c2', type: 'claim', label: 'Fasting alters pancreatic DNA permanently', trust: 'Rejected' },
  { id: 'c3', type: 'claim', label: 'Circadian rhythm alignment optimizes insulin', trust: 'CautionUnverified' }
];

const SOURCES = [
  { id: 's1', type: 'evidence', label: 'New England Journal of Medicine', nli: 'supports', relevance: 0.95, url: 'https://nejm.org' },
  { id: 's2', type: 'evidence', label: 'Nature Cell Biology', nli: 'supports', relevance: 0.88, url: 'https://nature.com' },
  { id: 's3', type: 'evidence', label: 'Dr. Satchin Panda, Salk Institute', nli: 'supports', relevance: 0.92, url: 'https://salk.edu' },
  { id: 's4', type: 'evidence', label: 'Healthline Blog', nli: 'neutral', relevance: 0.45, url: 'https://healthline.com' },
  { id: 's5', type: 'evidence', label: 'PubMed: Genetic Diet Study', nli: 'contradicts', relevance: 0.92, url: 'https://pubmed.gov' },
  { id: 's6', type: 'evidence', label: 'Fasting Myths (WebMD)', nli: 'contradicts', relevance: 0.75, url: 'https://webmd.com' },
  { id: 's7', type: 'evidence', label: 'Random Twitter Thread', nli: 'supports', relevance: 0.12, url: 'https://twitter.com' },
];

const LINKS = [
  { source: 'c1', target: 's1' },
  { source: 'c1', target: 's2' },
  { source: 'c1', target: 's4' },
  { source: 'c2', target: 's5' },
  { source: 'c2', target: 's6' },
  { source: 'c2', target: 's7' },
  { source: 'c3', target: 's3' },
  { source: 'c3', target: 's4' },
];

// TruthLens Palette
const TRUST_COLORS: Record<string, string> = {
  VerifiedSafe: '#00D68F',
  CautionUnverified: '#FFB020',
  Rejected: '#FF4757',
};

const NLI_COLORS: Record<string, string> = {
  supports: '#00D68F',
  contradicts: '#FF4757',
  neutral: '#949fb8'
};

export default function EvidenceNetwork({ selectedModels }: EvidenceNetworkProps) {
  const [activeNode, setActiveNode] = useState<any | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // SVG Coordinate System mappings
  const VIEW_W = 1000;
  const VIEW_H = 800;
  const COL_CLAIM_X = 250; // Shifted left to give labels room
  const COL_SOURCE_X = 750; // Shifted right

  // Distribute nodes evenly along the Y axis
  const layoutData = useMemo(() => {
    const getY = (index: number, total: number, startY: number, endY: number) => {
      if (total === 1) return (startY + endY) / 2;
      return startY + (index * (endY - startY) / (total - 1));
    };

    const positionedClaims = CLAIMS.map((c, i) => ({
      ...c,
      x: COL_CLAIM_X,
      y: getY(i, CLAIMS.length, 200, 600)
    }));

    const positionedSources = SOURCES.map((s, i) => ({
      ...s,
      x: COL_SOURCE_X,
      y: getY(i, SOURCES.length, 100, 700)
    }));

    return { claims: positionedClaims, sources: positionedSources };
  }, []);

  // Generate smooth cubic bezier curve
  const createSmoothCurve = (x1: number, y1: number, x2: number, y2: number) => {
    // FIX: Add a microscopic 1px offset if perfectly horizontal to prevent SVG filter clipping
    const safeY2 = y1 === y2 ? y2 + 1 : y2; 
    
    const controlPointOffset = (x2 - x1) * 0.5;
    return `M ${x1},${y1} C ${x1 + controlPointOffset},${y1} ${x2 - controlPointOffset},${safeY2} ${x2},${safeY2}`;
  };

  const getPos = (val: number, isX: boolean) => `${(val / (isX ? VIEW_W : VIEW_H)) * 100}%`;

  const isLinkActive = (sourceId: string, targetId: string) => {
    const focusNode = hoveredNode || activeNode?.id;
    if (!focusNode) return false;
    return focusNode === sourceId || focusNode === targetId;
  };

  return (
    <div className="flex flex-col items-center w-full h-full p-8 overflow-hidden bg-[#0A0E1A]">
      
      {/* Main Diagram Container */}
      <div className="relative flex w-full max-w-6xl flex-1 bg-[#121825] border border-[#2C3A50] rounded-2xl shadow-2xl overflow-hidden">
        
        {/* LEFT: The Interactive Bipartite Graph */}
        <div className="relative flex-1 h-full" onClick={() => setActiveNode(null)}>
          
          <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="absolute inset-0 w-full h-full z-0">
            
            {/* --- NEW: GLOW FILTER DEFINITION --- */}
            <defs>
              <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* EDGES (Lines) */}
            {LINKS.map((link, i) => {
              const source = layoutData.claims.find(c => c.id === link.source);
              const target = layoutData.sources.find(s => s.id === link.target);
              if (!source || !target) return null;

              const isFocus = isLinkActive(source.id, target.id);
              const hasAnyFocus = hoveredNode || activeNode;
              
              const edgeColor = NLI_COLORS[target.nli];
              const strokeWidth = isFocus ? 4 : target.relevance * 3;
              
              // Dim unrelated edges aggressively to make the active ones pop
              const opacity = hasAnyFocus ? (isFocus ? 0.7 : 0.05) : 0.3;
              const pathString = createSmoothCurve(source.x, source.y, target.x, target.y);

              return (
                <g key={`link-${i}`}>
                  {/* LAYER 1: The Base Track */}
                  <motion.path
                    d={pathString}
                    fill="none"
                    stroke={edgeColor}
                    strokeWidth={Math.max(1, strokeWidth)}
                    animate={{ opacity }}
                    transition={{ duration: 0.3 }}
                    className="transition-all duration-300"
                    strokeLinecap="round"
                  />
                  
                  {/* LAYER 2: The Glowing Energy Beads */}
                  <motion.path
                    d={pathString}
                    fill="none"
                    stroke={edgeColor}
                    strokeWidth={Math.max(2, strokeWidth * 1.5)} // Slightly thicker for the glow
                    filter="url(#neonGlow)" // Applies the gaussian blur defined in <defs>
                    style={{
                      strokeDasharray: "12 36", // 12px bead, 36px empty gap
                      strokeLinecap: "round"
                    }}
                    animate={{ 
                      opacity: isFocus ? 1 : (hasAnyFocus ? 0 : 0.4), // Bright glow when focused
                      strokeDashoffset: [-100, 0] // Negative to Positive makes it flow Left to Right
                    }}
                    transition={{ 
                      opacity: { duration: 0.3 },
                      // If it's focused, beads flow fast. If idle, they flow slowly.
                      strokeDashoffset: { 
                        repeat: Infinity, 
                        ease: "linear", 
                        duration: isFocus ? 1.5 : 4 
                      }
                    }}
                  />
                </g>
              );
            })}
          </svg>

          {/* HTML Layer for Nodes */}
          {[...layoutData.claims, ...layoutData.sources].map((node) => {
            const isClaim = node.type === 'claim';
            const isFocus = hoveredNode === node.id || activeNode?.id === node.id;
            const hasAnyFocus = hoveredNode || activeNode;
            
            let isConnected = false;
            if (hasAnyFocus) {
               isConnected = isFocus || LINKS.some(l => 
                 (l.source === (hoveredNode || activeNode?.id) && l.target === node.id) ||
                 (l.target === (hoveredNode || activeNode?.id) && l.source === node.id)
               );
            }
            const opacity = hasAnyFocus ? (isConnected ? 1 : 0.15) : 1;
            const nodeColor = isClaim ? TRUST_COLORS[node.trust] : NLI_COLORS[(node as any).nli];

            return (
              <motion.div
                key={node.id}
                className="absolute -translate-x-1/2 -translate-y-1/2 flex items-center justify-center cursor-pointer transition-all duration-300 z-10"
                style={{
                  left: getPos(node.x, true),
                  top: getPos(node.y, false),
                  opacity: opacity,
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={(e) => { e.stopPropagation(); setActiveNode(node); }}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                {/* CLAIM LABELS (Left of node) */}
                {isClaim && (
                  <div className="absolute right-12 top-1/2 -translate-y-1/2 w-48 text-right pointer-events-none">
                    <span className="text-[#EBF0FF] text-sm font-semibold leading-snug drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
                      {node.label}
                    </span>
                  </div>
                )}

                {/* The Circle */}
                <div 
                  className={`rounded-full transition-all duration-300 ${isFocus ? 'shadow-[0_0_20px_rgba(255,255,255,0.2)]' : ''}`}
                  style={{
                    width: isClaim ? '36px' : '24px',
                    height: isClaim ? '36px' : '24px',
                    backgroundColor: nodeColor,
                    border: `3px solid ${isFocus ? '#FFFFFF' : '#121825'}`,
                    boxShadow: isFocus ? `0 0 20px ${nodeColor}` : 'none'
                  }}
                />

                {/* SOURCE LABELS (Right of node) */}
                {!isClaim && (
                  <div className="absolute left-8 top-1/2 -translate-y-1/2 w-64 text-left pointer-events-none flex flex-col items-start gap-1">
                    <span className="text-[#EBF0FF] text-sm font-medium leading-snug drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]">
                      {node.label}
                    </span>
                    <span className="text-[9px] uppercase tracking-widest px-2 py-0.5 rounded bg-[#1A2335] border border-[#2C3A50] shadow-md" style={{ color: nodeColor }}>
                      {(node as any).nli} ({(node as any).relevance * 100}%)
                    </span>
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* RIGHT: Detail Panel Overlay */}
        <AnimatePresence>
          {activeNode && (
            <motion.div
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 100 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="absolute right-0 top-0 bottom-0 w-80 bg-[#1A2335]/95 backdrop-blur-md border-l border-[#2C3A50] shadow-2xl p-6 flex flex-col z-30 shrink-0"
            >
              <button 
                onClick={() => setActiveNode(null)}
                className="absolute top-6 right-6 text-[#5E6E81] hover:text-[#EBF0FF] transition-colors"
              >
                ✕
              </button>

              <div className="flex items-center gap-3 mb-6">
                {activeNode.type === 'claim' ? <FileText className="text-[#A9BDE8]" /> : <LinkIcon className="text-[#A9BDE8]" />}
                <h3 className="text-[#90A2B3] font-bold tracking-widest uppercase text-xs">
                  {activeNode.type === 'claim' ? 'Claim Details' : 'Source Details'}
                </h3>
              </div>

              <p className="text-[#EBF0FF] text-lg leading-relaxed font-semibold mb-6">
                "{activeNode.label}"
              </p>

              {activeNode.type === 'evidence' && (
                <div className="space-y-4">
                  <div className="bg-[#0A0E1A] p-4 rounded-xl border border-[#2C3A50]/50">
                    <span className="block text-[#5E6E81] text-[10px] uppercase tracking-widest mb-1">Entailment Verdict</span>
                    <div className="flex items-center gap-2">
                      {activeNode.nli === 'supports' ? <CheckCircle2 className="w-4 h-4 text-[#00D68F]" /> : 
                       activeNode.nli === 'contradicts' ? <XCircle className="w-4 h-4 text-[#FF4757]" /> : 
                       <AlertCircle className="w-4 h-4 text-[#4A4060]" />}
                      <span className="text-sm font-bold uppercase" style={{ color: NLI_COLORS[activeNode.nli] }}>
                        {activeNode.nli}
                      </span>
                    </div>
                  </div>

                  <div className="bg-[#0A0E1A] p-4 rounded-xl border border-[#2C3A50]/50 flex justify-between items-center">
                    <div>
                      <span className="block text-[#5E6E81] text-[10px] uppercase tracking-widest mb-1">Relevance Score</span>
                      <span className="text-[#EBF0FF] text-2xl font-bold font-mono">
                        {Math.round(activeNode.relevance * 100)}%
                      </span>
                    </div>
                  </div>

                  <button 
                    onClick={() => window.open(activeNode.url, '_blank')}
                    className="w-full mt-4 py-3 flex justify-center items-center gap-2 bg-[#1A2335] hover:bg-[#34445A] border border-[#2C3A50] hover:border-[#A9BDE8]/50 text-[#A9BDE8] font-semibold rounded-xl transition-all text-sm"
                  >
                    <ExternalLink className="w-4 h-4" />
                    View Source
                  </button>
                </div>
              )}

              {activeNode.type === 'claim' && (
                <div className="bg-[#0A0E1A] p-4 rounded-xl border border-[#2C3A50]/50">
                  <span className="block text-[#5E6E81] text-[10px] uppercase tracking-widest mb-1">Trust Status</span>
                  <span className="text-lg font-bold" style={{ 
                    color: activeNode.trust === 'VerifiedSafe' ? '#00D68F' : 
                           activeNode.trust === 'Rejected' ? '#FF4757' : '#FFB020'
                  }}>
                    {activeNode.trust.replace(/([A-Z])/g, ' $1').trim()}
                  </span>
                </div>
              )}

            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
}