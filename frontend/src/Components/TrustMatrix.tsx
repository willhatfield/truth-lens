import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { AnalysisResult } from '../types';

const VERDICT_COLORS: Record<string, string> = {
  SAFE: '#00D68F',
  CAUTION: '#FFB020',
  REJECT: '#FF4757',
};

const MODEL_COLORS: Record<string, string> = {
  'GPT-4 (OpenAI)': '#10A37F',
  'Gemini (Google)': '#428F54',
  'Claude (Anthropic)': '#E8825A',
  'Llama 3 (Meta)': '#A8555F',
  'Kimi (Moonshot)': '#5273FB',
};

const MODELS = Object.keys(MODEL_COLORS);

interface TrustMatrixProps {
  selectedModels: string[];
  result: AnalysisResult | null;
}

interface MatrixPoint {
  id: string;
  text: string;
  agreementScore: number;
  verificationScore: number;
  independenceScore: number;
  consistencyScore: number;
  trustScore: number;
  verdict: 'SAFE' | 'CAUTION' | 'REJECT';
  claimCount: number;
  models: string[];
}

const MOCK_POINTS: MatrixPoint[] = [
  { id: 'mock-1', text: 'Intermittent fasting reduces insulin resistance by 15-30%.', agreementScore: 80, verificationScore: 78, independenceScore: 80, consistencyScore: 85, trustScore: 80, verdict: 'SAFE', claimCount: 4, models: ['GPT-4 (OpenAI)', 'Gemini (Google)', 'Claude (Anthropic)', 'Llama 3 (Meta)'] },
  { id: 'mock-2', text: 'OpenAI was founded in December 2015.', agreementScore: 100, verificationScore: 92, independenceScore: 100, consistencyScore: 95, trustScore: 96, verdict: 'SAFE', claimCount: 5, models: MODELS },
  { id: 'mock-3', text: 'Quantum entanglement allows faster-than-light communication.', agreementScore: 40, verificationScore: 8, independenceScore: 40, consistencyScore: 50, trustScore: 28, verdict: 'REJECT', claimCount: 2, models: ['Llama 3 (Meta)', 'Kimi (Moonshot)'] },
  { id: 'mock-4', text: 'The Paris Agreement aims to limit warming to 1.5°C.', agreementScore: 100, verificationScore: 88, independenceScore: 100, consistencyScore: 90, trustScore: 93, verdict: 'SAFE', claimCount: 5, models: MODELS },
  { id: 'mock-5', text: 'Python is the most popular ML language.', agreementScore: 60, verificationScore: 55, independenceScore: 60, consistencyScore: 70, trustScore: 60, verdict: 'CAUTION', claimCount: 3, models: ['GPT-4 (OpenAI)', 'Claude (Anthropic)', 'Kimi (Moonshot)'] },
  { id: 'mock-6', text: 'Humans only use 10% of their brains.', agreementScore: 20, verificationScore: 5, independenceScore: 20, consistencyScore: 30, trustScore: 14, verdict: 'REJECT', claimCount: 1, models: ['Kimi (Moonshot)'] },
  { id: 'mock-7', text: 'Mars has two moons: Phobos and Deimos.', agreementScore: 100, verificationScore: 95, independenceScore: 100, consistencyScore: 98, trustScore: 97, verdict: 'SAFE', claimCount: 5, models: MODELS },
  { id: 'mock-8', text: 'AGI will be achieved by 2027.', agreementScore: 40, verificationScore: 20, independenceScore: 40, consistencyScore: 35, trustScore: 32, verdict: 'REJECT', claimCount: 2, models: ['GPT-4 (OpenAI)', 'Llama 3 (Meta)'] },
  { id: 'mock-9', text: 'Recent studies show promise for mRNA cancer vaccines.', agreementScore: 20, verificationScore: 72, independenceScore: 20, consistencyScore: 60, trustScore: 42, verdict: 'CAUTION', claimCount: 1, models: ['Claude (Anthropic)'] },
  { id: 'mock-10', text: 'Cold fusion has been successfully demonstrated.', agreementScore: 20, verificationScore: 3, independenceScore: 20, consistencyScore: 20, trustScore: 12, verdict: 'REJECT', claimCount: 1, models: ['Kimi (Moonshot)'] },
  { id: 'mock-11', text: 'Vitamin C prevents the common cold.', agreementScore: 40, verificationScore: 12, independenceScore: 40, consistencyScore: 45, trustScore: 28, verdict: 'REJECT', claimCount: 2, models: ['Llama 3 (Meta)', 'Gemini (Google)'] },
  { id: 'mock-12', text: 'Exercise improves cognitive function in elderly patients.', agreementScore: 20, verificationScore: 82, independenceScore: 20, consistencyScore: 75, trustScore: 48, verdict: 'CAUTION', claimCount: 1, models: ['Gemini (Google)'] },
];

const QUADRANTS = [
  { label: 'Verified Consensus', color: '#00D68F', x: 'right', y: 'top' },
  { label: 'Evidence-Backed Minority', color: '#38BDF8', x: 'left', y: 'top' },
  { label: 'Unverified Consensus', color: '#FFB020', x: 'right', y: 'bottom' },
  { label: 'Disputed & Unverified', color: '#FF4757', x: 'left', y: 'bottom' },
] as const;

export default function TrustMatrix({ selectedModels, result }: TrustMatrixProps) {
  const [hoveredPoint, setHoveredPoint] = useState<string | null>(null);

  const points = useMemo<MatrixPoint[]>(() => {
    if (!result) return MOCK_POINTS;
    const { cluster_scores, clusters } = result;
    return cluster_scores.map((score, idx) => {
      const cluster = clusters.find(c => c.cluster_id === score.cluster_id);
      const claimCount = cluster ? cluster.claim_ids.length : 1;

      const agreementScore = Math.round((score.agreement.count / 5) * 100);
      const verificationScore = Math.round(
        Math.max(0, Math.min(100, (score.verification.best_entailment_prob - score.verification.best_contradiction_prob) * 100))
      );

      const uniqueFamilies = new Set(score.agreement.models_supporting.map(m => m.split('(')[1]?.replace(')', '').trim() ?? m));
      const independenceScore = Math.round((uniqueFamilies.size / 5) * 100);

      const consistencyScore = claimCount >= 3 ? 90 : claimCount === 2 ? 70 : 50;

      return {
        id: `cluster-${idx}`,
        text: cluster?.representative_text ?? score.cluster_id,
        agreementScore,
        verificationScore,
        independenceScore,
        consistencyScore,
        trustScore: Math.round(score.trust_score),
        verdict: score.verdict,
        claimCount,
        models: score.agreement.models_supporting,
      };
    });
  }, [result]);

  const visiblePoints = points.filter(p =>
    p.models.some(m => selectedModels.includes(m))
  );

  // SVG layout constants
  const svgW = 900;
  const svgH = 560;
  const pad = { top: 40, right: 40, bottom: 60, left: 70 };
  const plotW = svgW - pad.left - pad.right;
  const plotH = svgH - pad.top - pad.bottom;

  const toX = (v: number) => pad.left + (v / 100) * plotW;
  const toY = (v: number) => pad.top + ((100 - v) / 100) * plotH;

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-8 overflow-hidden bg-[#0A0E1A]">
      <div className="relative w-full max-w-5xl">
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="w-full"
        style={{ aspectRatio: `${svgW}/${svgH}` }}
      >
        <defs>
          {/* Quadrant background gradients */}
          <radialGradient id="q-tr" cx="100%" cy="0%" r="80%">
            <stop offset="0%" stopColor="#00D68F" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#00D68F" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="q-tl" cx="0%" cy="0%" r="80%">
            <stop offset="0%" stopColor="#38BDF8" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#38BDF8" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="q-br" cx="100%" cy="100%" r="80%">
            <stop offset="0%" stopColor="#FFB020" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#FFB020" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="q-bl" cx="0%" cy="100%" r="80%">
            <stop offset="0%" stopColor="#FF4757" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#FF4757" stopOpacity="0" />
          </radialGradient>

          <clipPath id="plot-clip">
            <rect x={pad.left} y={pad.top} width={plotW} height={plotH} rx="12" />
          </clipPath>

          {visiblePoints.map(p => (
            <filter key={`glow-${p.id}`} id={`glow-${p.id}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation={2 + p.consistencyScore / 30} result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          ))}
        </defs>

        {/* Plot area background */}
        <rect x={pad.left} y={pad.top} width={plotW} height={plotH} rx="12" fill="#0A0A0E" stroke="#2C3A50" strokeOpacity="0.5" />

        {/* Quadrant fills (clipped to rounded plot area) */}
        <g clipPath="url(#plot-clip)">
          <rect x={pad.left} y={pad.top} width={plotW / 2} height={plotH / 2} fill="url(#q-tl)" />
          <rect x={pad.left + plotW / 2} y={pad.top} width={plotW / 2} height={plotH / 2} fill="url(#q-tr)" />
          <rect x={pad.left} y={pad.top + plotH / 2} width={plotW / 2} height={plotH / 2} fill="url(#q-bl)" />
          <rect x={pad.left + plotW / 2} y={pad.top + plotH / 2} width={plotW / 2} height={plotH / 2} fill="url(#q-br)" />
        </g>

        {/* Midpoint dividers */}
        <line x1={toX(50)} y1={pad.top} x2={toX(50)} y2={pad.top + plotH} stroke="#3A4B66" strokeOpacity="0.5" strokeDasharray="6 4" />
        <line x1={pad.left} y1={toY(50)} x2={pad.left + plotW} y2={toY(50)} stroke="#3A4B66" strokeOpacity="0.5" strokeDasharray="6 4" />

        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map(v => (
          <g key={`grid-${v}`}>
            {v !== 50 && (
              <>
                <line x1={toX(v)} y1={pad.top} x2={toX(v)} y2={pad.top + plotH} stroke="#3A4B66" strokeOpacity="0.2" />
                <line x1={pad.left} y1={toY(v)} x2={pad.left + plotW} y2={toY(v)} stroke="#3A4B66" strokeOpacity="0.2" />
              </>
            )}
            <text x={toX(v)} y={pad.top + plotH + 18} textAnchor="middle" fill="#588983" fontSize="11" fontFamily="monospace">{v}</text>
            <text x={pad.left - 10} y={toY(v) + 4} textAnchor="end" fill="#588983" fontSize="11" fontFamily="monospace">{v}</text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={pad.left + plotW / 2} y={svgH - 8} textAnchor="middle" fill="#A9BDE8" fontSize="10" fontWeight="bold" letterSpacing="0.15em" style={{ textTransform: 'uppercase' }}>
          Agreement Score
        </text>
        <text x={14} y={pad.top + plotH / 2} textAnchor="middle" fill="#A9BDE8" fontSize="10" fontWeight="bold" letterSpacing="0.15em" style={{ textTransform: 'uppercase' }} transform={`rotate(-90, 14, ${pad.top + plotH / 2})`}>
          Verification Score
        </text>

        {/* Quadrant labels */}
        {QUADRANTS.map(q => {
          const qx = q.x === 'right' ? pad.left + plotW - 12 : pad.left + 12;
          const qy = q.y === 'top' ? pad.top + 20 : pad.top + plotH - 12;
          const anchor = q.x === 'right' ? 'end' : 'start';
          return (
            <text key={q.label} x={qx} y={qy} textAnchor={anchor} fill={q.color} fillOpacity="0.4" fontSize="13" fontWeight="800" letterSpacing="0.12em" style={{ textTransform: 'uppercase' }}>
              {q.label}
            </text>
          );
        })}

        {/* Data points */}
        {visiblePoints.map((p, i) => {
          const cx = toX(p.agreementScore);
          const cy = toY(p.verificationScore);
          const baseRadius = 6 + Math.min(p.claimCount, 5) * 2.5;
          const color = VERDICT_COLORS[p.verdict];
          const isHovered = hoveredPoint === p.id;
          const glowWidth = 2 + (p.consistencyScore / 100) * 4;

          return (
            <motion.g
              key={p.id}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.04, duration: 0.4, type: 'spring' }}
              style={{ transformOrigin: `${cx}px ${cy}px` }}
            >
              {/* Consistency glow ring */}
              <circle
                cx={cx} cy={cy}
                r={baseRadius + glowWidth}
                fill="none"
                stroke={color}
                strokeOpacity={isHovered ? 0.6 : 0.25}
                strokeWidth={glowWidth}
                filter={`url(#glow-${p.id})`}
              />
              {/* Main point */}
              <motion.circle
                cx={cx} cy={cy}
                r={baseRadius}
                fill="#1A1215"
                stroke={color}
                strokeWidth={2}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint(p.id)}
                onMouseLeave={() => setHoveredPoint(null)}
                animate={{ scale: isHovered ? 1.15 : 1 }}
                transition={{ duration: 0.2, type: 'spring', stiffness: 400, damping: 25 }}
              />
              {/* Claim count label */}
              <text
                x={cx} y={cy + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="white"
                fontSize="11"
                fontWeight="bold"
                style={{ pointerEvents: 'none' }}
              >
                {p.claimCount}
              </text>
            </motion.g>
          );
        })}
      </svg>

      {/* HTML tooltip overlay */}
      <AnimatePresence>
        {hoveredPoint && (() => {
          const p = visiblePoints.find(pt => pt.id === hoveredPoint);
          if (!p) return null;
          const cx = toX(p.agreementScore);
          const cy = toY(p.verificationScore);
          const showLeft = p.agreementScore > 60;
          const showAbove = p.verificationScore < 40;

          return (
            <motion.div
              key={p.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute w-80 p-4 rounded-xl bg-[#1A2335]/95 backdrop-blur-md border border-[#2C3A50] shadow-2xl pointer-events-none z-50"
              style={{
                left: `calc(${(cx / svgW) * 100}% + ${showLeft ? '-340px' : '20px'})`,
                top: `calc(${(cy / svgH) * 100}% + ${showAbove ? '-220px' : '20px'})`,
              }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-[#90A2B3] uppercase tracking-wider">Claim Cluster</span>
                <span
                  className="text-xs font-bold px-2 py-0.5 rounded-md"
                  style={{ backgroundColor: `${VERDICT_COLORS[p.verdict]}20`, color: VERDICT_COLORS[p.verdict] }}
                >
                  {p.trustScore}% Trust
                </span>
              </div>

              {/* Claim text */}
              <p className="text-[#EBF0FF] text-[13px] leading-snug mb-3">"{p.text}"</p>

              {/* Component score mini bars */}
              <div className="space-y-1.5 mb-3">
                {[
                  { label: 'Agreement', value: p.agreementScore, weight: '35%', color: '#A9BDE8' },
                  { label: 'Verification', value: p.verificationScore, weight: '35%', color: '#38BDF8' },
                  { label: 'Independence', value: p.independenceScore, weight: '15%', color: '#C084FC' },
                  { label: 'Consistency', value: p.consistencyScore, weight: '15%', color: '#F472B6' },
                ].map(bar => (
                  <div key={bar.label} className="flex items-center gap-2">
                    <span className="text-[10px] text-[#90A2B3] w-20 shrink-0 text-right">{bar.label}</span>
                    <div className="flex-1 h-1.5 bg-[#2C3A50]/60 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${bar.value}%`, backgroundColor: bar.color }} />
                    </div>
                    <span className="text-[10px] text-[#EBF0FF] w-8 font-mono">{bar.value}</span>
                    <span className="text-[9px] text-[#588983] w-6">({bar.weight})</span>
                  </div>
                ))}
              </div>

              {/* Model tags */}
              <div className="pt-2 border-t border-[#2C3A50]/50 flex flex-wrap gap-1.5">
                {p.models.map(model => {
                  const active = selectedModels.includes(model);
                  return (
                    <span
                      key={model}
                      className={`text-[9px] uppercase tracking-widest font-bold px-1.5 py-0.5 rounded transition-opacity ${active ? 'opacity-100' : 'opacity-30 line-through'}`}
                      style={{ backgroundColor: `${MODEL_COLORS[model]}20`, color: MODEL_COLORS[model], border: `1px solid ${MODEL_COLORS[model]}40` }}
                    >
                      {model.split(' ')[0]}
                    </span>
                  );
                })}
              </div>
            </motion.div>
          );
        })()}
      </AnimatePresence>
      </div>
    </div>
  );
}
