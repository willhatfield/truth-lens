import { useState, useMemo } from 'react';
import { ShieldCheck, Activity, ChevronRight, ExternalLink, BrainCircuit, Database, Layers } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { AnalysisResult } from '../types';
import { MODEL_ID_MAP, MODEL_COLORS } from '../constants/models';

interface SynthesisViewProps {
  onOpenVisualizer: (view: string) => void;
  result: AnalysisResult | null;
}

// 1. ADDED TYPESCRIPT INTERFACE
interface SynthesisBlock {
  id: string;
  text: string;
  models: string[];
  trust: 'VerifiedSafe' | 'CautionUnverified' | 'Rejected';
  expandedReasoning: string;
}

// --- MOCK DATA ---
const SUMMARY_METRICS = {
  trustScore: 94,
  claimsExtracted: 32,
  sourcesVerified: 18,
  modelsReachedConsensus: 5
};

// 2. TYPED THE INITIAL DATA
const INITIAL_SYNTHESIS: SynthesisBlock[] = [
  {
    id: 's1',
    text: "Intermittent fasting has been consistently shown to reduce insulin resistance by 15-30% in short-term clinical trials. ",
    models: ['GPT-4 (OpenAI)', 'Gemini (Google)', 'Claude (Anthropic)'],
    trust: 'VerifiedSafe',
    expandedReasoning: "3 models strongly agreed on this claim. Cross-referenced with 4 medical journals including NEJM, showing exact statistical alignment."
  },
  {
    id: 's2',
    text: "It achieves this primarily by aligning nutrient intake with natural circadian rhythms, allowing pancreatic cells to rest. ",
    models: ['Gemini (Google)', 'Kimi (Moonshot)'],
    trust: 'VerifiedSafe',
    expandedReasoning: "Extracted primarily from Gemini's biomedical reasoning and verified against the Salk Institute circadian rhythm studies."
  },
  {
    id: 's3',
    text: "Clinical data suggests that the metabolic switch to ketone utilization during the fasting window further stabilizes blood glucose levels. ",
    models: ['Llama 3 (Meta)', 'Claude (Anthropic)', 'Gemini (Google)'],
    trust: 'VerifiedSafe',
    expandedReasoning: "Consensus reached on the 'metabolic switch' mechanism. Verified against Johns Hopkins metabolic research papers (2023)."
  },
  {
    id: 's4',
    text: "Furthermore, cellular autophagy—the process of clearing damaged proteins—is significantly upregulated after 16 hours of fasting. ",
    models: ['Gemini (Google)', 'Kimi (Moonshot)', 'GPT-4 (OpenAI)'],
    trust: 'VerifiedSafe',
    expandedReasoning: "High-confidence claim. All three models mapped this to the 16:8 protocol with 98% semantic overlap in biological pathways."
  },
  {
    id: 's5',
    text: "While early results are promising, long-term effectiveness compared to standard caloric restriction remains a point of active debate. ",
    models: ['GPT-4 (OpenAI)', 'Claude (Anthropic)', 'Llama 3 (Meta)', 'Gemini (Google)', 'Kimi (Moonshot)'],
    trust: 'CautionUnverified',
    expandedReasoning: "All 5 models noted this nuance. The evidence network found conflicting longitudinal data, marking it as a 'Debated' topic."
  },
  {
    id: 's6',
    text: "Individuals with high baseline cortisol should exercise caution, as fasting can occasionally trigger an acute stress response. ",
    models: ['Claude (Anthropic)', 'Llama 3 (Meta)'],
    trust: 'VerifiedSafe',
    expandedReasoning: "Sourced from endocrinology-specific datasets. Verified as a legitimate safety boundary for hormonal dysregulation."
  },
  {
    id: 's7',
    text: "Ultimately, the primary driver of success in reversing insulin resistance is the sustained reduction in liver fat through adherence. ",
    models: ['Gemini (Google)', 'GPT-4 (OpenAI)', 'Kimi (Moonshot)', 'Claude (Anthropic)'],
    trust: 'VerifiedSafe',
    expandedReasoning: "Strong consensus. Models cross-referenced the 'Twin Cycles' hypothesis of diabetes reversal with 4 peer-reviewed sources."
  }
];


const TRUST_COLORS: Record<string, string> = {
  VerifiedSafe: '#00D68F',
  CautionUnverified: '#FFB020',
  Rejected: '#FF4757'
};

export default function SynthesisView({ onOpenVisualizer, result }: SynthesisViewProps) {
  const [activeSegment, setActiveSegment] = useState<string | null>(null);
  const [isExpanding, setIsExpanding] = useState(false);

  // 3. TYPED THE STATE (Replaced <any[]> with <SynthesisBlock[]>)
  const [additionalBlocks, setAdditionalBlocks] = useState<SynthesisBlock[]>([]);

  // Derive data from real result, or fall back to mock
  const derivedSynthesis = useMemo<SynthesisBlock[]>(() => {
    if (!result || !result.safe_answer || !result.clusters) return INITIAL_SYNTHESIS;
    const { clusters, cluster_scores, claims, safe_answer } = result;
    const supportedIds = new Set(safe_answer.supported_cluster_ids);
    return clusters
      .filter(c => supportedIds.has(c.cluster_id))
      .map(c => {
        const score = cluster_scores.find(s => s.cluster_id === c.cluster_id);
        const verdict = score?.verdict ?? 'SAFE';
        const trust: SynthesisBlock['trust'] =
          verdict === 'SAFE' ? 'VerifiedSafe' :
          verdict === 'CAUTION' ? 'CautionUnverified' : 'Rejected';
        const modelIds = [...new Set(
          claims.filter(cl => c.claim_ids.includes(cl.claim_id)).map(cl => MODEL_ID_MAP[cl.model_id] ?? cl.model_id)
        )];
        return {
          id: c.cluster_id,
          text: c.representative_text + ' ',
          models: modelIds.length > 0 ? modelIds : ['unknown'],
          trust,
          expandedReasoning: score
            ? `Trust score: ${score.trust_score}. ${score.agreement.count} model(s) supporting.`
            : 'No score data available.',
        };
      });
  }, [result]);

  const derivedMetrics = useMemo(() => {
    if (!result) return SUMMARY_METRICS;
    const { claims, nli_results, clusters, cluster_scores } = result;
    const safeClusters = cluster_scores.filter(s => s.verdict === 'SAFE');
    const avgTrust = safeClusters.length > 0
      ? Math.round(safeClusters.reduce((a, s) => a + s.trust_score, 0) / safeClusters.length)
      : 0;
    return {
      trustScore: avgTrust,
      claimsExtracted: claims.length,
      sourcesVerified: nli_results.filter(r => r.label === 'entailment').length,
      modelsReachedConsensus: clusters.filter(c => c.claim_ids.length > 1).length,
    };
  }, [result]);

  const displaySynthesis = result ? derivedSynthesis : INITIAL_SYNTHESIS;
  const displayMetrics = result ? derivedMetrics : SUMMARY_METRICS;

  const allSegments = [...displaySynthesis, ...additionalBlocks];

  const handleExpandSynthesis = () => {
    if (result) return;
    setIsExpanding(true);
    setTimeout(() => {
      const newBlock: SynthesisBlock = {
        id: `s${allSegments.length + 1}`,
        text: "Deep Scan Result: Circadian-aligned fasting specifically optimizes the BMAL1 and CLOCK gene expressions, which are the master regulators of cellular insulin sensitivity. This 'genetic priming' distinguishes fasting from simple caloric restriction in clinical outcomes. ",
        models: ['Gemini (Google)', 'Claude (Anthropic)', 'Kimi (Moonshot)'],
        trust: 'VerifiedSafe',
        expandedReasoning: "Cross-verified across 3 genomic studies. High semantic alignment between Gemini's biological summary and Claude's clinical analysis regarding BMAL1 pathways."
      };
      setAdditionalBlocks(prev => [...prev, newBlock]);
      setIsExpanding(false);
    }, 1500);
  };

  const getHighlightStyle = (models: string[], isActive: boolean) => {
    if (models.length === 1) {
      const color = MODEL_COLORS[models[0]];
      return {
        backgroundColor: isActive ? `${color}40` : `${color}15`, 
        borderBottom: `2px solid ${color}`
      };
    }
    return {
      background: isActive 
        ? `linear-gradient(90deg, ${models.map(m => `${MODEL_COLORS[m]}40`).join(', ')})`
        : `linear-gradient(90deg, ${models.map(m => `${MODEL_COLORS[m]}15`).join(', ')})`,
      borderImage: `linear-gradient(90deg, ${models.map(m => MODEL_COLORS[m]).join(', ')}) 1`,
      borderBottom: '2px solid'
    };
  };

  return (
    <div className="flex flex-col w-full h-full p-6 overflow-y-auto no-scrollbar bg-[#0A0E1A]">
      <div className="max-w-[960px] mx-auto w-full space-y-5">
        
        {/* HEADER */}
        <div className="mb-6">
          <span className="text-[#90A2B3] text-[10px] font-bold tracking-[0.2em] uppercase mb-1.5 block">Original Query</span>
          <h1 className="text-[#EBF0FF] text-[22px] font-bold leading-tight">
            "{result ? result.prompt : 'Is intermittent fasting actually effective for fixing insulin resistance, and does it change your DNA?'}"
          </h1>
        </div>

        <div className="grid grid-cols-12 gap-5">
          
          {/* LEFT COLUMN */}
          <div className="col-span-8 space-y-5">
            <div className="bg-[#121825] border border-[#2C3A50] rounded-xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#10A37F] via-[#00D68F] to-[#5273FB]" />

              <div className="flex items-center justify-between mb-5">
                <h2 className="text-[#EBF0FF] text-lg font-bold flex items-center gap-2">
                  <BrainCircuit className="w-4 h-4 text-[#A9BDE8]" />
                  Synthesized Safe Answer
                </h2>
                <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#1A2335] border border-[#2C3A50] rounded-full">
                  <ShieldCheck className="w-3.5 h-3.5 text-[#00D68F]" />
                  <span className="text-[#00D68F] text-[10px] font-bold tracking-widest uppercase">Verified Consensus</span>
                </div>
              </div>

              {/* THE ANSWER CONTENT */}
              <div className="text-[#EBF0FF] text-[15px] leading-relaxed">
                {displaySynthesis.map((segment) => (
                  <span
                    key={segment.id}
                    onClick={() => setActiveSegment(activeSegment === segment.id ? null : segment.id)}
                    className="cursor-pointer transition-all duration-300 rounded-sm px-1 mr-1"
                    style={getHighlightStyle(segment.models, activeSegment === segment.id)}
                  >
                    {segment.text}
                  </span>
                ))}
                {result && displaySynthesis.length === 0 && (
                  <span className="text-[#5E6E81] italic">No verified claims were found for this query.</span>
                )}

                {/* THE EXPANDED BLOCKS ARE RENDERED HERE */}
                <AnimatePresence>
                  {additionalBlocks.map((block) => (
                    <motion.span 
                      key={block.id}
                      initial={{ opacity: 0, x: -5 }}
                      animate={{ opacity: 1, x: 0 }}
                      onClick={() => setActiveSegment(activeSegment === block.id ? null : block.id)}
                      className="cursor-pointer transition-all duration-300 rounded-sm px-1 mr-1 border-l-2 border-[#588983]"
                      style={getHighlightStyle(block.models, activeSegment === block.id)}
                    >
                      {block.text}
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>

              {/* THE EXPAND ACTION BAR */}
              <div className="mt-8 pt-6 border-t border-[#2C3A50]/50 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleExpandSynthesis}
                    disabled={isExpanding || !!result}
                    className="flex items-center gap-2 px-4 py-2 bg-[#1A2335] hover:bg-[#2C3A50] border border-[#588983]/40 rounded-lg transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isExpanding ? (
                      <Activity className="w-4 h-4 text-[#588983] animate-spin" />
                    ) : (
                      <Layers className="w-4 h-4 text-[#588983] group-hover:scale-110 transition-transform" />
                    )}
                    <span className="text-[#EBF0FF] text-[13px] font-bold">
                      {isExpanding ? "Analyzing Evidence..." : "Expand Synthesis"}
                    </span>
                  </button>
                  <p className="text-[#90A2B3] text-[11px] max-w-[200px] leading-tight hidden sm:block">
                    Triggers a secondary deep-scan of biological mechanisms.
                  </p>
                </div>
              </div>
            </div>

            {/* REASONING PANEL */}
            <AnimatePresence mode="wait">
              {activeSegment && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 5 }}
                  className="bg-[#1A2335] border border-[#2C3A50] rounded-xl p-5 shadow-lg"
                >
                  {(() => {
                    const segment = allSegments.find(s => s.id === activeSegment);
                    if (!segment) return null;
                    return (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-[#90A2B3] text-[10px] font-bold tracking-widest uppercase">Evidence Trace</span>
                          <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-[#121825] border border-[#2C3A50]" style={{ color: TRUST_COLORS[segment.trust] }}>
                            {segment.trust.replace(/([A-Z])/g, ' $1').trim()}
                          </span>
                        </div>
                        <p className="text-[#CCD8FF] text-[13px] leading-relaxed">{segment.expandedReasoning}</p>
                        <div className="pt-3 border-t border-[#2C3A50]/50 flex items-center gap-3">
                          <span className="text-[#588983] text-[10px] font-bold uppercase">Sources:</span>
                          <div className="flex gap-2">
                            {/* 5. EXPLICITLY TYPED 'm: string' HERE */}
                            {segment.models.map((m: string) => (
                              <div key={m} className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#121825] border border-[#2C3A50]">
                                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: MODEL_COLORS[m] }} />
                                <span className="text-[#EBF0FF] text-[9px] uppercase font-bold">{m}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* RIGHT COLUMN */}
          <div className="col-span-4 space-y-5">
            <div className="bg-[#121825] border border-[#2C3A50] rounded-xl p-5 shadow-xl flex flex-col items-center justify-center relative overflow-hidden">
              <span className="text-[#90A2B3] text-[10px] font-bold tracking-[0.2em] uppercase mb-3">Overall Trust Score</span>
              <div className="relative flex items-center justify-center w-32 h-32 mb-1">
                <svg className="w-full h-full transform -rotate-90">
                  <circle cx="64" cy="64" r="56" fill="none" stroke="#1A2335" strokeWidth="10" />
                  <motion.circle cx="64" cy="64" r="56" fill="none" stroke="#00D68F" strokeWidth="10" strokeLinecap="round" strokeDasharray="351.86" initial={{ strokeDashoffset: 351.86 }} animate={{ strokeDashoffset: 351.86 - (351.86 * (displayMetrics.trustScore / 100)) }} transition={{ duration: 1.5 }} />
                </svg>
                <div className="absolute flex flex-col items-center">
                  <span className="text-[#EBF0FF] text-4xl font-bold">{displayMetrics.trustScore}</span>
                  <span className="text-[#00D68F] text-[10px] font-bold uppercase mt-0.5">/ 100</span>
                </div>
              </div>
            </div>

            <div className="bg-[#121825] border border-[#2C3A50] rounded-xl p-5 shadow-xl space-y-3.5">
              <div className="flex items-center justify-between pb-3.5 border-b border-[#2C3A50]">
                <div className="flex items-center gap-2 text-[#90A2B3]"><Layers className="w-3.5 h-3.5" /><span className="text-[13px]">Claims Extracted</span></div>
                <span className="text-[#EBF0FF] text-[13px] font-mono font-bold">{displayMetrics.claimsExtracted}</span>
              </div>
              <div className="flex items-center justify-between pb-3.5 border-b border-[#2C3A50]">
                <div className="flex items-center gap-2 text-[#90A2B3]"><Database className="w-3.5 h-3.5" /><span className="text-[13px]">Sources Verified</span></div>
                <span className="text-[#EBF0FF] text-[13px] font-mono font-bold">{displayMetrics.sourcesVerified}</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-[#90A2B3]"><Activity className="w-3.5 h-3.5" /><span className="text-[13px]">Consensus Matches</span></div>
                <span className="text-[#EBF0FF] text-[13px] font-mono font-bold">{displayMetrics.modelsReachedConsensus} / 5</span>
              </div>
            </div>

            <button onClick={() => onOpenVisualizer('Evidence Network')} className="w-full group relative flex items-center justify-between bg-gradient-to-r from-[#1A2335] to-[#2C3A50] hover:to-[#3D2E50] border border-[#588983]/30 p-4 rounded-xl shadow-lg transition-all duration-300">
              <div className="flex flex-col items-start"><span className="text-[#EBF0FF] font-bold text-[15px]">View Reasoning</span><span className="text-[#A9BDE8] text-[11px]">Explore Evidence Network</span></div>
              <div className="w-8 h-8 rounded-full bg-[#121825] border border-[#2C3A50] flex items-center justify-center group-hover:scale-110 transition-transform"><ChevronRight className="w-4 h-4 text-[#EBF0FF]" /></div>
            </button>

            <button onClick={() => onOpenVisualizer('Constellation')} className="w-full group flex items-center justify-between bg-[#121825] hover:bg-[#1A2335] border border-[#2C3A50] p-3 rounded-xl transition-all duration-300">
              <span className="text-[#90A2B3] group-hover:text-[#EBF0FF] text-[13px] font-medium transition-colors">Open 3D Constellation</span>
              <ExternalLink className="w-3.5 h-3.5 text-[#588983]" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}