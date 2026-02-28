import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { FileText, BrainCircuit, Link as LinkIcon, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

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

interface KnowledgeDeckProps {
  selectedModels: string[];
}

// --- MOCK DATA ---
const mockData = {
  query: "What is the impact of intermittent fasting on insulin resistance, and what are the cellular mechanisms behind it?",
  logic: [
    {
      model: 'GPT-4 (OpenAI)',
      summary: "Identifies a 15-30% reduction in insulin resistance. Focuses heavily on the depletion of hepatic glycogen stores and the subsequent metabolic switch to fatty acid oxidation.",
      status: 'VerifiedSafe'
    },
    {
      model: 'Gemini (Google)',
      summary: "Corroborates the 15-30% reduction. Adds context regarding the downregulation of mTOR pathways and the upregulation of AMPK, promoting autophagy.",
      status: 'VerifiedSafe'
    },
    {
      model: 'Claude (Anthropic)',
      summary: "Agrees on the general benefits but cautions that long-term human trial data over 5+ years is still limited compared to caloric restriction diets.",
      status: 'CautionUnverified'
    },
    {
      model: 'Llama 3 (Meta)',
      summary: "Claims intermittent fasting completely cures Type 2 Diabetes within 4 weeks by permanently altering pancreatic beta-cell DNA.",
      status: 'Rejected'
    },
    {
      model: 'Kimi (Moonshot)',
      summary: "Focuses primarily on the circadian rhythm alignment, suggesting that eating windows synchronized with daylight hours optimize insulin sensitivity.",
      status: 'CautionUnverified'
    }
  ],
  evidence: [
    { id: 1, title: "Effects of Intermittent Fasting on Health, Aging, and Disease", source: "New England Journal of Medicine", url: "#", trustScore: 98 },
    { id: 2, title: "mTOR and AMPK in Autophagy Regulation", source: "Nature Cell Biology", url: "#", trustScore: 95 },
    { id: 3, title: "Dietary myths and pancreas DNA", source: "Unverified Health Blog", url: "#", trustScore: 12 },
  ]
};

export default function KnowledgeDeck({ selectedModels }: KnowledgeDeckProps) {
  // Filter logic based on selected models in the sidebar
  const visibleLogic = useMemo(() => {
    return mockData.logic.filter(l => selectedModels.includes(l.model));
  }, [selectedModels]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'VerifiedSafe': return <CheckCircle2 className="w-5 h-5 text-[#00D68F]" />;
      case 'CautionUnverified': return <AlertTriangle className="w-5 h-5 text-[#FFB020]" />;
      case 'Rejected': return <XCircle className="w-5 h-5 text-[#FF4757]" />;
      default: return null;
    }
  };

  return (
    <div className="w-full h-full p-8 overflow-hidden flex flex-col">
      
      {/* 3-Column Grid */}
      <div className="flex-1 grid grid-cols-3 gap-6 min-h-0">
        
        {/* COLUMN 1: CONTEXT */}
        <div className="flex flex-col bg-[#121825] border border-[#2C3A50] rounded-2xl overflow-hidden shadow-xl">
          <div className="bg-[#1A2335] p-4 border-b border-[#2C3A50] flex items-center gap-3 shrink-0">
            <FileText className="w-5 h-5 text-[#A9BDE8]" />
            <h3 className="text-[#EBF0FF] font-semibold tracking-wide uppercase text-sm">Query Context</h3>
          </div>
          <div className="p-6 overflow-y-auto no-scrollbar flex-1">
            <div className="bg-[#0A0E1A] p-4 rounded-xl border border-[#2C3A50]/50 mb-6">
              <span className="text-xs font-bold text-[#90A2B3] uppercase tracking-wider block mb-2">User Prompt</span>
              <p className="text-[#EBF0FF] text-lg leading-relaxed">"{mockData.query}"</p>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-[#90A2B3] text-sm font-semibold uppercase tracking-wider border-b border-[#2C3A50] pb-2">Synthesis Summary</h4>
              <p className="text-[#EBF0FF] text-sm leading-relaxed opacity-80">
                The models generally agree that intermittent fasting reduces insulin resistance by 15-30% through metabolic switching and autophagy. However, there are conflicting timelines and unverified claims regarding permanent genetic alterations.
              </p>
            </div>
          </div>
        </div>

        {/* COLUMN 2: MODEL LOGIC */}
        <div className="flex flex-col bg-[#121825] border border-[#2C3A50] rounded-2xl overflow-hidden shadow-xl">
          <div className="bg-[#1A2335] p-4 border-b border-[#2C3A50] flex items-center gap-3 shrink-0">
            <BrainCircuit className="w-5 h-5 text-[#A9BDE8]" />
            <h3 className="text-[#EBF0FF] font-semibold tracking-wide uppercase text-sm">Model Reasoning</h3>
          </div>
          <div className="p-6 overflow-y-auto no-scrollbar flex-1 space-y-4">
            {visibleLogic.length === 0 ? (
              <p className="text-[#5E6E81] text-center italic mt-10">Select models from the sidebar to view their logic.</p>
            ) : (
              visibleLogic.map((logic, idx) => (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  key={logic.model} 
                  className="bg-[#0A0E1A] p-4 rounded-xl border border-[#2C3A50]/50"
                >
                  <div className="flex justify-between items-center mb-3">
                    <span 
                      className="text-xs font-bold px-2.5 py-1 rounded-md"
                      style={{ 
                        color: MODEL_COLORS[logic.model], 
                        backgroundColor: `${MODEL_COLORS[logic.model]}15`,
                        border: `1px solid ${MODEL_COLORS[logic.model]}40`
                      }}
                    >
                      {logic.model}
                    </span>
                    {getStatusIcon(logic.status)}
                  </div>
                  <p className="text-[#EBF0FF] text-sm leading-relaxed">
                    {logic.summary}
                  </p>
                </motion.div>
              ))
            )}
          </div>
        </div>

        {/* COLUMN 3: EVIDENCE */}
        <div className="flex flex-col bg-[#121825] border border-[#2C3A50] rounded-2xl overflow-hidden shadow-xl">
          <div className="bg-[#1A2335] p-4 border-b border-[#2C3A50] flex items-center gap-3 shrink-0">
            <LinkIcon className="w-5 h-5 text-[#A9BDE8]" />
            <h3 className="text-[#EBF0FF] font-semibold tracking-wide uppercase text-sm">Cited Evidence</h3>
          </div>
          <div className="p-6 overflow-y-auto no-scrollbar flex-1 space-y-4">
            {mockData.evidence.map((item, idx) => (
              <motion.div 
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.15 }}
                key={item.id} 
                // 1. ADDED ONCLICK TO OPEN URL IN NEW TAB
                onClick={() => window.open(item.url, '_blank')}
                className="group relative bg-[#0A0E1A] p-4 rounded-xl border border-[#2C3A50]/50 hover:border-[#A9BDE8]/50 transition-colors cursor-pointer"
              >
                <div className="flex justify-between items-start mb-2">
                  {/* 2. BUMPED PADDING FROM pr-8 TO pr-14 */}
                  <h4 className="text-[#EBF0FF] font-semibold text-sm leading-snug group-hover:text-[#A9BDE8] transition-colors pr-14">
                    {item.title}
                  </h4>
                  <div 
                    className="absolute top-4 right-4 flex items-center justify-center w-8 h-8 rounded-full font-bold text-xs shrink-0"
                    style={{ 
                      backgroundColor: item.trustScore > 80 ? '#00D68F20' : '#FF475720',
                      color: item.trustScore > 80 ? '#00D68F' : '#FF4757',
                      border: `1px solid ${item.trustScore > 80 ? '#00D68F50' : '#FF475750'}`
                    }}
                  >
                    {item.trustScore}
                  </div>
                </div>
                {/* 3. ADDED pr-14 TO SUBTITLE JUST IN CASE IT GETS LONG */}
                <p className="text-[#90A2B3] text-xs uppercase tracking-wider pr-14">{item.source}</p>
              </motion.div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}