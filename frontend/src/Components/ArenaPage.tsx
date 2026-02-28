import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import type { AnalysisResult } from '../types';
import { Clock, X, ShieldCheck, AlertCircle, LogOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Constellation from './Constellation';
import Heatmap from './Heatmap';
import KnowledgeDeck from './KnowledgeDeck';
import Pipeline from './Pipeline';
import EvidenceNetwork from './EvidenceNetwork';
import LiveHUD from './LiveHUD';
import SynthesisView from './SynthesisView';

// --- MOCK HISTORY DATA ---
const MOCK_HISTORY = [
  { id: 1, query: "Is intermittent fasting actually effective for fixing insulin resistance...", date: "Today, 10:42 AM", score: 92, status: 'VerifiedSafe' },
  { id: 2, query: "Did OpenAI transition to a fully for-profit model in 2024?", date: "Yesterday", score: 88, status: 'VerifiedSafe' },
  { id: 3, query: "Does quantum entanglement allow for faster-than-light communication?", date: "Oct 12, 2025", score: 15, status: 'Rejected' },
];

export default function ArenaPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const jobId = new URLSearchParams(location.search).get('job');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  
  // Auth & History States
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  
  useEffect(() => {
    // Check if user logged in via LandingPage
    if (localStorage.getItem('truthlens_auth') === 'true') {
      setIsLoggedIn(true);
    }
    // Load analysis result from sessionStorage
    if (jobId) {
      const stored = sessionStorage.getItem(`result_${jobId}`);
      if (stored) {
        try {
          setResult(JSON.parse(stored));
        } catch {
          // ignore parse errors
        }
      }
    }
  }, [jobId]);

  const handleLogout = () => {
    localStorage.removeItem('truthlens_auth');
    setIsLoggedIn(false);
    setShowHistory(false);
    navigate('/'); 
  };

  // Active Visualization State
  const [activeVisualization, setActiveVisualization] = useState("Synthesized Answer");

  // Accordion States
  const [showVisualizationMenu, setShowVisualizationMenu] = useState(true);
  const [showModels, setShowModels] = useState(true);
  const [showTrust, setShowTrust] = useState(true);
  
  // State for AI Model Toggles
  const [selectedModels, setSelectedModels] = useState(['GPT-4 (OpenAI)', 'Gemini (Google)', 'Claude (Anthropic)', 'Llama 3 (Meta)', 'Kimi (Moonshot)']);

  const navItems = ["Synthesized Answer", "Constellation", "Pipeline", "Heatmap", "Evidence Network", "Knowledge Deck"];
  const aiModels = [
    { name: 'GPT-4 (OpenAI)', color: '#10A37F', checkColor: '#024023' },
    { name: 'Gemini (Google)', color: '#428F54', checkColor: '#02542D' },
    { name: 'Claude (Anthropic)', color: '#E8825A', checkColor: '#682D03' },
    { name: 'Llama 3 (Meta)', color: '#A8555F', checkColor: '#900B09' },
    { name: 'Kimi (Moonshot)', color: '#5273FB', checkColor: '#2943B0' }
  ];

  const safetyValues = [
    { name: 'Verified / Safe', color: '#00D68F' },
    { name: 'Unverified', color: '#FFB020' },
    { name: 'Rejected', color: '#FF4757' },
    { name: 'Subjective', color: '#7D8BAF' },
    { name: 'Neutral / No Data', color: '#404A60' }
  ];

  const toggleModel = (name: string) => {
    setSelectedModels(prev => 
      prev.includes(name) ? prev.filter(m => m !== name) : [...prev, name]
    );
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#0A0E1A] text-[#EBF0FF]">
      <LiveHUD result={result} />
      <style>{`
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* 1. BACKGROUND LAYER */}
      <div 
        className="absolute inset-0 z-0 opacity-30 pointer-events-none"
        style={{ 
          backgroundImage: "url('/background-grid.svg')", 
          backgroundSize: 'cover',
          backgroundPosition: 'center' 
        }}
      />

      {/* 2. DYNAMIC MENU TOGGLE BUTTON (Animated Edge Tab) */}
      <motion.button 
        onClick={() => setIsSidebarVisible(!isSidebarVisible)}
        initial={false}
        animate={{
          width: isSidebarVisible ? 40 : 12,
          height: isSidebarVisible ? 40 : 80,
          left: isSidebarVisible ? 240 : 0,
          top: isSidebarVisible ? 36 : 40,
          borderRadius: isSidebarVisible ? 12 : '0 16px 16px 0',
          backgroundColor: isSidebarVisible ? 'transparent' : 'rgba(18, 24, 37, 0.4)',
          borderColor: isSidebarVisible ? 'transparent' : 'rgba(44, 58, 80, 0.5)',
          borderWidth: isSidebarVisible ? 0 : 1,
          borderLeftWidth: 0,
        }}
        whileHover={{
          width: isSidebarVisible ? 40 : 64,
          backgroundColor: isSidebarVisible ? 'rgba(44, 58, 80, 0.2)' : 'rgba(18, 24, 37, 0.95)',
        }}
        transition={{ type: "spring", stiffness: 400, damping: 30 }}
        className="absolute z-50 flex items-center overflow-hidden backdrop-blur-md cursor-pointer group shadow-xl"
      >
        {/* The Edge Grip (Only visible when closed & not hovered) */}
        {!isSidebarVisible && (
          <div className="absolute left-[4px] top-1/2 -translate-y-1/2 w-[3px] h-8 bg-[#90A2B3]/50 rounded-full group-hover:opacity-0 transition-opacity duration-200" />
        )}

        {/* The Icon Container */}
        <div className={`absolute top-0 bottom-0 flex items-center justify-center transition-all duration-200 ${isSidebarVisible ? 'left-0 right-0 opacity-100' : 'left-0 w-[64px] opacity-0 group-hover:opacity-100'}`}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            {isSidebarVisible ? (
              <polyline points="15 18 9 12 15 6" className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M3 12H21M3 6H21M3 18H16" className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
        </div>
      </motion.button>

      {/* 3. DYNAMIC HOME BUTTON (Animated Edge Tab) */}
      <motion.button 
        onClick={() => navigate('/')}
        initial={false}
        animate={{
          width: isSidebarVisible ? 40 : 12,
          height: isSidebarVisible ? 40 : 80,
          left: isSidebarVisible ? 32 : 0,
          bottom: isSidebarVisible ? 44 : 40,
          borderRadius: isSidebarVisible ? 12 : '0 16px 16px 0',
          backgroundColor: isSidebarVisible ? 'transparent' : 'rgba(18, 24, 37, 0.4)',
          borderColor: isSidebarVisible ? 'transparent' : 'rgba(44, 58, 80, 0.5)',
          borderWidth: isSidebarVisible ? 0 : 1,
          borderLeftWidth: 0,
        }}
        whileHover={{
          width: isSidebarVisible ? 40 : 64,
          backgroundColor: isSidebarVisible ? 'rgba(44, 58, 80, 0.2)' : 'rgba(18, 24, 37, 0.95)',
        }}
        transition={{ type: "spring", stiffness: 400, damping: 30 }}
        className="absolute z-50 flex items-center overflow-hidden backdrop-blur-md cursor-pointer group shadow-xl"
      >
        {!isSidebarVisible && (
          <div className="absolute left-[4px] top-1/2 -translate-y-1/2 w-[3px] h-8 bg-[#A9BDE8]/50 rounded-full group-hover:opacity-0 transition-opacity duration-200" />
        )}
        
        <div className={`absolute top-0 bottom-0 flex items-center justify-center transition-all duration-200 ${isSidebarVisible ? 'left-0 right-0 opacity-100' : 'left-0 w-[64px] opacity-0 group-hover:opacity-100'}`}>
          <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 41 41" fill="none">
            <path d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.9419 5.125 35.0729 5.125 34.1667V15.3751Z" className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </motion.button>

      {/* 4. RESTRUCTURED SIDEBAR */}
      <aside 
        className={`absolute z-40 bg-[#121825]/80 backdrop-blur-2xl border border-[#2C3A50]/50 shadow-[4px_0_24px_rgba(0,0,0,0.3)] flex flex-col transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
          isSidebarVisible ? 'translate-x-0 opacity-100' : '-translate-x-[120%] opacity-0'
        }`}
        style={{ left: '16px', top: '24px', bottom: '24px', width: '280px', borderRadius: '24px' }}
      >
        <div className="h-16 w-full shrink-0 border-b border-[#2C3A50]/30 flex items-center px-6">
          <span className="text-[#EBF0FF] font-bold tracking-[0.2em] text-sm uppercase opacity-90">TruthLens</span>
        </div>

        <div className="flex-1 overflow-y-auto no-scrollbar p-4 space-y-4">
          
          {/* Card 1: Visualization */}
          <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm">
            <button onClick={() => setShowVisualizationMenu(!showVisualizationMenu)} className="w-full px-4 py-3 flex justify-between items-center bg-[#253145]/80 hover:bg-[#2A374D] transition-colors">
              <span className="font-bold text-[12px] tracking-widest uppercase text-[#90A2B3]">Visualization</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#90A2B3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: showVisualizationMenu ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}><polyline points="6 9 12 15 18 9"></polyline></svg>
            </button>
            {showVisualizationMenu && (
              <div className="p-2 space-y-1">
                {navItems.map((item) => (
                  <div key={item} onClick={() => setActiveVisualization(item)} className={`flex w-full h-[32px] items-center px-3 cursor-pointer rounded-lg transition-all duration-200 ${activeVisualization === item ? 'bg-[#34445A] border-l-2 border-[#A9BDE8]' : 'hover:bg-[#34445A]/40 border-l-2 border-transparent'}`}>
                    <span className={`text-[13px] ${activeVisualization === item ? 'text-[#EBF0FF] font-semibold' : 'text-[#90A2B3] font-medium'}`}>{item}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Card 2: AI Models */}
          <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm">
            <button onClick={() => setShowModels(!showModels)} className="w-full px-4 py-3 flex justify-between items-center bg-[#253145]/80 hover:bg-[#2A374D] transition-colors">
              <span className="font-bold text-[12px] tracking-widest uppercase text-[#90A2B3]">Active Models</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#90A2B3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: showModels ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}><polyline points="6 9 12 15 18 9"></polyline></svg>
            </button>
            {showModels && (
              <div className="p-3 space-y-3">
                {aiModels.map((model) => (
                  <div key={model.name} className="flex items-center w-full cursor-pointer group" onClick={() => toggleModel(model.name)}>
                    <div className="flex items-center justify-center transition-all duration-200 shrink-0" style={{ width: '16px', height: '16px', backgroundColor: selectedModels.includes(model.name) ? model.color : 'transparent', border: selectedModels.includes(model.name) ? 'none' : `2px solid ${model.color}`, borderRadius: '4px' }}>
                      {selectedModels.includes(model.name) && (<svg width="10" height="10" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17L4 12" stroke={model.checkColor} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/></svg>)}
                    </div>
                    <span className="ml-3 text-[#EBF0FF] text-[13px] font-medium group-hover:text-[#A9BDE8] transition-colors">{model.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Card 3: Trust Status Legend */}
          <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm">
            <button onClick={() => setShowTrust(!showTrust)} className="w-full px-4 py-3 flex justify-between items-center bg-[#253145]/80 hover:bg-[#2A374D] transition-colors">
              <span className="font-bold text-[12px] tracking-widest uppercase text-[#90A2B3]">Trust Status</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#90A2B3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: showTrust ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}><polyline points="6 9 12 15 18 9"></polyline></svg>
            </button>
            {showTrust && (
              <div className="p-3 space-y-3">
                {safetyValues.map((val) => (
                  <div key={val.name} className="flex items-center w-full">
                    <div className="shrink-0" style={{ width: '14px', height: '14px', backgroundColor: val.color, borderRadius: '3px' }} />
                    <span className="ml-3 text-[#EBF0FF] text-[13px] font-medium">{val.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Card 4: User Profile / History (Only shows if logged in) */}
          {isLoggedIn && (
            <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm p-2 flex flex-col gap-1 mt-4">
              <button 
                onClick={() => setShowHistory(true)} 
                className="flex items-center gap-3 px-3 py-2 text-[#90A2B3] hover:text-[#EBF0FF] hover:bg-[#253145] rounded-lg transition-colors w-full text-left text-[13px] font-medium"
              >
                <Clock className="w-4 h-4 text-[#A9BDE8]" />
                View Chat History
              </button>
              <button 
                onClick={handleLogout} 
                className="flex items-center gap-3 px-3 py-2 text-[#90A2B3] hover:text-[#FF4757] hover:bg-[#FF4757]/10 rounded-lg transition-colors w-full text-left text-[13px] font-medium"
              >
                <LogOut className="w-4 h-4" />
                Sign Out
              </button>
            </div>
          )}

        </div>

        <div className="h-20 w-full shrink-0 border-t border-[#2C3A50]/30" />
      </aside>

      {/* 5. MAIN STAGE */}
      <main 
        className="absolute z-10 flex items-center justify-center overflow-hidden transition-all duration-500 border rounded-[24px] border-[#2C3A50]/20 bg-[#0A0E1A]/40 backdrop-blur-sm"
        style={{ left: isSidebarVisible ? '312px' : '24px', top: '24px', right: '24px', bottom: '24px' }}
      >
        {(() => {
          switch (activeVisualization) {
            case "Synthesized Answer": return <SynthesisView onOpenVisualizer={(view) => setActiveVisualization(view)} result={result} />;
            case "Constellation": return <Constellation selectedModels={selectedModels} result={result} />;
            case "Heatmap": return <Heatmap selectedModels={selectedModels} result={result} />;
            case "Knowledge Deck": return <KnowledgeDeck selectedModels={selectedModels} result={result} />;
            case "Pipeline": return <Pipeline selectedModels={selectedModels} />;
            case "Evidence Network": return <EvidenceNetwork selectedModels={selectedModels} />;
            default: return <p className="text-[#5E6E81] font-['Inter'] animate-pulse text-lg tracking-widest uppercase">{activeVisualization} View</p>;
          }
        })()}
      </main>

      {/* 6. HISTORY SIDEBAR PANEL */}
      <AnimatePresence>
        {showHistory && (
          <>
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setShowHistory(false)}
              className="absolute inset-0 bg-[#0A0E1A]/40 backdrop-blur-sm z-40"
            />
            
            <motion.div
              initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="absolute right-0 top-0 bottom-0 w-96 bg-[#121825]/95 backdrop-blur-xl border-l border-[#2C3A50] z-50 p-6 flex flex-col shadow-2xl"
            >
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-[#EBF0FF] text-xl font-bold flex items-center gap-2">
                  <Clock className="w-5 h-5 text-[#A9BDE8]"/> Chat History
                </h2>
                <button onClick={() => setShowHistory(false)} className="text-[#90A2B3] hover:text-[#EBF0FF] transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto no-scrollbar space-y-4">
                {MOCK_HISTORY.map((chat) => (
                  <button 
                    key={chat.id}
                    onClick={() => {
                      setShowHistory(false);
                      setActiveVisualization("Synthesized Answer");
                    }}
                    className="w-full flex flex-col text-left p-4 bg-[#1A2335] hover:bg-[#251B35] border border-[#2C3A50] hover:border-[#588983] rounded-xl transition-all group"
                  >
                    <span className="text-[#90A2B3] text-[10px] uppercase tracking-widest mb-2 block">{chat.date}</span>
                    <span className="text-[#EBF0FF] text-sm font-medium leading-snug line-clamp-2 mb-3">
                      "{chat.query}"
                    </span>
                    
                    <div className="flex items-center justify-between w-full">
                      <div className="flex items-center gap-1.5 px-2 py-1 bg-[#121825] rounded-md border border-[#2C3A50]">
                        <span className="text-[#EBF0FF] text-xs font-bold">{chat.score}</span>
                        <span className="text-[#90A2B3] text-[10px]">/100</span>
                      </div>
                      
                      {chat.status === 'VerifiedSafe' ? (
                        <div className="flex items-center gap-1">
                          <ShieldCheck className="w-3.5 h-3.5 text-[#00D68F]" />
                          <span className="text-[#00D68F] text-[10px] uppercase font-bold">Verified</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1">
                          <AlertCircle className="w-3.5 h-3.5 text-[#FF4757]" />
                          <span className="text-[#FF4757] text-[10px] uppercase font-bold">Rejected</span>
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

    </div>
  );
}