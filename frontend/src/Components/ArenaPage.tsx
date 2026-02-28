import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Constellation from './Constellation';
import Heatmap from './Heatmap';
import KnowledgeDeck from './KnowledgeDeck';
import Pipeline from './Pipeline';
import EvidenceNetwork from './EvidenceNetwork';
import LiveHUD from './LiveHUD';

export default function ArenaPage() {
  const navigate = useNavigate();
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  
  // Active Visualization State
  const [activeVisualization, setActiveVisualization] = useState("Constellation");
  
  // Accordion States - Set to true by default so the sidebar looks populated!
  const [showVisualizationMenu, setShowVisualizationMenu] = useState(true);
  const [showModels, setShowModels] = useState(true);
  const [showTrust, setShowTrust] = useState(true);
  
  // State for AI Model Toggles
  const [selectedModels, setSelectedModels] = useState(['GPT-4 (OpenAI)', 'Gemini (Google)', 'Claude (Anthropic)', 'Llama 3 (Meta)', 'Kimi (Moonshot)']);

  const navItems = ["Constellation", "Pipeline", "Heatmap", "Evidence Network", "Knowledge Deck"];

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
      <LiveHUD />
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

      {/* 2. DYNAMIC MENU TOGGLE BUTTON */}
      <button 
        onClick={() => setIsSidebarVisible(!isSidebarVisible)}
        className={`absolute z-50 flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] active:scale-95 cursor-pointer group ${
          isSidebarVisible 
            ? 'bg-transparent border-transparent shadow-none' // Bare icon when open
            : 'bg-[#121825]/60 backdrop-blur-xl border border-[#2C3A50]/50 shadow-xl hover:bg-[#121825]/80' // Frosted circle when closed
        }`}
        style={{ 
            width: isSidebarVisible ? '40px' : '66px', 
            height: isSidebarVisible ? '40px' : '66px', 
            borderRadius: '26px',
            left: isSidebarVisible ? '240px' : '40px', // Slides into the sidebar header
            top: isSidebarVisible ? '36px' : '40px',
        }}
      >
        {!isSidebarVisible && (
          <span className="absolute inset-0 transition-opacity duration-300 rounded-full opacity-0 blur-md bg-[#A9BDE8]/20 group-hover:opacity-100" />
        )}
        
        <svg 
            width={isSidebarVisible ? "24" : "32"} 
            height={isSidebarVisible ? "24" : "32"} 
            viewBox="0 0 24 24" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg"
            className="relative z-10 transition-colors duration-200"
        >
          {isSidebarVisible ? (
            // Collapse Chevron (LexonAI style)
            <polyline points="15 18 9 12 15 6" className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          ) : (
            // Expand Hamburger Menu
            <path d="M3 12H21M3 6H21M3 18H16" className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
          )}
        </svg>
      </button>

      {/* 3. DYNAMIC HOME BUTTON */}
      <button 
        onClick={() => navigate('/')}
        className={`absolute z-50 flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] active:scale-95 cursor-pointer group ${
          isSidebarVisible 
            ? 'bg-transparent border-transparent shadow-none' // Bare icon when open
            : 'bg-[#121825]/60 backdrop-blur-xl border border-[#2C3A50]/50 shadow-xl hover:bg-[#121825]/80' // Frosted circle when closed
        }`}
        style={{ 
            width: isSidebarVisible ? '40px' : '66px', 
            height: isSidebarVisible ? '40px' : '66px', 
            borderRadius: '26px',
            left: isSidebarVisible ? '32px' : '40px', // Slides into the sidebar footer
            bottom: isSidebarVisible ? '44px' : '40px',
        }}
      >
        {!isSidebarVisible && (
          <span className="absolute inset-0 transition-opacity duration-300 rounded-full opacity-0 blur-md bg-[#A9BDE8]/20 group-hover:opacity-100" />
        )}
        
        <svg 
            xmlns="http://www.w3.org/2000/svg" 
            width={isSidebarVisible ? "26" : "34"} 
            height={isSidebarVisible ? "26" : "34"} 
            viewBox="0 0 41 41" 
            fill="none"
            className="relative z-10 transition-colors duration-200"
        >
            <path 
            d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.9419 5.125 35.0729 5.125 34.1667V15.3751Z" 
            className="stroke-[#5E6E81] group-hover:stroke-[#EBF0FF]" 
            strokeWidth="3" 
            strokeLinecap="round" 
            strokeLinejoin="round"
            />
        </svg>
      </button>

      {/* 4. RESTRUCTURED SIDEBAR (Flexbox Drawer with LexonAI Card Style) */}
      <aside 
        className={`absolute z-40 bg-[#121825]/80 backdrop-blur-2xl border border-[#2C3A50]/50 shadow-[4px_0_24px_rgba(0,0,0,0.3)] flex flex-col transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
          isSidebarVisible ? 'translate-x-0 opacity-100' : '-translate-x-[120%] opacity-0'
        }`}
        style={{ left: '16px', top: '24px', bottom: '24px', width: '280px', borderRadius: '24px' }}
      >
        {/* Sidebar Header (Menu button physically lives on top of this) */}
        <div className="h-16 w-full shrink-0 border-b border-[#2C3A50]/30 flex items-center px-6">
          <span className="text-[#EBF0FF] font-bold tracking-[0.2em] text-sm uppercase opacity-90">TruthLens</span>
        </div>

        {/* Scrollable Content Zone */}
        <div className="flex-1 overflow-y-auto no-scrollbar p-4 space-y-4">
          
          {/* Card 1: Visualization */}
          <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm">
            <button 
              onClick={() => setShowVisualizationMenu(!showVisualizationMenu)}
              className="w-full px-4 py-3 flex justify-between items-center bg-[#253145]/80 hover:bg-[#2A374D] transition-colors"
            >
              <span className="font-bold text-[12px] tracking-widest uppercase text-[#90A2B3]">Visualization</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#90A2B3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: showVisualizationMenu ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}>
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </button>
            
            {showVisualizationMenu && (
              <div className="p-2 space-y-1">
                {navItems.map((item) => {
                  const isSelected = activeVisualization === item;
                  return (
                    <div 
                      key={item} 
                      onClick={() => setActiveVisualization(item)}
                      className={`flex w-full h-[32px] items-center px-3 cursor-pointer rounded-lg transition-all duration-200 ${
                        isSelected ? 'bg-[#34445A] border-l-2 border-[#A9BDE8]' : 'hover:bg-[#34445A]/40 border-l-2 border-transparent'
                      }`}
                    >
                      <span className={`text-[13px] ${isSelected ? 'text-[#EBF0FF] font-semibold' : 'text-[#90A2B3] font-medium'}`}>
                        {item}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Card 2: AI Models */}
          <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm">
            <button 
              onClick={() => setShowModels(!showModels)}
              className="w-full px-4 py-3 flex justify-between items-center bg-[#253145]/80 hover:bg-[#2A374D] transition-colors"
            >
              <span className="font-bold text-[12px] tracking-widest uppercase text-[#90A2B3]">Active Models</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#90A2B3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: showModels ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}>
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </button>
            
            {showModels && (
              <div className="p-3 space-y-3">
                {aiModels.map((model) => {
                  const isSelected = selectedModels.includes(model.name);
                  return (
                    <div 
                      key={model.name} 
                      className="flex items-center w-full cursor-pointer group"
                      onClick={() => toggleModel(model.name)}
                    >
                      <div 
                        className="flex items-center justify-center transition-all duration-200 shrink-0"
                        style={{ 
                          width: '16px', height: '16px', 
                          backgroundColor: isSelected ? model.color : 'transparent', 
                          border: isSelected ? 'none' : `2px solid ${model.color}`,
                          borderRadius: '4px' 
                        }}
                      >
                        {isSelected && (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none">
                            <path d="M20 6L9 17L4 12" stroke={model.checkColor} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        )}
                      </div>
                      <span className="ml-3 text-[#EBF0FF] text-[13px] font-medium group-hover:text-[#A9BDE8] transition-colors">{model.name}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Card 3: Trust Status Legend */}
          <div className="bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-xl overflow-hidden backdrop-blur-sm">
            <button 
              onClick={() => setShowTrust(!showTrust)}
              className="w-full px-4 py-3 flex justify-between items-center bg-[#253145]/80 hover:bg-[#2A374D] transition-colors"
            >
              <span className="font-bold text-[12px] tracking-widest uppercase text-[#90A2B3]">Trust Status</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#90A2B3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ transform: showTrust ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}>
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </button>
            
            {showTrust && (
              <div className="p-3 space-y-3">
                {safetyValues.map((val) => (
                  <div key={val.name} className="flex items-center w-full">
                    <div 
                      className="shrink-0"
                      style={{ width: '14px', height: '14px', backgroundColor: val.color, borderRadius: '3px' }}
                    />
                    <span className="ml-3 text-[#EBF0FF] text-[13px] font-medium">{val.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>

        {/* Sidebar Footer (Home button physically lives on top of this) */}
        <div className="h-20 w-full shrink-0 border-t border-[#2C3A50]/30" />
      </aside>

      {/* 5. MAIN STAGE */}
      <main 
        className="absolute z-10 flex items-center justify-center overflow-hidden transition-all duration-500 border rounded-[24px] border-[#2C3A50]/20 bg-[#0A0E1A]/40 backdrop-blur-sm"
        style={{ left: isSidebarVisible ? '312px' : '24px', top: '24px', right: '24px', bottom: '24px' }}
      >
        {(() => {
          switch (activeVisualization) {
            case "Constellation":
              return <Constellation selectedModels={selectedModels} />;
            case "Heatmap":
              return <Heatmap selectedModels={selectedModels} />;
            case "Knowledge Deck":
              return <KnowledgeDeck selectedModels={selectedModels} />;
            case "Pipeline":
              return <Pipeline selectedModels={selectedModels} />;
            case "Evidence Network":
              return <EvidenceNetwork selectedModels={selectedModels} />;
            default:
              return (
                <p className="text-[#5E6E81] font-['Inter'] animate-pulse text-lg tracking-widest uppercase">
                  {activeVisualization} View (Coming Soon)
                </p>
              );
          }
        })()}
      </main>
    </div>
  );
}