import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ArenaPage() {
  const navigate = useNavigate();
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  
  // Accordion States
  const [showVisualizationMenu, setShowVisualizationMenu] = useState(false);
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
      
      {/* CSS Injection for hiding scrollbars globally in this component */}
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

      {/* 2. FLOATING MENU TOGGLE */}
      <button 
        onClick={() => setIsSidebarVisible(!isSidebarVisible)}
        className="absolute z-50 flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] bg-[#121825] shadow-xl hover:brightness-110 active:scale-95"
        style={{ 
          width: isSidebarVisible ? '41px' : '66px', 
          height: isSidebarVisible ? '41px' : '66px', 
          borderRadius: '26px',
          left: isSidebarVisible ? '224px' : '40px', 
          top: isSidebarVisible ? '37px' : '40px',
        }}
      >
        <img src="/menu_open.svg" alt="Toggle Menu" style={{ width: '41px', height: '41px' }} />
      </button>

      {/* 3. FLOATING HOME BUTTON */}
      <button 
        onClick={() => navigate('/')}
        className="absolute z-50 flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] bg-[#121825] shadow-xl hover:brightness-110 active:scale-95"
        style={{ 
          width: isSidebarVisible ? '41px' : '66px', 
          height: isSidebarVisible ? '41px' : '66px', 
          borderRadius: '26px',
          left: isSidebarVisible ? '31px' : '40px',
          bottom: isSidebarVisible ? '42px' : '40px',
        }}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="41" height="41" viewBox="0 0 41 41" fill="none">
          <path d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.9419 5.125 35.0729 5.125 34.1667V15.3751Z" stroke="#5E6E81" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {/* 4. SIDEBAR BODY */}
      <aside 
        className={`absolute z-20 bg-[#121825] flex flex-col pt-20 transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
          isSidebarVisible ? 'translate-x-0 opacity-100' : '-translate-x-[110%] opacity-0'
        }`}
        style={{ left: '15px', top: '25px', bottom: '25px', width: '267px', borderRadius: '26px' }}
      >
        <div className="flex flex-col items-center w-full space-y-3 overflow-y-auto no-scrollbar pb-10">
          
          {/* Section: Visualization */}
          <div className="flex flex-col items-center w-full">
            <button 
              onClick={() => setShowVisualizationMenu(!showVisualizationMenu)}
              className="flex items-center px-4 transition-all bg-[#34445A] hover:brightness-110"
              style={{ width: '248px', height: '38px', borderRadius: showVisualizationMenu ? '10px 10px 0 0' : '10px' }}
            >
              <div className="flex items-center space-x-3">
                <svg width="14" height="10" viewBox="0 0 18 12" fill="none" style={{ transform: showVisualizationMenu ? 'rotate(0deg)' : 'rotate(-90deg)', transition: 'transform 0.2s ease' }}>
                  <path d="M1 1L9 10L17 1" stroke="#90A2B3" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
                <span className="font-bold text-[16px] text-[#90A2B3]">Visualization</span>
              </div>
            </button>
            {showVisualizationMenu && (
              <div className="flex flex-col items-center py-2 space-y-1 shadow-xl bg-[#586983]" style={{ width: '248px', borderRadius: '0 0 10px 10px' }}>
                {navItems.map((item) => (
                  <div key={item} className="flex w-[200px] h-[26px] justify-start items-center px-2 cursor-pointer group">
                    <span className="text-[#EBF0FF] group-hover:text-[#1A2335] transition-colors text-[14px]">{item}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Section: AI Models */}
          <div className="flex flex-col items-center w-full">
            <button 
              onClick={() => setShowModels(!showModels)}
              className="flex items-center px-4 transition-all bg-[#34445A] hover:brightness-110"
              style={{ width: '248px', height: '38px', borderRadius: showModels ? '10px 10px 0 0' : '10px' }}
            >
              <div className="flex items-center space-x-3">
                {/* Caret now visible */}
                <svg width="14" height="10" viewBox="0 0 18 12" fill="none" style={{ transform: showModels ? 'rotate(0deg)' : 'rotate(-90deg)', transition: 'transform 0.2s ease' }}>
                  <path d="M1 1L9 10L17 1" stroke="#90A2B3" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
                <span className="font-bold text-[16px] text-[#90A2B3]">Models</span>
              </div>
            </button>
            {showModels && (
              <div className="flex flex-col items-center py-3 space-y-2.5 bg-[#586983]" style={{ width: '248px', borderRadius: '0 0 10px 10px' }}>
                {aiModels.map((model) => {
                  const isSelected = selectedModels.includes(model.name);
                  return (
                    <div 
                      key={model.name} 
                      className="flex items-center justify-start w-[210px] cursor-pointer space-x-3"
                      onClick={() => toggleModel(model.name)}
                    >
                      <div 
                        className="flex items-center justify-center transition-all duration-200"
                        style={{ 
                          width: '16px', 
                          height: '16px', 
                          backgroundColor: isSelected ? model.color : 'rgba(255, 255, 255, 0.75)', 
                          border: isSelected ? 'none' : `2px solid ${model.color}`,
                          borderRadius: '3px' 
                        }}
                      >
                        {isSelected && (
                          <svg width="10" height="10" viewBox="0 0 22 24" fill="none">
                            <path d="M18.3333 6L8.24996 17L3.66663 12" stroke={model.checkColor} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"/>
                          </svg>
                        )}
                      </div>
                      <span className="text-[#EBF0FF] text-[14px] font-normal">{model.name}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Section: Safety Values (Legend) */}
          <div className="flex flex-col items-center w-full">
            <button 
              onClick={() => setShowTrust(!showTrust)}
              className="flex items-center px-4 transition-all bg-[#34445A] hover:brightness-110"
              style={{ width: '248px', height: '38px', borderRadius: showTrust ? '10px 10px 0 0' : '10px' }}
            >
              <div className="flex items-center space-x-3">
                {/* Caret now visible */}
                <svg width="14" height="10" viewBox="0 0 18 12" fill="none" style={{ transform: showTrust ? 'rotate(0deg)' : 'rotate(-90deg)', transition: 'transform 0.2s ease' }}>
                  <path d="M1 1L9 10L17 1" stroke="#90A2B3" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
                <span className="font-bold text-[16px] text-[#90A2B3]">Trust Status</span>
              </div>
            </button>
            {showTrust && (
              <div className="flex flex-col items-center py-3 space-y-2.5 bg-[#586983]" style={{ width: '248px', borderRadius: '0 0 10px 10px' }}>
                {safetyValues.map((val) => (
                  <div key={val.name} className="flex items-center justify-start w-[210px] space-x-3">
                    <div 
                      style={{ width: '16px', height: '16px', backgroundColor: val.color, borderRadius: '3px' }}
                    />
                    <span className="text-[#EBF0FF] text-[14px] font-normal">{val.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </aside>

      {/* 5. MAIN STAGE */}
      <main 
        className="absolute z-10 flex items-center justify-center overflow-hidden transition-all duration-500 border rounded-[26px] border-[#2C3A50]/20 bg-[#0A0E1A]/40 backdrop-blur-sm"
        style={{ left: isSidebarVisible ? '307px' : '25px', top: '25px', right: '25px', bottom: '25px' }}
      >
        <p className="text-[#5E6E81] font-['Inter'] animate-pulse text-lg tracking-widest uppercase">
          {isSidebarVisible ? 'Initializing Neural Net...' : 'Immersive View'}
        </p>
      </main>
    </div>
  );
}