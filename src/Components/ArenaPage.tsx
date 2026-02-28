import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function ArenaPage() {
  const navigate = useNavigate();
  const [showVisualizationMenu, setShowVisualizationMenu] = useState(false);
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);

  const navItems = [
    "Constellation",
    "Pipeline",
    "Heatmap",
    "Evidence Network",
    "Knowledge Deck"
  ];

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-deep-bg text-text-primary">
      
      {/* 1. BACKGROUND LAYER */}
      <div 
        className="absolute inset-0 z-0 opacity-30 pointer-events-none"
        style={{ 
          backgroundImage: "url('/background-grid.svg')", 
          backgroundSize: 'cover',
          backgroundPosition: 'center' 
        }}
      />

      {/* 2. FLOATING MENU TOGGLE (Positioned via Layer Spec) */}
      <button 
        onClick={() => setIsSidebarVisible(!isSidebarVisible)}
        className="absolute z-50 flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] bg-panel-bg shadow-xl hover:brightness-110 active:scale-95"
        style={{ 
          width: isSidebarVisible ? '41px' : '66px', 
          height: isSidebarVisible ? '41px' : '66px', 
          borderRadius: '26px',
          // 224px from left matches your spec when sidebar is 267px wide
          left: isSidebarVisible ? '224px' : '40px', 
          top: isSidebarVisible ? '37px' : '40px',
        }}
      >
        <img 
          src="/menu_open.svg" 
          alt="Toggle Menu" 
          style={{ width: '41px', height: '41px' }} 
        />
      </button>

      {/* 3. FLOATING HOME BUTTON */}
      <button 
        onClick={() => navigate('/')}
        className="absolute z-50 flex items-center justify-center transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] bg-panel-bg shadow-xl hover:brightness-110 active:scale-95"
        style={{ 
          width: isSidebarVisible ? '41px' : '66px', 
          height: isSidebarVisible ? '41px' : '66px', 
          borderRadius: '26px',
          left: isSidebarVisible ? '31px' : '40px',
          bottom: isSidebarVisible ? '42px' : '40px',
        }}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="41" height="41" viewBox="0 0 41 41" fill="none">
          <path d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.0729 5.125 35.0729 5.125 34.1667V15.3751Z" stroke="#5E6E81" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>

      {/* 4. SIDEBAR BODY */}
      <aside 
        className={`absolute z-20 bg-panel-bg flex flex-col transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
          isSidebarVisible ? 'translate-x-0 opacity-100' : '-translate-x-[110%] opacity-0'
        }`}
        style={{ 
          left: '15px', 
          top: '25px', 
          bottom: '25px', 
          width: '267px', 
          borderRadius: '26px' 
        }}
      >
        {/* Visualization Menu Section */}
        <div className="absolute flex flex-col items-center w-full" style={{ top: '80px' }}>
          <button 
            onClick={() => setShowVisualizationMenu(!showVisualizationMenu)}
            className="flex items-center px-4 transition-all bg-[#34445A] hover:brightness-110 active:scale-95"
            style={{ 
              width: '248px', 
              height: '47px', 
              borderRadius: showVisualizationMenu ? '10px 10px 0 0' : '10px' 
            }}
          >
            {/* V Icon from your screenshot */}
            <div className="flex items-center space-x-3">
               <svg 
                width="18" height="12" viewBox="0 0 18 12" fill="none"
                style={{ 
                  transform: showVisualizationMenu ? 'rotate(0deg)' : 'rotate(-90deg)',
                  transition: 'transform 0.2s ease'
                }}
              >
                <path d="M1 1L9 10L17 1" stroke="#90A2B3" strokeWidth="2.5" strokeLinecap="round"/>
              </svg>
              <span style={{ color: '#90A2B3', fontFamily: 'Inter', fontSize: '20px', fontWeight: 700 }}>
                Visualization
              </span>
            </div>
          </button>

          {showVisualizationMenu && (
            <div 
              className="flex flex-col items-center py-4 space-y-2 shadow-xl"
              style={{ width: '248px', borderRadius: '0 0 10px 10px', backgroundColor: '#586983' }}
            >
              {navItems.map((item) => (
                <div key={item} className="flex w-[200px] h-[40px] justify-start items-center px-2 cursor-pointer font-['Inter'] text-[18px]">
                  <span className="text-[#EBF0FF] hover:text-[#1A2335] transition-colors duration-200">
                    {item}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* 5. MAIN STAGE */}
      <main 
        className="absolute z-10 flex items-center justify-center overflow-hidden transition-all duration-500 border rounded-[26px] border-border/20 bg-deep-bg/40 backdrop-blur-sm"
        style={{
          left: isSidebarVisible ? '307px' : '25px', 
          top: '25px',
          right: '25px', 
          bottom: '25px',
        }}
      >
        <p className="text-[#5E6E81] font-['Inter'] animate-pulse text-lg tracking-widest uppercase">
          {isSidebarVisible ? 'Initializing Neural Net...' : 'Immersive View'}
        </p>
      </main>

    </div>
  );
}