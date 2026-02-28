import { useNavigate } from 'react-router-dom';

export default function ArenaPage() {
  const navigate = useNavigate();

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-deep-bg">
      
      {/* LEFT SIDEBAR */}
      {/* Pinned to top: 25px and bottom: 25px so it scales to fit YOUR screen perfectly */}
      <aside 
        className="absolute bg-panel-bg"
        style={{ 
          left: '15px', 
          top: '25px', 
          bottom: '25px', /* Replaces hardcoded 1067px height */
          width: '267px', 
          borderRadius: '26px' 
        }}
      >
        {/* Sidebar Menu Button (menu_open.svg) */}
        <button 
          className="absolute transition-opacity hover:opacity-70"
          style={{ right: '17px', top: '12px', width: '41px', height: '41px' }}
        >
          <img src="/menu_open.svg" alt="Open Menu" className="w-full h-full" />
        </button>

        {/* Visualization Menu Button */}
        <div className="absolute flex justify-center w-full" style={{ top: '80px' }}>
          <button 
            className="flex items-center justify-between px-4 transition-all bg-border hover:brightness-110 active:scale-95"
            style={{ width: '248px', height: '47px', borderRadius: '10px' }}
          >
            <span style={{ color: '#90A2B3', fontFamily: 'Inter', fontSize: '20px', fontWeight: 700 }}>
              Visualization
            </span>
            <div style={{ width: '18.35px', height: '18.35px' }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="19" height="19" viewBox="0 0 19 19" fill="none">
                <path fillRule="evenodd" clipRule="evenodd" d="M0.179809 0.929673C0.535326 0.181545 1.41847 -0.112372 2.15149 0.273423L16.9279 8.05077C17.4377 8.09732 17.9032 8.41583 18.1202 8.92967C18.1336 8.96141 18.1443 8.99419 18.1554 9.02635C18.3671 9.44598 18.3927 9.95489 18.1798 10.4033C17.9099 10.9715 17.3365 11.2771 16.755 11.2226L2.06262 18.0967C1.31225 18.4477 0.442476 18.1138 0.120239 17.3506C-0.201809 16.5874 0.145344 15.684 0.895629 15.333L13.3009 9.52831L0.864379 2.98338C0.131288 2.59755 -0.175636 1.67797 0.179809 0.929673Z" fill="#90A2B3"/>
              </svg>
            </div>
          </button>
        </div>

        {/* Home Button (Bottom Left) */}
        {/* Anchored to the bottom instead of counting pixels from the top */}
        <button 
          onClick={() => navigate('/')}
          className="absolute transition-all hover:scale-110 active:scale-95"
          style={{ left: '16px', bottom: '17px', width: '41px', height: '41px' }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="41" height="41" viewBox="0 0 41 41" fill="none">
            <path d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.9419 5.125 35.0729 5.125 34.1667V15.3751Z" stroke="#5E6E81" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </aside>

      {/* RIGHT MAIN STAGE (3D Canvas Area) */}
      <main 
        className="absolute flex items-center justify-center overflow-hidden transition-all border rounded-[26px] border-border/20"
        style={{
          left: '307px', // 15px left padding + 267px sidebar + 25px gap
          top: '25px',
          right: '25px', 
          bottom: '25px',
        }}
      >
        <p className="text-text-secondary font-['Inter'] animate-pulse">
          3D Canvas initializing here...
        </p>
      </main>

    </div>
  );
}