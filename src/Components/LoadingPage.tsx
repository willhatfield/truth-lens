import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const loadingMessages = [
  "gathering information...",
  "consulting AIs...",
  "parsing through information...",
  "creating visuals...",
  "making everything pretty..."
];

export default function LoadingPage() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const startTimeout = setTimeout(() => {
      const timer = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 50) {
            clearInterval(timer);
            setTimeout(() => navigate('/arena'), 1000);
            return 50;
          }
          return prev + 1;
        });
      }, 120);
      return () => clearInterval(timer);
    }, 500);
    return () => clearTimeout(startTimeout);
  }, [navigate]);

  const percentage = Math.round((progress / 50) * 100);
  const messageIndex = Math.min(
    Math.floor((progress / 50) * loadingMessages.length),
    loadingMessages.length - 1
  );

  return (
    <div className="relative flex flex-col items-center justify-center w-screen h-screen bg-deep-bg overflow-hidden px-4">
      
      {/* Main Content Wrapper - Locked to 817px */}
      <div className="flex flex-col items-start w-[817px]">
        
        {/* Dynamic Loading Text with Pulse Animation */}
        <div className="flex items-center h-[71px]">
          <motion.div
            key={messageIndex} // Resets animation on text change
            initial={{ opacity: 0.4 }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ 
              duration: 2, 
              repeat: Infinity, 
              ease: "easeInOut" 
            }}
          >
            <p style={{ color: '#CCD8FF', fontFamily: 'Inter', fontSize: '20px' }}>
              {loadingMessages[messageIndex]}
            </p>
          </motion.div>
        </div>

        {/* 70px Percentage Display */}
        <div className="flex flex-col justify-center h-[174px]">
          <p style={{ color: '#CCD8FF', fontFamily: 'Inter', fontSize: '70px', fontWeight: 400 }}>
            {percentage} %
          </p>
        </div>

        {/* Progress Bar */}
        <div className="flex items-center justify-between mb-6 w-[817px]">
          {[...Array(50)].map((_, i) => (
            <motion.div
              key={i}
              initial={{ height: "4px", opacity: 0.2 }}
              animate={i < progress ? { height: '73px', opacity: 1 } : { height: '4px', opacity: 0.2 }}
              transition={{
                type: "spring",
                mass: 1,
                stiffness: 100,
                damping: 15
              }}
              style={{ 
                width: '12px',
                backgroundColor: i < progress ? '#A9BDE8' : '#1E2636',
                borderRadius: '26px'
              }}
            />
          ))}
        </div>

        {/* The Line - The Anchor for the layout */}
        <div className="mb-6" style={{ width: '817px', height: '2px', background: '#90A2B3' }} />

        {/* Graph Controls */}
        <div className="flex flex-col justify-center space-y-4 w-[792px] h-[162px]">
          <h2 style={{ color: '#A9BDE8', fontSize: '25px', fontFamily: 'Inter', fontWeight: 400 }}>
            graph controls
          </h2>
          
          <div className="flex flex-col items-start leading-snug">
            <p style={{ color: '#A9BDE8', fontSize: '20px', fontFamily: 'Inter', fontWeight: 400 }}>
              <span style={{ fontWeight: 700 }}>click</span> on a node for more details
            </p>
            <p style={{ color: '#A9BDE8', fontSize: '20px', fontFamily: 'Inter', fontWeight: 400 }}>
              <span style={{ fontWeight: 700 }}>pinch</span> or <span style={{ fontWeight: 700 }}>scroll</span> to zoom
            </p>
            <p style={{ color: '#A9BDE8', fontSize: '20px', fontFamily: 'Inter', fontWeight: 400 }}>
              <span style={{ fontWeight: 700 }}>click track pad with one finger, move with another</span> to pan
            </p>
            <p style={{ color: '#A9BDE8', fontSize: '20px', fontFamily: 'Inter', fontWeight: 400 }}>
              <span style={{ fontWeight: 700 }}>hold shift while panning</span> to rotate
            </p>
          </div>
        </div>
      </div>

      {/* Home Button */}
      <div className="absolute bottom-10 left-10">
        <button 
          onClick={() => navigate('/')}
          className="flex items-center justify-center transition-all hover:brightness-110 active:scale-95 bg-panel-bg"
          style={{ width: '66px', height: '66px', borderRadius: '26px' }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="41" height="41" viewBox="0 0 41 41" fill="none">
            <path d="M15.375 37.5834V20.5001H25.625V37.5834M5.125 15.3751L20.5 3.41675L35.875 15.3751V34.1667C35.875 35.0729 35.515 35.9419 34.8743 36.5827C34.2335 37.2234 33.3645 37.5834 32.4583 37.5834H8.54167C7.63551 37.5834 6.76647 37.2234 6.12572 36.5827C5.48497 35.9419 5.125 35.0729 5.125 34.1667V15.3751Z" 
              stroke="#5E6E81" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  );
}