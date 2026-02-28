import { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Database, Zap, GripVertical } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function LiveHUD() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [claims, setClaims] = useState(142);
  const [entropy, setEntropy] = useState(1.24);
  
  // Define safe boundaries so the user can't drag the HUD off-screen
  const [bounds, setBounds] = useState({ left: -500, right: 500, top: -10, bottom: 800 });

  // Compute boundaries dynamically based on browser window size
  useEffect(() => {
    const updateBounds = () => {
      setBounds({
        left: -(window.innerWidth / 2) + 150, // HUD starts centered, so left bound is negative
        right: (window.innerWidth / 2) - 150,
        top: -10, // Allow bumping slightly into the top margin
        bottom: window.innerHeight - 100 // Prevent dragging off the bottom
      });
    };
    
    updateBounds();
    window.addEventListener('resize', updateBounds);
    return () => window.removeEventListener('resize', updateBounds);
  }, []);

  // Simulate live background processing
  useEffect(() => {
    const interval = setInterval(() => {
      if (Math.random() > 0.7) setClaims(c => c + 1);
      if (Math.random() > 0.5) setEntropy(e => +(e + (Math.random() * 0.04 - 0.02)).toFixed(2));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="absolute top-6 left-1/2 -translate-x-1/2 z-50 flex justify-center">
      
      <motion.div
        layout
        // --- DRAG PROPERTIES ---
        drag
        dragConstraints={bounds}
        dragMomentum={false} // Stops exactly where you drop it (no sliding ice physics)
        whileDrag={{ scale: 1.05, cursor: "grabbing" }} // Pops up slightly when grabbed
        // -----------------------
        onClick={() => setIsExpanded(!isExpanded)}
        className="cursor-grab overflow-hidden bg-[#121825]/80 backdrop-blur-2xl border border-[#2C3A50]/80 shadow-[0_8px_32px_rgba(0,0,0,0.5)] transition-colors hover:border-[#5E6E81]/80"
        style={{ borderRadius: isExpanded ? '16px' : '32px' }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <div className={`flex items-center transition-all ${isExpanded ? 'p-2' : 'px-4 py-2.5'}`}>
          
          {/* Header / Toggle Button */}
          <motion.div layout className="flex items-center gap-2 px-2">
            
            {/* Tiny Drag Handle Icon */}
            <GripVertical className="w-3.5 h-3.5 text-[#5E6E81] opacity-60 hover:text-[#EBF0FF] transition-colors shrink-0" />
            
            <div className="relative flex h-2.5 w-2.5 shrink-0 ml-1">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#FF4757] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#FF4757] shadow-[0_0_8px_#FF4757]"></span>
            </div>
            <motion.span layout className="text-[#EBF0FF] text-[11px] font-bold tracking-[0.2em] uppercase whitespace-nowrap ml-1">
              Live HUD
            </motion.span>
          </motion.div>

          {/* Expanded Metrics */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                layout
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.2, ease: "easeInOut" }}
                className="flex items-center overflow-hidden shrink-0"
              >
                {/* Subtle Divider */}
                <div className="w-px h-8 bg-[#2C3A50]/50 mx-2 shrink-0" />
                
                {/* Metric Cards */}
                <div className="flex gap-2 shrink-0">
                  
                  {/* Metric 1 */}
                  <div className="flex items-center gap-3 bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-lg px-3 py-1.5 shadow-inner hover:bg-[#1A2335] transition-colors">
                    <Database className="w-4 h-4 text-[#A9BDE8]" />
                    <div className="flex flex-col">
                      <span className="text-[#5E6E81] text-[8px] font-bold tracking-widest uppercase">Claims Extracted</span>
                      <span className="text-[#EBF0FF] text-sm font-mono font-semibold leading-tight drop-shadow-md">{claims}</span>
                    </div>
                  </div>

                  {/* Metric 2 */}
                  <div className="flex items-center gap-3 bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-lg px-3 py-1.5 shadow-inner hover:bg-[#1A2335] transition-colors">
                    <ShieldCheck className="w-4 h-4 text-[#00D68F]" />
                    <div className="flex flex-col">
                      <span className="text-[#5E6E81] text-[8px] font-bold tracking-widest uppercase">Verification Rate</span>
                      <span className="text-[#00D68F] text-sm font-mono font-semibold leading-tight drop-shadow-[0_0_8px_rgba(0,214,143,0.4)]">84.2%</span>
                    </div>
                  </div>

                  {/* Metric 3 */}
                  <div className="flex items-center gap-3 bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-lg px-3 py-1.5 shadow-inner hover:bg-[#1A2335] transition-colors">
                    <Activity className="w-4 h-4 text-[#FFB020]" />
                    <div className="flex flex-col">
                      <span className="text-[#5E6E81] text-[8px] font-bold tracking-widest uppercase">Semantic Entropy</span>
                      <span className="text-[#FFB020] text-sm font-mono font-semibold leading-tight drop-shadow-[0_0_8px_rgba(255,176,32,0.4)]">{entropy} nats</span>
                    </div>
                  </div>

                  {/* Metric 4 */}
                  <div className="flex items-center gap-3 bg-[#1A2335]/60 border border-[#2C3A50]/80 rounded-lg px-3 py-1.5 shadow-inner hover:bg-[#1A2335] transition-colors">
                    <Zap className="w-4 h-4 text-[#E8825A]" />
                    <div className="flex flex-col">
                      <span className="text-[#5E6E81] text-[8px] font-bold tracking-widest uppercase">Model Drift</span>
                      <span className="text-[#E8825A] text-sm font-mono font-semibold leading-tight drop-shadow-[0_0_8px_rgba(232,130,90,0.4)]">0.14</span>
                    </div>
                  </div>

                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
        </div>
      </motion.div>
    </div>
  );
}