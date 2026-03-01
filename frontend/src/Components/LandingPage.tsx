import { useState, useEffect, useRef } from 'react';
import { Search, ChevronRight, Paperclip, X, FileText, Clock, User, ShieldCheck, AlertCircle, Activity } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { API_BASE } from '../config';
import React from 'react';

// Custom component to recreate the Figma glowing variant
const NavButton = ({ children, width, onClick }: { children: React.ReactNode; width: string, onClick?: () => void }) => {
  return (
    <button 
      onClick={onClick}
      className="relative flex items-center justify-center transition-colors group text-[#90A2B3] hover:text-[#EBF0FF] font-bold font-['Inter']"
      style={{ width: width, height: '40px', fontSize: '16px' }}
    >
      <span className="relative z-10 flex items-center justify-center w-full gap-2">{children}</span>
      <span className="absolute inset-0 z-0 transition-opacity duration-300 rounded-full opacity-0 blur-md bg-[#A9BDE8]/30 group-hover:opacity-100" />
    </button>
  );
};

// --- MOCK HISTORY DATA ---
const MOCK_HISTORY = [
  { id: 1, query: "Is intermittent fasting actually effective for fixing insulin resistance...", date: "Today, 10:42 AM", score: 92, status: 'VerifiedSafe' },
  { id: 2, query: "Did OpenAI transition to a fully for-profit model in 2024?", date: "Yesterday", score: 88, status: 'VerifiedSafe' },
  { id: 3, query: "Does quantum entanglement allow for faster-than-light communication?", date: "Oct 12, 2025", score: 15, status: 'Rejected' },
];

export default function LandingPage() {
  const navigate = useNavigate();

  // Main Search States
  const [placeholderText, setPlaceholderText] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showError, setShowError] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");

  // Auth & History States
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [showAuthScreen, setShowAuthScreen] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('signup');
  
  // Auth Form States
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [authError, setAuthError] = useState(false);
  const [authErrorMsg, setAuthErrorMsg] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // CSS Injection for the shake animation
  const shakeKeyframes = `
  @keyframes shake {
    10%, 90% { transform: translate3d(-1px, 0, 0); }
    20%, 80% { transform: translate3d(2px, 0, 0); }
    30%, 50%, 70% { transform: translate3d(-4px, 0, 0); }
    40%, 60% { transform: translate3d(4px, 0, 0); }
  }
  .animate-shake { animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both; }

  /* ---- TRUTHLens Glow Animation ---- */

  @keyframes waveMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  
  .truthlens-glow {
    position: relative;
    background: linear-gradient(
      120deg,
      #90A2B3 0%,
      #A9BDE8 30%,
      #EBF0FF 50%,
      #A9BDE8 70%,
      #90A2B3 100%
    );
    background-size: 300% 300%;
    animation: waveMove 10s ease-in-out infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: brightness(1.05);
  }
  
  .truthlens-glow::after {
    content: attr(data-text);
    position: absolute;
    inset: 0;
    background: linear-gradient(
      120deg,
      transparent 40%,
      #EBF0FF 50%,
      transparent 60%
    );
    background-size: 300% 300%;
    animation: waveMove 10s ease-in-out infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: blur(14px);
    opacity: 0.7;
    pointer-events: none;
  }

  /* Hide scrollbar for history panel */
  .no-scrollbar::-webkit-scrollbar { display: none; }
  .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
`;

  // On Mount: Check LocalStorage to see if already logged in
  useEffect(() => {
    const authStatus = localStorage.getItem('truthlens_auth');
    if (authStatus === 'true') {
      setIsLoggedIn(true);
    }

    const text = "enter a prompt or upload an article...";
    let timeoutId: ReturnType<typeof setTimeout>;
    let currentIndex = 0;

    const typeChar = () => {
      setPlaceholderText(text.slice(0, currentIndex + 1));
      currentIndex++;
      if (currentIndex < text.length) {
        timeoutId = setTimeout(typeChar, 100);
      }
    };

    timeoutId = setTimeout(typeChar, 500); 
    return () => clearTimeout(timeoutId);
  }, []);

  // Validation handler for main search
  const handleSubmit = async () => {
    if (!inputValue.trim() && !selectedFile) {
      setShowError(true);
      setTimeout(() => setShowError(false), 2000);
      return;
    }
    setIsSubmitting(true);
    setSubmitError("");
    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: inputValue }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const { analysis_id } = await res.json();
      navigate(`/loading?job=${analysis_id}`);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Failed to start analysis");
      setIsSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSubmit();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      if (showError) setShowError(false);
    }
    e.target.value = '';
  };

  // Validation handler for Auth
  const handleAuthSubmit = () => {
    // Check Empties
    if (!email.trim() || !password.trim() || (authMode === 'signup' && (!username.trim() || !confirmPassword.trim()))) {
      setAuthErrorMsg("Please fill in all required fields.");
      setAuthError(true);
      setTimeout(() => setAuthError(false), 2000);
      return;
    }

    // Check Passwords Match
    if (authMode === 'signup' && password !== confirmPassword) {
      setAuthErrorMsg("Passwords do not match.");
      setAuthError(true);
      setTimeout(() => setAuthError(false), 2000);
      return;
    }

    // Simulate successful login/signup
    localStorage.setItem('truthlens_auth', 'true'); // Save to local storage!
    setIsLoggedIn(true);
    setShowAuthScreen(false);
    setEmail(""); setUsername(""); setPassword(""); setConfirmPassword("");
  };

  const handleLogout = () => {
    localStorage.removeItem('truthlens_auth');
    setIsLoggedIn(false);
    setShowHistory(false);
  };

  return (
    <div className="relative flex flex-col w-screen h-screen overflow-hidden bg-[#0A0E1A] text-[#EBF0FF]">
      <style>{shakeKeyframes}</style>

      {/* Background GIF Layer */}
      <div 
        className="absolute inset-0 z-0 opacity-[0.05] pointer-events-none"
        style={{ backgroundImage: "url('/Gridgif-ezgif.com-speed.gif')", backgroundSize: 'cover', backgroundPosition: 'center' }}
      />

      {/* Top Navigation */}
      <nav className="relative z-20 flex items-center justify-between w-full px-12 py-6">
        <div className="flex justify-start flex-1">
          <div className="relative" style={{ fontSize: '45px' }}>
            {/* Main Animated Text */}
            <div className="relative truthlens-glow font-bold font-['Inter'] leading-none">
              TRUTHLens
            </div>
          </div>
        </div>
        
        <div className="flex items-center px-6 bg-[#121825] border border-[#2C3A50]/50" style={{ height: '55px', borderRadius: '26px' }}>
          <NavButton width="95px">Home</NavButton>
          <NavButton width="95px">About</NavButton>
          <NavButton width="95px">Pricing</NavButton>
          <NavButton width="120px">Discovery</NavButton>
          
          {isLoggedIn && (
            <>
              <div className="w-px h-6 bg-[#2C3A50] mx-2" />
              <NavButton width="130px" onClick={() => setShowHistory(true)}>
                <Clock className="w-4 h-4" /> History
              </NavButton>
            </>
          )}
        </div>

        <div className="flex justify-end flex-1">
          {isLoggedIn ? (
            <button 
              onClick={handleLogout}
              className="relative flex items-center justify-center transition-colors group bg-[#121825] border border-[#2C3A50]/50 text-[#90A2B3] hover:text-[#EBF0FF] shrink-0 font-bold font-['Inter']"
              style={{ width: '120px', height: '55px', borderRadius: '26px', fontSize: '16px' }}
            >
              <span className="relative z-10 flex items-center gap-2"><User className="w-4 h-4"/> Sign Out</span>
              <span className="absolute inset-0 z-0 transition-opacity duration-300 opacity-0 blur-md bg-[#A9BDE8]/20 group-hover:opacity-100 rounded-full" />
            </button>
          ) : (
            <button 
              onClick={() => setShowAuthScreen(true)}
              className="relative flex items-center justify-center transition-colors group bg-[#121825] border border-[#2C3A50]/50 text-[#90A2B3] hover:text-[#EBF0FF] shrink-0 font-bold font-['Inter']"
              style={{ width: '95px', height: '55px', borderRadius: '26px', fontSize: '16px' }}
            >
              <span className="relative z-10">Login</span>
              <span className="absolute inset-0 z-0 transition-opacity duration-300 opacity-0 blur-md bg-[#A9BDE8]/20 group-hover:opacity-100 rounded-full" />
            </button>
          )}
        </div>
      </nav>

      {/* Main Hero Content */}
      <main className="relative z-10 flex flex-col items-center justify-center flex-1 px-4">
        <h1 className="max-w-3xl mx-auto mb-4 text-5xl font-extrabold tracking-tight text-center md:text-6xl text-[#EBF0FF]">
          the AI trust <br /> intelligence platform
        </h1>
        <p className="mb-[120px] text-lg text-[#90A2B3] text-center mx-auto">
          Get verified consensus, not just a single hallucination.
        </p>

        {/* Search Container */}
        <div className="flex flex-col items-center">
          <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".txt,.pdf,.docx,.md" />

          <div 
            className={`flex items-center px-4 transition-all border backdrop-blur-sm ${
              showError ? 'border-[#FF4757]/50 bg-[#FF4757]/10 animate-shake' : 'bg-[#161D2E]/80 border-[#2C3A50] focus-within:border-[#A9BDE8] focus-within:shadow-[0_0_20px_rgba(169,189,232,0.15)]'
            }`}
            style={{ width: '694px', minHeight: '55px', borderRadius: '26px' }}
          >
            <Search className={`w-5 h-5 mr-3 shrink-0 transition-colors ${showError ? 'text-[#FF4757]' : 'text-[#90A2B3]'}`} />
            
            {selectedFile && (
              <div className="flex items-center gap-2 px-3 py-1.5 mr-2 bg-[#1A2335] border border-[#2C3A50] rounded-full shrink-0 animate-in fade-in zoom-in duration-200">
                <FileText className="w-3.5 h-3.5 text-[#A9BDE8]" />
                <span className="text-xs font-medium text-[#EBF0FF] max-w-[120px] truncate">{selectedFile.name}</span>
                <button onClick={() => setSelectedFile(null)} className="p-0.5 rounded-full hover:bg-[#2C3A50] transition-colors">
                  <X className="w-3.5 h-3.5 text-[#FF4757]" />
                </button>
              </div>
            )}

            <input 
              type="text" 
              value={inputValue}
              onChange={(e) => { setInputValue(e.target.value); if (showError) setShowError(false); }}
              placeholder={selectedFile ? "Add an optional prompt..." : placeholderText}
              onKeyDown={handleKeyDown}
              className={`flex-1 text-base bg-transparent outline-none transition-colors min-w-0 py-2 ${
                showError ? 'text-red-100 placeholder:text-red-400/60' : 'text-[#EBF0FF] placeholder:text-[#90A2B3]'
              }`}
            />
            
            <button 
              className={`p-2 ml-2 transition-colors rounded-full shrink-0 ${
                selectedFile ? 'text-[#A9BDE8] bg-[#1A2335]' : showError ? 'text-[#FF4757] hover:bg-[#FF4757]/20' : 'text-[#90A2B3] hover:text-[#EBF0FF] hover:bg-[#1A2335]'
              }`}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip className="w-5 h-5" />
            </button>

            <button
              className={`p-2 ml-1 transition-colors rounded-full shrink-0 disabled:opacity-50 ${
                showError ? 'bg-[#FF4757]/20 text-[#FF4757] hover:bg-[#FF4757]/30' : 'bg-[#1A2335] text-[#90A2B3] hover:text-[#EBF0FF] hover:bg-[#2C3A50]'
              }`}
              onClick={handleSubmit}
              disabled={isSubmitting}
            >
              {isSubmitting ? <Activity className="w-5 h-5 animate-spin" /> : <ChevronRight className="w-5 h-5" />}
            </button>
          </div>

          <div className="h-6 mt-3">
            <span className={`text-[#FF4757]/90 text-[14px] tracking-wide transition-all duration-300 ${showError ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2 pointer-events-none'}`}>
              Please enter a prompt or upload a document to begin.
            </span>
            {submitError && (
              <span className="text-[#FF4757]/90 text-[14px] tracking-wide block mt-1">
                {submitError}
              </span>
            )}
          </div>
        </div>
      </main>

      {/* AUTHENTICATION OVERLAY */}
      {showAuthScreen && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-[#0A0E1A]/80 backdrop-blur-md animate-in fade-in duration-200">
          <div className="bg-[#121825] border border-[#2C3A50] rounded-3xl p-8 w-full max-w-md shadow-2xl relative">
            <button onClick={() => setShowAuthScreen(false)} className="absolute top-6 right-6 text-[#90A2B3] hover:text-[#EBF0FF] transition-colors"><X className="w-5 h-5" /></button>

            <h2 className="text-3xl font-bold text-[#EBF0FF] mb-2 text-center">
              {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
            </h2>
            <p className="text-[#90A2B3] text-center mb-8">
              {authMode === 'login' ? 'Access your verified history.' : 'Join the consensus engine.'}
            </p>

            <div className={`space-y-4 ${authError ? 'animate-shake' : ''}`}>
              <div>
                <label className="block text-xs font-bold tracking-widest text-[#588983] uppercase mb-2 ml-1">Email</label>
                <input 
                  type="email" value={email} onChange={(e) => { setEmail(e.target.value); setAuthError(false); }}
                  className={`w-full bg-[#161D2E] border ${authError && !email ? 'border-[#FF4757]' : 'border-[#2C3A50]'} rounded-xl px-4 py-3 text-[#EBF0FF] focus:outline-none focus:border-[#A9BDE8] transition-colors ${authError && !email ? 'placeholder:text-[#FF4757]/60' : 'placeholder:text-[#90A2B3]/50'}`}
                  placeholder="name@example.com"
                />
              </div>

              {authMode === 'signup' && (
                <div>
                  <label className="block text-xs font-bold tracking-widest text-[#588983] uppercase mb-2 ml-1">Username</label>
                  <input 
                    type="text" value={username} onChange={(e) => { setUsername(e.target.value); setAuthError(false); }}
                    className={`w-full bg-[#161D2E] border ${authError && !username ? 'border-[#FF4757]' : 'border-[#2C3A50]'} rounded-xl px-4 py-3 text-[#EBF0FF] focus:outline-none focus:border-[#A9BDE8] transition-colors ${authError && !username ? 'placeholder:text-[#FF4757]/60' : 'placeholder:text-[#90A2B3]/50'}`}
                    placeholder="truthseeker99"
                  />
                </div>
              )}

              <div>
                <label className="block text-xs font-bold tracking-widest text-[#588983] uppercase mb-2 ml-1">Password</label>
                <input 
                  type="password" value={password} onChange={(e) => { setPassword(e.target.value); setAuthError(false); }}
                  className={`w-full bg-[#161D2E] border ${authError && (!password || (authMode === 'signup' && password !== confirmPassword)) ? 'border-[#FF4757]' : 'border-[#2C3A50]'} rounded-xl px-4 py-3 text-[#EBF0FF] focus:outline-none focus:border-[#A9BDE8] transition-colors ${authError && !password ? 'placeholder:text-[#FF4757]/60' : 'placeholder:text-[#90A2B3]/50'}`}
                  placeholder="••••••••"
                />
              </div>

              {authMode === 'signup' && (
                <div>
                  <label className="block text-xs font-bold tracking-widest text-[#588983] uppercase mb-2 ml-1">Confirm Password</label>
                  <input 
                    type="password" value={confirmPassword} onChange={(e) => { setConfirmPassword(e.target.value); setAuthError(false); }}
                    className={`w-full bg-[#161D2E] border ${authError && (!confirmPassword || password !== confirmPassword) ? 'border-[#FF4757]' : 'border-[#2C3A50]'} rounded-xl px-4 py-3 text-[#EBF0FF] focus:outline-none focus:border-[#A9BDE8] transition-colors ${authError && !confirmPassword ? 'placeholder:text-[#FF4757]/60' : 'placeholder:text-[#90A2B3]/50'}`}
                    placeholder="••••••••"
                  />
                </div>
              )}

              <button 
                onClick={handleAuthSubmit}
                className="w-full mt-6 bg-gradient-to-r from-[#1A2335] to-[#2C3A50] hover:to-[#3D2E50] border border-[#588983]/30 hover:border-[#588983] text-[#EBF0FF] font-bold py-3 rounded-xl transition-all"
              >
                {authMode === 'login' ? 'Sign In' : 'Sign Up'}
              </button>

              {authError && (
                <p className="text-[#FF4757] text-sm text-center mt-2">{authErrorMsg}</p>
              )}
            </div>

            <div className="mt-8 pt-6 border-t border-[#2C3A50] text-center">
              <span className="text-[#90A2B3] text-sm">{authMode === 'login' ? "Don't have an account? " : "Already have an account? "}</span>
              <button 
                onClick={() => { setAuthMode(authMode === 'login' ? 'signup' : 'login'); setAuthError(false); }}
                className="text-[#A9BDE8] font-bold text-sm hover:underline"
              >
                {authMode === 'login' ? 'Sign Up' : 'Log In'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* HISTORY SIDEBAR PANEL */}
      <AnimatePresence>
        {showHistory && (
          <>
            {/* Dark Overlay to close when clicking outside */}
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setShowHistory(false)}
              className="absolute inset-0 bg-[#0A0E1A]/40 backdrop-blur-sm z-40"
            />
            
            {/* The Sidebar itself */}
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
                    onClick={() => navigate('/arena')} // Goes to the arena assuming it loads this chat
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
