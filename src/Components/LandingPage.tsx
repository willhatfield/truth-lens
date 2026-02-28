import { useState, useEffect } from 'react';
import { Search, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

// Custom component to recreate the Figma glowing variant
const NavButton = ({ children, width }: { children: string; width: string }) => {
  return (
    <button 
      className="relative flex items-center justify-center transition-colors group text-text-secondary hover:text-text-primary font-bold font-['Inter']"
      style={{ 
        width: width, 
        height: '40px', // Matches your Figma height spec
        fontSize: '16px' 
      }}
    >
      <span className="relative z-10">{children}</span>
      <span className="absolute inset-0 z-0 transition-opacity duration-300 rounded-full opacity-0 blur-md bg-text-accent/30 group-hover:opacity-100" />
    </button>
  );
};

export default function LandingPage() {
  const [placeholderText, setPlaceholderText] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [showError, setShowError] = useState(false);
  const navigate = useNavigate();

  // CSS Injection for the shake animation
  const shakeKeyframes = `
    @keyframes shake {
      10%, 90% { transform: translate3d(-1px, 0, 0); }
      20%, 80% { transform: translate3d(2px, 0, 0); }
      30%, 50%, 70% { transform: translate3d(-4px, 0, 0); }
      40%, 60% { transform: translate3d(4px, 0, 0); }
    }
    .animate-shake {
      animation: shake 0.5s cubic-bezier(.36,.07,.19,.97) both;
    }
  `;

  // Bulletproof Typewriter effect
  useEffect(() => {
    const text = "enter a prompt...";
    let timeoutId: ReturnType<typeof setTimeout>;
    let currentIndex = 0;

    const typeChar = () => {
      setPlaceholderText(text.slice(0, currentIndex + 1));
      currentIndex++;
      if (currentIndex < text.length) {
        timeoutId = setTimeout(typeChar, 150); // Typing speed
      }
    };

    timeoutId = setTimeout(typeChar, 500); 

    return () => clearTimeout(timeoutId);
  }, []);

  // Validation handler
  const handleSubmit = () => {
    if (!inputValue.trim()) {
      setShowError(true);
      // Remove the error state after the animation finishes
      setTimeout(() => setShowError(false), 2000); 
      return;
    }
    navigate('/loading'); // Routes to the 5-Model dashboard
  };

  // Trigger navigation when user hits Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="relative flex flex-col w-screen h-screen overflow-hidden bg-deep-bg text-text-primary">
      <style>{shakeKeyframes}</style>

      {/* Background GIF Layer */}
      <div 
        className="absolute inset-0 z-0 opacity-5 pointer-events-none"
        style={{ 
          backgroundImage: "url('/Gridgif-ezgif.com-speed.gif')", 
          backgroundSize: 'cover',
          backgroundPosition: 'center' 
        }}
      />

      {/* Top Navigation */}
      <nav className="relative z-10 flex items-center justify-between w-full px-12 py-6">
        
        {/* Left: Logo */}
        <div className="flex justify-start flex-1">
          <div style={{ fontSize: '45px' }} className="font-bold text-text-secondary font-['Inter'] leading-none">
            TRUTHLens
          </div>
        </div>
        
        {/* Center: Navigation Links */}
        <div 
          className="flex items-center px-6 bg-panel-bg"
          style={{ height: '55px', borderRadius: '26px' }}
        >
          <NavButton width="95px">Home</NavButton>
          <NavButton width="95px">About</NavButton>
          <NavButton width="95px">Pricing</NavButton>
          <NavButton width="152px">Discovery</NavButton>
        </div>

        {/* Right: Login Button */}
        <div className="flex justify-end flex-1">
          <button 
            className="relative flex items-center justify-center transition-colors group bg-panel-bg text-text-secondary hover:text-text-primary shrink-0 font-bold font-['Inter']"
            style={{ 
              width: '95px', 
              height: '55px', 
              borderRadius: '26px',
              fontSize: '16px'
            }}
          >
            <span className="relative z-10">Login</span>
            <span 
              className="absolute inset-0 z-0 transition-opacity duration-300 opacity-0 blur-md bg-text-accent/30 group-hover:opacity-100" 
              style={{ borderRadius: '26px' }} 
            />
          </button>
        </div>

      </nav>

      {/* Hero Content */}
      <main className="relative z-10 flex flex-col items-center justify-center flex-1 px-4 text-center">
        <h1 className="max-w-3xl mb-4 text-5xl font-extrabold tracking-tight md:text-6xl text-text-primary">
          the AI trust <br /> intelligence platform
        </h1>
        
        {/* Space Between Search and Header */}
        <p className="mb-[120px] text-lg text-text-secondary">
          Get verified consensus, not just a single hallucination.
        </p>

        {/* Search Container */}
        <div className="flex flex-col items-center">
          {/* Search Input Bar */}
          <div 
            className={`flex items-center px-4 transition-all border backdrop-blur-sm ${
              showError 
                ? 'border-red-500/50 bg-red-500/10 animate-shake' 
                : 'bg-input/80 border-border focus-within:border-text-accent focus-within:shadow-[0_0_20px_rgba(169,189,232,0.15)]'
            }`}
            style={{ width: '694px', height: '55px', borderRadius: '26px' }}
          >
            <Search className={`w-5 h-5 mr-3 transition-colors ${showError ? 'text-red-400' : 'text-text-secondary'}`} />
            <input 
              type="text" 
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                if (showError) setShowError(false); // Clear error as soon as they start typing
              }}
              placeholder={placeholderText}
              onKeyDown={handleKeyDown}
              className={`flex-1 text-base bg-transparent outline-none transition-colors ${
                showError 
                  ? 'text-red-100 placeholder:text-red-400/60' 
                  : 'text-text-primary placeholder:text-text-secondary'
              }`}
            />
            <button 
              className={`p-2 ml-2 transition-colors rounded-full ${
                showError 
                  ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                  : 'bg-elevated text-text-secondary hover:text-text-primary'
              }`}
              onClick={handleSubmit}
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          {/* Error Message Wrapper (Keeps layout from jumping) */}
          <div className="h-6 mt-3">
            <span 
              className={`text-red-400/90 text-[15px] transition-all duration-300 ${
                showError ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2 pointer-events-none'
              }`}
            >
              Please enter a prompt to begin your search.
            </span>
          </div>
        </div>
      </main>

    </div>
  );
}