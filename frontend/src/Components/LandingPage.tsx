import { useState, useEffect, useRef } from 'react';
import { Search, ChevronRight, Paperclip, X, FileText } from 'lucide-react';
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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showError, setShowError] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
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
    const text = "enter a prompt or upload an article...";
    let timeoutId: ReturnType<typeof setTimeout>;
    let currentIndex = 0;

    const typeChar = () => {
      setPlaceholderText(text.slice(0, currentIndex + 1));
      currentIndex++;
      if (currentIndex < text.length) {
        timeoutId = setTimeout(typeChar, 100); // Typing speed
      }
    };

    timeoutId = setTimeout(typeChar, 500); 

    return () => clearTimeout(timeoutId);
  }, []);

  // Validation handler
  const handleSubmit = () => {
    // Check if both input and file are empty
    if (!inputValue.trim() && !selectedFile) {
      setShowError(true);
      // Remove the error state after the animation finishes
      setTimeout(() => setShowError(false), 2000); 
      return;
    }
    
    // In a real app, you would pass the file and prompt to your global state/backend here
    navigate('/loading'); // Routes to the 5-Model dashboard
  };

  // Trigger navigation when user hits Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  // File selection handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      if (showError) setShowError(false);
    }
    // Reset the input value so the same file can be selected again if removed
    e.target.value = '';
  };

  const removeFile = () => {
    setSelectedFile(null);
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
          
          {/* Hidden File Input */}
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            className="hidden"
            accept=".txt,.pdf,.docx,.md" // Restrict to text-based documents
          />

          {/* Search Input Bar */}
          <div 
            className={`flex items-center px-4 transition-all border backdrop-blur-sm ${
              showError 
                ? 'border-[#FF4757]/50 bg-[#FF4757]/10 animate-shake' 
                : 'bg-input/80 border-border focus-within:border-text-accent focus-within:shadow-[0_0_20px_rgba(169,189,232,0.15)]'
            }`}
            style={{ width: '694px', minHeight: '55px', borderRadius: '26px' }}
          >
            <Search className={`w-5 h-5 mr-3 shrink-0 transition-colors ${showError ? 'text-[#FF4757]' : 'text-text-secondary'}`} />
            
            {/* Display Selected File Pill */}
            {selectedFile && (
              <div className="flex items-center gap-2 px-3 py-1.5 mr-2 bg-[#1A2335] border border-[#2C3A50] rounded-full shrink-0 animate-in fade-in zoom-in duration-200">
                <FileText className="w-3.5 h-3.5 text-[#A9BDE8]" />
                <span className="text-xs font-medium text-[#EBF0FF] max-w-[120px] truncate">
                  {selectedFile.name}
                </span>
                <button 
                  onClick={removeFile}
                  className="p-0.5 rounded-full hover:bg-[#2C3A50] transition-colors"
                >
                  <X className="w-3.5 h-3.5 text-[#FF4757]" />
                </button>
              </div>
            )}

            <input 
              type="text" 
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                if (showError) setShowError(false); // Clear error as soon as they start typing
              }}
              placeholder={selectedFile ? "Add an optional prompt..." : placeholderText}
              onKeyDown={handleKeyDown}
              className={`flex-1 text-base bg-transparent outline-none transition-colors min-w-0 py-2 ${
                showError 
                  ? 'text-red-100 placeholder:text-red-400/60' 
                  : 'text-text-primary placeholder:text-text-secondary'
              }`}
            />
            
            {/* Attachment Button */}
            <button 
              className={`p-2 ml-2 transition-colors rounded-full shrink-0 ${
                selectedFile
                  ? 'text-[#A9BDE8] bg-[#1A2335]'
                  : showError
                    ? 'text-[#FF4757] hover:bg-[#FF4757]/20'
                    : 'text-[#90A2B3] hover:text-[#EBF0FF] hover:bg-[#1A2335]'
              }`}
              onClick={() => fileInputRef.current?.click()}
              title="Upload an article or document"
            >
              <Paperclip className="w-5 h-5" />
            </button>

            {/* Submit Button */}
            <button 
              className={`p-2 ml-1 transition-colors rounded-full shrink-0 ${
                showError 
                  ? 'bg-[#FF4757]/20 text-[#FF4757] hover:bg-[#FF4757]/30' 
                  : 'bg-elevated text-text-secondary hover:text-text-primary hover:bg-[#2C3A50]'
              }`}
              onClick={handleSubmit}
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          {/* Error Message Wrapper (Keeps layout from jumping) */}
          <div className="h-6 mt-3">
            <span 
              className={`text-[#FF4757]/90 text-[14px] tracking-wide transition-all duration-300 ${
                showError ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2 pointer-events-none'
              }`}
            >
              Please enter a prompt or upload a document to begin.
            </span>
          </div>
        </div>
      </main>

    </div>
  );
}