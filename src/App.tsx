import { Routes, Route } from 'react-router-dom';
import LandingPage from './Components/LandingPage';
import LoadingPage from './Components/LoadingPage';
import ArenaPage from './Components/ArenaPage';

function App() {
  return (
    <Routes>
      {/* 1. The Entry Point */}
      <Route path="/" element={<LandingPage />} />
      
      {/* 2. The Animated Transition (50 blocks + instructions) */}
      <Route path="/loading" element={<LoadingPage />} />
      
      {/* 3. The 3D Canvas + Dashboard */}
      <Route path="/arena" element={<ArenaPage />} />
    </Routes>
  );
}

export default App;