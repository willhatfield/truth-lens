import type { ReactNode } from 'react';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    // The main wrapper: full screen, deep background color, standard text color
    <div className="relative flex w-screen h-screen overflow-hidden bg-deep-bg text-text-primary">
      
      {/* Background GIF Layer - set to low opacity so it doesn't overpower the 3D constellation */}
      <div 
        className="absolute inset-0 z-0 opacity-30 pointer-events-none"
        style={{ 
          backgroundImage: "url('public/background-grid.svg')", 
          backgroundSize: 'cover',
          backgroundPosition: 'center' 
        }}
      />

      {/* Sidebar Placeholder */}
      <aside className="relative z-10 flex flex-col w-64 border-r bg-panel-bg border-border-light">
        <div className="p-4 border-b border-border-light">
          {/* We will build out the toggle menu and Legend here */}
          <h1 className="text-xl font-bold text-text-primary tracking-wide">TruthLens</h1>
        </div>
        <div className="p-4 border-b border-border-light flex justify-between items-center">
          <h2 className="text-lg font-semibold text-text-primary">Visualizations</h2>
          <button 
            onClick={() => setShowVisualizationMenu(!showVisualizationMenu)}
            className="px-3 py-2 bg-accent-blue hover:bg-accent-blue-dark text-white text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-accent-blue focus:ring-opacity-50"
          >
            {showVisualizationMenu ? 'Hide Menu' : 'Open Menu'}
          </button>
        </div>
        {showVisualizationMenu && (
          <div className="p-4 bg-border-light w-[240px] h-[215px] rounded-b-[10px]">
            <ul className="space-y-2">
              <li className="flex w-[132px] h-[26px] justify-center items-center text-text-primary hover:text-accent-blue cursor-pointer">Constellation</li>
              <li className="flex w-[132px] h-[26px] justify-center items-center text-text-primary hover:text-accent-blue cursor-pointer">Pipeline</li>
              <li className="flex w-[132px] h-[26px] justify-center items-center text-text-primary hover:text-accent-blue cursor-pointer">Heat Map</li>
              <li className="flex w-[132px] h-[26px] justify-center items-center text-text-primary hover:text-accent-blue cursor-pointer">Evidence Network</li>
              <li className="flex w-[132px] h-[26px] justify-center items-center text-text-primary hover:text-accent-blue cursor-pointer">Knowledge Deck</li>
            </ul>
          </div>
        )}
        <div className="flex-1 p-4">
           <p className="text-sm text-text-secondary">Legend & Controls coming soon...</p>
        </div>
      </aside>

      {/* Main Stage (This is where the 3D Constellation or Knowledge Deck will go) */}
      <main className="relative z-10 flex-1 h-full">
        {children}
      </main>

    </div>
  );
}