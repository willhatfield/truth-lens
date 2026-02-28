import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, useCursor } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';

// Trust Status Colors
const TRUST_COLORS: Record<string, string> = {
  VerifiedSafe: '#00D68F',
  CautionUnverified: '#FFB020',
  Rejected: '#FF4757',
};

// Model Identity Colors
const MODEL_COLORS: Record<string, string> = {
  'GPT-4 (OpenAI)': '#10A37F',
  'Gemini (Google)': '#428F54',
  'Claude (Anthropic)': '#E8825A',
  'Llama 3 (Meta)': '#A8555F',
  'Kimi (Moonshot)': '#5273FB',
};

const MODELS = Object.keys(MODEL_COLORS);

const MOCK_TEXTS = [
  "Intermittent fasting reduces insulin resistance by 15-30% in adults.",
  "OpenAI was originally founded as a non-profit in December 2015.",
  "Quantum entanglement allows for information transfer faster than light.", // False
  "The Paris Agreement aims to limit global warming to 1.5°C.",
  "Python is the most utilized programming language for machine learning."
];

interface ClaimNode {
  id: string;
  clusterId: number;
  model: string;
  position: [number, number, number];
  trustStatus: 'VerifiedSafe' | 'CautionUnverified' | 'Rejected';
  text: string;
  confidence: number;
}

interface ConstellationProps {
  selectedModels: string[];
}

// --- MOCK DATA GENERATION ---
const generateMockData = (): ClaimNode[] => {
  const nodes: ClaimNode[] = [];
  
  // 5 Main Consensus Clusters
  for (let cluster = 0; cluster < 5; cluster++) {
    const clusterCenter = [
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
    ];
    
    const text = MOCK_TEXTS[cluster];
    const baseConfidence = cluster === 2 ? 15 : 80 + Math.random() * 15;
    
    MODELS.forEach((model, i) => {
      if (Math.random() > 0.2) {
        nodes.push({
          id: `node-${cluster}-${i}`,
          clusterId: cluster,
          model: model,
          position: [
            clusterCenter[0] + (Math.random() - 0.5) * 4,
            clusterCenter[1] + (Math.random() - 0.5) * 4,
            clusterCenter[2] + (Math.random() - 0.5) * 4,
          ],
          trustStatus: cluster === 2 ? 'Rejected' : (Math.random() > 0.8 ? 'CautionUnverified' : 'VerifiedSafe'),
          text: text,
          confidence: Math.floor(baseConfidence + (Math.random() - 0.5) * 10),
        });
      }
    });
  }

  // Scattered Outliers
  for (let i = 0; i < 15; i++) {
    nodes.push({
      id: `outlier-${i}`,
      clusterId: 999 + i, // Unique clusters so they don't connect
      model: MODELS[Math.floor(Math.random() * MODELS.length)],
      position: [
        (Math.random() - 0.5) * 35,
        (Math.random() - 0.5) * 35,
        (Math.random() - 0.5) * 35,
      ],
      trustStatus: Math.random() > 0.5 ? 'Rejected' : 'CautionUnverified',
      text: "This is a hallucinated or completely independent claim made by a single model.",
      confidence: Math.floor(Math.random() * 40),
    });
  }
  return nodes;
};

// --- ANIMATED BEAD COMPONENT ---
function FlowingBead({ start, end, color, delay = 0, speed = 0.3, isDimmed }: { start: number[], end: number[], color: string, delay?: number, speed?: number, isDimmed: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);
  const [progress, setProgress] = useState(-delay);

  const startVec = useMemo(() => new THREE.Vector3(...start), [start]);
  const endVec = useMemo(() => new THREE.Vector3(...end), [end]);
  const currentPos = useMemo(() => new THREE.Vector3(), []);

  useFrame((_state, delta) => {
    if (!meshRef.current || !materialRef.current) return;
    if (isDimmed) {
      materialRef.current.opacity = 0; // Hide beads if line is dimmed
      return;
    }

    let newProgress = progress + delta * speed;
    if (newProgress > 1) newProgress = 0; 
    setProgress(newProgress);

    if (newProgress > 0) {
      currentPos.lerpVectors(startVec, endVec, newProgress);
      meshRef.current.position.copy(currentPos);
      materialRef.current.opacity = Math.sin(newProgress * Math.PI); 
    } else {
      materialRef.current.opacity = 0;
    }
  });

  return (
    <mesh ref={meshRef} position={new THREE.Vector3(...start)}>
      <sphereGeometry args={[0.12, 8, 8]} /> 
      <meshStandardMaterial 
        ref={materialRef}
        color={color} 
        emissive={color} 
        emissiveIntensity={4} 
        transparent 
        depthWrite={false}
      />
    </mesh>
  );
}

// --- INTERACTIVE NODE COMPONENT ---
function InteractiveNode({ 
  node, 
  isSelected, 
  isDimmed, 
  onClick 
}: { 
  node: ClaimNode, 
  isSelected: boolean, 
  isDimmed: boolean, 
  onClick: () => void 
}) {
  const [hovered, setHovered] = useState(false);
  useCursor(hovered); // Automatically changes cursor to pointer!

  // Visual logic
  const scale = isSelected ? 1.6 : hovered ? 1.3 : 1;
  const opacity = isDimmed ? 0.15 : 1;
  const emissiveIntensity = isSelected ? 2 : hovered ? 1.2 : 0.8;

  return (
    <group>
      {/* The Edge Connection */}
      <Line 
        points={[node.position, [0, 0, 0]]}       
        color="#588983" 
        lineWidth={isSelected ? 3 : 1.5} 
        transparent
        opacity={isDimmed ? 0.05 : isSelected ? 0.5 : 0.2} 
      />
      
      {/* Only show beads if not dimmed */}
      <FlowingBead start={node.position} end={[0,0,0]} color={MODEL_COLORS[node.model]} speed={0.2} delay={0} isDimmed={isDimmed} />
      <FlowingBead start={node.position} end={[0,0,0]} color={MODEL_COLORS[node.model]} speed={0.2} delay={0.5} isDimmed={isDimmed} />

      {/* The Sphere */}
      <Sphere 
        position={node.position} 
        args={[0.3, 16, 16]}
        scale={scale}
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial 
          color={TRUST_COLORS[node.trustStatus]} 
          emissive={TRUST_COLORS[node.trustStatus]}
          emissiveIntensity={emissiveIntensity}
          roughness={0.2}
          metalness={0.5}
          transparent
          opacity={opacity}
        />
      </Sphere>
    </group>
  );
}

// --- MAIN GRAPH COMPONENT ---
function RotatingGroup({ children, isPaused }: { children: React.ReactNode, isPaused: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => {
    // Stop rotating if the user is looking at a specific node
    if (groupRef.current && !isPaused) {
      groupRef.current.rotation.y += 0.0005;
    }
  });
  return <group ref={groupRef}>{children}</group>;
}

export default function Constellation({ selectedModels }: ConstellationProps) {
  const mockNodes = useMemo(() => generateMockData(), []);
  const [activeNode, setActiveNode] = useState<ClaimNode | null>(null);

  // Filter nodes based on sidebar toggles
  const visibleNodes = mockNodes.filter(node => selectedModels.includes(node.model));

  // Determine which cluster is currently active (if any) to highlight related connections
  const activeClusterId = activeNode?.clusterId;

  return (
    <div className="relative w-full h-full">
      
      {/* 3D CANVAS */}
      <Canvas 
        camera={{ position: [0, 0, 35], fov: 60 }} 
        gl={{ alpha: true, antialias: true }}
        onPointerMissed={() => setActiveNode(null)} // Clicking empty space clears selection
      >
        <ambientLight intensity={1.2} color="#EBF0FF" />
        <pointLight position={[10, 10, 10]} intensity={2} color="#A9BDE8" />
        <pointLight position={[-10, -10, -10]} intensity={1} color="#588983" />
        
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          dampingFactor={0.05}
        />

        <RotatingGroup isPaused={activeNode !== null}>
          {visibleNodes.map((node) => {
            const isSelected = activeNode?.id === node.id;
            // If a node is selected, dim everything that doesn't share its clusterId
            const isDimmed = activeClusterId !== undefined && activeClusterId !== node.clusterId;

            return (
              <InteractiveNode 
                key={node.id}
                node={node}
                isSelected={isSelected}
                isDimmed={isDimmed}
                onClick={() => setActiveNode(node)}
              />
            );
          })}
          
          {/* Central 'Safe Answer' */}
          <Sphere position={[0,0,0]} args={[1.2, 32, 32]}>
             <meshStandardMaterial 
                color="#EBF0FF" 
                emissive="#EBF0FF" 
                emissiveIntensity={0.3} 
                transparent 
                opacity={activeNode ? 0.05 : 0.15} // Dim center when investigating a claim
             />
          </Sphere>
        </RotatingGroup>
      </Canvas>

      {/* 2D HTML OVERLAY PANEL */}
      <AnimatePresence>
        {activeNode && (
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 50 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute top-8 right-8 w-80 flex flex-col gap-4 z-50"
            // Stop clicks on the panel from passing through to the 3D canvas
            onPointerDown={(e) => e.stopPropagation()} 
          >
            {/* CLAIM CARD */}
            <div className="bg-[#121825] border border-[#2C3A50] rounded-xl overflow-hidden shadow-2xl">
              <div className="bg-[#1A2335] px-4 py-2 border-b border-[#2C3A50]">
                <span className="text-[#90A2B3] text-xs font-bold tracking-widest uppercase">Claim</span>
              </div>
              <div className="p-4">
                <p className="text-[#EBF0FF] text-[15px] leading-relaxed">
                  "{activeNode.text}"
                </p>
                
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-[#90A2B3] text-xs">Generated by:</span>
                  <span 
                    className="text-xs font-semibold px-2.5 py-1 rounded border"
                    style={{ 
                      color: MODEL_COLORS[activeNode.model], 
                      backgroundColor: `${MODEL_COLORS[activeNode.model]}20`,
                      borderColor: `${MODEL_COLORS[activeNode.model]}50`
                    }}
                  >
                    {activeNode.model.split(' ')[0]}
                  </span>
                </div>
              </div>
            </div>

            {/* TRUST STATUS CARD */}
            <div className="bg-[#121825] border border-[#2C3A50] rounded-xl overflow-hidden shadow-2xl">
              <div className="bg-[#1A2335] px-4 py-2 border-b border-[#2C3A50]">
                <span className="text-[#90A2B3] text-xs font-bold tracking-widest uppercase">Trust Status</span>
              </div>
              <div className="p-4">
                
                {/* Progress Bar Container */}
                <div className="w-full h-3 bg-[#1A2335] rounded-full overflow-hidden mb-3">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${activeNode.confidence}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="h-full"
                    style={{ backgroundColor: TRUST_COLORS[activeNode.trustStatus] }}
                  />
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-3xl font-bold text-[#EBF0FF]">{activeNode.confidence}%</span>
                  <p className="text-[#90A2B3] text-sm leading-tight">
                    this claim is <span style={{ color: TRUST_COLORS[activeNode.trustStatus] }} className="font-semibold lowercase">{activeNode.trustStatus.replace(/([A-Z])/g, ' $1').trim()}</span>
                  </p>
                </div>
                
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}