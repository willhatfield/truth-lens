import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

// Trust Status Colors (for the claim nodes)
const TRUST_COLORS: Record<string, string> = {
  VerifiedSafe: '#00D68F',
  CautionUnverified: '#FFB020',
  Rejected: '#FF4757',
};

// Model Identity Colors (for the flowing beads)
const MODEL_COLORS: Record<string, string> = {
  'GPT-4 (OpenAI)': '#10A37F',
  'Gemini (Google)': '#428F54',
  'Claude (Anthropic)': '#E8825A',
  'Llama 3 (Meta)': '#A8555F',
  'Kimi (Moonshot)': '#5273FB',
};

const MODELS = Object.keys(MODEL_COLORS);

interface ClaimNode {
  id: string;
  model: string;
  position: [number, number, number];
  trustStatus: 'VerifiedSafe' | 'CautionUnverified' | 'Rejected';
}

interface ConstellationProps {
  selectedModels: string[];
}

// --- MOCK DATA GENERATION ---
const generateMockData = (): ClaimNode[] => {
  const nodes: ClaimNode[] = [];
  for (let cluster = 0; cluster < 5; cluster++) {
    const clusterCenter = [
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 20,
    ];
    MODELS.forEach((model, i) => {
      if (Math.random() > 0.2) {
        nodes.push({
          id: `node-${cluster}-${i}`,
          model: model,
          position: [
            clusterCenter[0] + (Math.random() - 0.5) * 3,
            clusterCenter[1] + (Math.random() - 0.5) * 3,
            clusterCenter[2] + (Math.random() - 0.5) * 3,
          ],
          trustStatus: Math.random() > 0.7 ? 'CautionUnverified' : 'VerifiedSafe',
        });
      }
    });
  }
  for (let i = 0; i < 20; i++) {
    nodes.push({
      id: `outlier-${i}`,
      model: MODELS[Math.floor(Math.random() * MODELS.length)],
      position: [
        (Math.random() - 0.5) * 35,
        (Math.random() - 0.5) * 35,
        (Math.random() - 0.5) * 35,
      ],
      trustStatus: Math.random() > 0.5 ? 'Rejected' : 'CautionUnverified',
    });
  }
  return nodes;
};

// --- ANIMATED BEAD COMPONENT ---
function FlowingBead({ start, end, color, delay = 0, speed = 0.3 }: { start: number[], end: number[], color: string, delay?: number, speed?: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);
  const [progress, setProgress] = useState(-delay);

  const startVec = useMemo(() => new THREE.Vector3(...start), [start]);
  const endVec = useMemo(() => new THREE.Vector3(...end), [end]);
  const currentPos = useMemo(() => new THREE.Vector3(), []);

  useFrame((state, delta) => {
    if (!meshRef.current || !materialRef.current) return;

    let newProgress = progress + delta * speed;
    if (newProgress > 1) {
      newProgress = 0; 
    }
    setProgress(newProgress);

    if (newProgress > 0) {
      currentPos.lerpVectors(startVec, endVec, newProgress);
      meshRef.current.position.copy(currentPos);
      
      // Removed the * 0.8 multiplier so it hits a full 1.0 opacity at the peak of the arc
      materialRef.current.opacity = Math.sin(newProgress * Math.PI); 
    } else {
      materialRef.current.opacity = 0;
    }
  });

  return (
    <mesh ref={meshRef} position={new THREE.Vector3(...start)}>
      {/* Slightly larger bead radius: 0.12 instead of 0.08 */}
      <sphereGeometry args={[0.12, 8, 8]} /> 
      <meshStandardMaterial 
        ref={materialRef}
        color={color} 
        emissive={color} 
        emissiveIntensity={4} // Doubled intensity for a brighter glow
        transparent 
        depthWrite={false}
      />
    </mesh>
  );
}

// --- EDGE COMPONENT ---
function EdgeConnection({ node }: { node: ClaimNode }) {
  const endPoint = [0, 0, 0]; 
  const modelColor = MODEL_COLORS[node.model];
  const speed = useMemo(() => 0.15 + Math.random() * 0.1, []);

  return (
    <group>
      {/* Brighter, slightly thicker background wire */}
      <Line 
        points={[node.position, endPoint]}       
        color="#588983" // Lighter sub-field color
        lineWidth={1.5} 
        transparent
        opacity={0.2} // Doubled opacity so you can clearly see the connections
      />
      
      <FlowingBead start={node.position} end={endPoint} color={modelColor} speed={speed} delay={0} />
      <FlowingBead start={node.position} end={endPoint} color={modelColor} speed={speed} delay={0.5} />
    </group>
  );
}

// --- MAIN GRAPH COMPONENT ---
function RotatingGroup({ children }: { children: React.ReactNode }) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.001;
      groupRef.current.rotation.x += 0.0005;
    }
  });
  return <group ref={groupRef}>{children}</group>;
}

export default function Constellation({ selectedModels }: ConstellationProps) {
  const mockNodes = useMemo(() => generateMockData(), []);
  const visibleNodes = mockNodes.filter(node => selectedModels.includes(node.model));

  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [0, 0, 35], fov: 60 }} gl={{ alpha: true, antialias: true }}>
        <ambientLight intensity={1.2} color="#EBF0FF" />
        <pointLight position={[10, 10, 10]} intensity={2} color="#A9BDE8" />
        <pointLight position={[-10, -10, -10]} intensity={1} color="#588983" />
        
        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          dampingFactor={0.05}
        />

        <RotatingGroup>
          {visibleNodes.map((node) => (
            <group key={node.id}>
              <EdgeConnection node={node} />
              
              <Sphere position={node.position} args={[0.3, 16, 16]}>
                <meshStandardMaterial 
                  color={TRUST_COLORS[node.trustStatus]} 
                  emissive={TRUST_COLORS[node.trustStatus]}
                  emissiveIntensity={0.8}
                  roughness={0.2}
                  metalness={0.5}
                />
              </Sphere>
            </group>
          ))}
          
          <Sphere position={[0,0,0]} args={[1.2, 32, 32]}>
             <meshStandardMaterial 
                color="#EBF0FF" 
                emissive="#EBF0FF" 
                emissiveIntensity={0.3} 
                transparent 
                opacity={0.15} 
             />
          </Sphere>
        </RotatingGroup>
      </Canvas>
    </div>
  );
}