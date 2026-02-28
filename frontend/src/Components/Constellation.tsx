import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, useCursor, Text, Float } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import { AnalysisResult } from '../types';

// --- STYLING CONSTANTS ---
const TRUST_COLORS: Record<string, string> = {
  VerifiedSafe: '#00D68F',
  CautionUnverified: '#FFB020',
  Rejected: '#FF4757',
};

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
  "Quantum entanglement allows for information transfer faster than light.", 
  "The Paris Agreement aims to limit global warming to 1.5°C.",
  "Python is the most utilized programming language for machine learning."
];

interface Cluster {
  id: number;
  position: [number, number, number];
  text: string;
  isOutlier: boolean;
}

interface ClaimNode {
  id: string;
  clusterId: number;
  model: string;
  position: [number, number, number];
  trustStatus: 'VerifiedSafe' | 'CautionUnverified' | 'Rejected';
  text: string;
  confidence: number;
}

// --- PROPORTIONATE MOCK DATA ---
const generateGraphData = () => {
  const nodes: ClaimNode[] = [];
  const clusters: Cluster[] = [];
  
  for (let c = 0; c < 5; c++) {
    const clusterPos: [number, number, number] = [
      (Math.random() - 0.5) * 45, 
      (Math.random() - 0.5) * 30, 
      (Math.random() - 0.5) * 45, 
    ];
    
    clusters.push({ id: c, position: clusterPos, text: MOCK_TEXTS[c], isOutlier: c === 2 });
    const baseConfidence = c === 2 ? 15 : 80 + Math.random() * 15;
    
    MODELS.forEach((model, i) => {
      if (Math.random() > 0.2) {
        nodes.push({
          id: `node-${c}-${i}`,
          clusterId: c,
          model: model,
          position: [
            clusterPos[0] + (Math.random() - 0.5) * 10,
            clusterPos[1] + (Math.random() - 0.5) * 10,
            clusterPos[2] + (Math.random() - 0.5) * 10,
          ],
          trustStatus: c === 2 ? 'Rejected' : (Math.random() > 0.8 ? 'CautionUnverified' : 'VerifiedSafe'),
          text: MOCK_TEXTS[c],
          confidence: Math.floor(baseConfidence + (Math.random() - 0.5) * 10),
        });
      }
    });
  }

  for (let i = 0; i < 5; i++) {
    const outPos: [number, number, number] = [(Math.random() - 0.5) * 60, (Math.random() - 0.5) * 60, (Math.random() - 0.5) * 60];
    clusters.push({ id: 999 + i, position: outPos, text: "Independent Claim", isOutlier: true });
    
    nodes.push({
      id: `outlier-${i}`,
      clusterId: 999 + i,
      model: MODELS[Math.floor(Math.random() * MODELS.length)],
      position: [outPos[0] + 1.5, outPos[1] + 1.5, outPos[2] + 1.5],
      trustStatus: Math.random() > 0.5 ? 'Rejected' : 'CautionUnverified',
      text: "This is a hallucinated or completely independent claim made by a single model.",
      confidence: Math.floor(Math.random() * 40),
    });
  }
  return { nodes, clusters };
};

// --- ANIMATED BEAD COMPONENT ---
function FlowingBead({ start, end, color, delay = 0, speed = 0.25, isDimmed }: any) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial>(null);
  const [progress, setProgress] = useState(-delay);

  const startVec = useMemo(() => new THREE.Vector3(...start), [start]);
  const endVec = useMemo(() => new THREE.Vector3(...end), [end]);
  const currentPos = useMemo(() => new THREE.Vector3(), []);

  useFrame((_state, delta) => {
    if (!meshRef.current || !materialRef.current || isDimmed) {
      if (materialRef.current) materialRef.current.opacity = 0;
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
    <mesh ref={meshRef} position={startVec}>
      <sphereGeometry args={[0.35, 8, 8]} /> 
      <meshStandardMaterial ref={materialRef} color={color} emissive={color} emissiveIntensity={5} transparent depthWrite={false} />
    </mesh>
  );
}

// --- CLUSTER HUB ---
function ClusterHub({ cluster, isDimmed }: { cluster: Cluster, isDimmed: boolean }) {
  const isRejected = cluster.isOutlier;
  return (
    <group>
      <Line 
        points={[cluster.position, [0, 0, 0]]}       
        color={isRejected ? TRUST_COLORS.Rejected : "#588983"} 
        lineWidth={isRejected ? 1.5 : 2.5} 
        transparent
        opacity={isDimmed ? 0.05 : (isRejected ? 0.15 : 0.4)} 
      />
      <Sphere position={cluster.position} args={[0.8, 16, 16]}>
        <meshBasicMaterial color="#FFFFFF" transparent opacity={isDimmed ? 0.05 : 0.9} />
      </Sphere>
    </group>
  );
}

// --- INTERACTIVE NODE ---
function InteractiveNode({ node, clusterPos, isSelected, isDimmed, onClick }: any) {
  const [hovered, setHovered] = useState(false);
  const auraRef = useRef<THREE.Mesh>(null);
  useCursor(hovered);

  // Colors split by Model and Trust
  const nodeColor = MODEL_COLORS[node.model];
  const trustColor = TRUST_COLORS[node.trustStatus];
  
  const size = 0.9 + (node.confidence / 100) * 0.6; 
  const opacity = isDimmed ? 0.1 : 1;

  // Speed of beads based on confidence (from architecture spec)
  const beadSpeed = 0.1 + (node.confidence / 100) * 0.3;

  // Animate the soft Aura sphere instead of a hard ring
  useFrame((state) => {
    if (auraRef.current && !isDimmed) {
      const time = state.clock.elapsedTime;
      const mat = auraRef.current.material as THREE.MeshStandardMaterial;

      if (node.trustStatus === 'Rejected') {
        // Fast, erratic flashing and pulsing
        const pulse = 1 + Math.sin(time * 8) * 0.15;
        auraRef.current.scale.setScalar(pulse);
        mat.opacity = 0.3 + Math.sin(time * 8) * 0.3;
      } 
      else if (node.trustStatus === 'CautionUnverified') {
        // Slow, nervous breathing
        const pulse = 1 + Math.sin(time * 3) * 0.08;
        auraRef.current.scale.setScalar(pulse);
        mat.opacity = 0.2 + Math.sin(time * 3) * 0.15;
      } 
      else {
        // Verified: Steady, calm, large glow
        auraRef.current.scale.setScalar(1.05);
        mat.opacity = 0.15 + Math.sin(time * 1.5) * 0.05;
      }
    }
  });

  return (
    <group>
      {/* Edge to Cluster Hub -> Colored by TRUST STATUS */}
      <Line 
        points={[node.position, clusterPos]}       
        color={trustColor} 
        lineWidth={isSelected ? 3.5 : 2} 
        transparent
        opacity={isDimmed ? 0.02 : 0.4} 
      />
      
      {/* Flowing Beads -> Colored by MODEL IDENTITY */}
      <FlowingBead start={node.position} end={clusterPos} color={nodeColor} speed={beadSpeed} delay={0} isDimmed={isDimmed} />

      {/* Solid Core Node -> Colored by MODEL IDENTITY */}
      <Sphere 
        position={node.position} 
        args={[size, 32, 32]}
        scale={isSelected ? 1.3 : hovered ? 1.15 : 1}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial color={nodeColor} emissive={nodeColor} emissiveIntensity={hovered || isSelected ? 1 : 0.7} transparent opacity={opacity} />
      </Sphere>

      {/* Soft Glowing Aura -> Colored by TRUST STATUS */}
      <Sphere ref={auraRef} position={node.position} args={[size * 1.6, 32, 32]}>
        <meshStandardMaterial color={trustColor} emissive={trustColor} emissiveIntensity={2} transparent opacity={isDimmed ? 0 : 0.2} depthWrite={false} />
      </Sphere>

      {/* 3D Text Label */}
      {!isDimmed && (
        <Text 
          position={[node.position[0], node.position[1] + size + 1.2, node.position[2]]}
          fontSize={1.0}
          color="#EBF0FF"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.08}
          outlineColor="#0A0E1A"
        >
          {node.model.split(' ')[0]}
        </Text>
      )}
    </group>
  );
}

// --- MAIN WRAPPER ---
function RotatingGroup({ children, isPaused }: { children: React.ReactNode, isPaused: boolean }) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame(() => {
    if (groupRef.current && !isPaused) {
      groupRef.current.rotation.y += 0.0005;
    }
  });
  return <group ref={groupRef}>{children}</group>;
}

interface ConstellationProps {
  selectedModels: string[];
  result: AnalysisResult | null;
}

export default function Constellation({ selectedModels, result }: ConstellationProps) {
  // Derive real cluster hub and claim node positions from result
  const realNodes = useMemo(() => {
    if (!result) return null;
    const { clusters, claims, coords3d, cluster_scores } = result;

    const clusterHubs = clusters.map(cluster => {
      const memberCoords = cluster.claim_ids
        .map(id => coords3d[id])
        .filter(Boolean) as [number, number, number][];
      const centroid: [number, number, number] = memberCoords.length > 0
        ? [
            memberCoords.reduce((a, c) => a + c[0], 0) / memberCoords.length,
            memberCoords.reduce((a, c) => a + c[1], 0) / memberCoords.length,
            memberCoords.reduce((a, c) => a + c[2], 0) / memberCoords.length,
          ]
        : [0, 0, 0];
      const score = cluster_scores.find(s => s.cluster_id === cluster.cluster_id);
      const verdict = score?.verdict ?? 'SAFE';
      const trustStatus: 'VerifiedSafe' | 'CautionUnverified' | 'Rejected' =
        verdict === 'SAFE' ? 'VerifiedSafe' :
        verdict === 'CAUTION' ? 'CautionUnverified' : 'Rejected';
      return {
        id: cluster.cluster_id,
        position: centroid,
        text: cluster.representative_text,
        isOutlier: cluster.claim_ids.length === 1,
        trustStatus,
      };
    });

    const claimNodes = claims.map(claim => {
      const pos = coords3d[claim.claim_id] ?? [0, 0, 0] as [number, number, number];
      const clusterForClaim = clusters.find(c => c.claim_ids.includes(claim.claim_id));
      const score = clusterForClaim
        ? cluster_scores.find(s => s.cluster_id === clusterForClaim.cluster_id)
        : undefined;
      const verdict = score?.verdict ?? 'SAFE';
      const trustStatus: 'VerifiedSafe' | 'CautionUnverified' | 'Rejected' =
        verdict === 'SAFE' ? 'VerifiedSafe' :
        verdict === 'CAUTION' ? 'CautionUnverified' : 'Rejected';
      return {
        id: claim.claim_id,
        position: pos,
        model: claim.model_id,
        trustStatus,
        text: claim.claim_text,
        clusterId: clusterForClaim?.cluster_id ?? null,
      };
    });

    return { clusterHubs, claimNodes };
  }, [result]);

  const { nodes, clusters } = useMemo(() => generateGraphData(), []);
  const [activeNode, setActiveNode] = useState<ClaimNode | null>(null);

  const visibleNodes = nodes.filter(node => selectedModels.includes(node.model));
  const activeClusterId = activeNode?.clusterId;

  return (
    <div className="relative w-full h-full">
      <Canvas camera={{ position: [0, 15, 60], fov: 55 }} gl={{ alpha: true, antialias: true }} onPointerMissed={() => setActiveNode(null)}>
        <ambientLight intensity={1.5} color="#EBF0FF" />
        <pointLight position={[20, 20, 20]} intensity={2} color="#A9BDE8" />
        
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} dampingFactor={0.05} />

        <RotatingGroup isPaused={activeNode !== null}>
          
          {clusters.map(cluster => (
            <ClusterHub 
              key={`hub-${cluster.id}`} 
              cluster={cluster} 
              isDimmed={activeClusterId !== undefined && activeClusterId !== cluster.id} 
            />
          ))}

          {visibleNodes.map((node) => {
            const isSelected = activeNode?.id === node.id;
            const isDimmed = activeClusterId !== undefined && activeClusterId !== node.clusterId;
            const myCluster = clusters.find(c => c.id === node.clusterId)!;

            return (
              <InteractiveNode 
                key={node.id}
                node={node}
                clusterPos={myCluster.position}
                isSelected={isSelected}
                isDimmed={isDimmed}
                onClick={() => setActiveNode(node)}
              />
            );
          })}
          
          <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
            <Sphere position={[0,0,0]} args={[4.5, 32, 32]}>
              <meshStandardMaterial color="#EBF0FF" emissive="#A9BDE8" emissiveIntensity={activeNode ? 0.1 : 0.8} transparent opacity={activeNode ? 0.05 : 0.8} />
            </Sphere>
          </Float>
        </RotatingGroup>
      </Canvas>

      {/* 2D HTML OVERLAY PANEL */}
      <AnimatePresence>
        {activeNode && (
          <motion.div
            initial={{ opacity: 0, x: 50 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 50 }} transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="absolute top-8 right-8 w-80 flex flex-col gap-4 z-50 pointer-events-auto"
            onPointerDown={(e) => e.stopPropagation()} 
          >
            {/* CLAIM CARD */}
            <div className="bg-[#121825] border border-[#2C3A50] rounded-xl overflow-hidden shadow-2xl">
              <div className="bg-[#1A2335] px-4 py-2 border-b border-[#2C3A50]">
                <span className="text-[#90A2B3] text-xs font-bold tracking-widest uppercase">Claim</span>
              </div>
              <div className="p-4">
                <p className="text-[#EBF0FF] text-[15px] leading-relaxed">"{activeNode.text}"</p>
                <div className="mt-4 flex items-center justify-between">
                  <span className="text-[#90A2B3] text-xs">Generated by:</span>
                  <span className="text-xs font-semibold px-2.5 py-1 rounded border" style={{ color: MODEL_COLORS[activeNode.model], backgroundColor: `${MODEL_COLORS[activeNode.model]}20`, borderColor: `${MODEL_COLORS[activeNode.model]}50` }}>
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
                <div className="w-full h-3 bg-[#1A2335] rounded-full overflow-hidden mb-3">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${activeNode.confidence}%` }} transition={{ duration: 0.8, ease: "easeOut" }} className="h-full" style={{ backgroundColor: TRUST_COLORS[activeNode.trustStatus] }} />
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