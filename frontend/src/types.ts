export interface AnalysisResult {
  schema_version: string;
  analysis_id: string;
  prompt: string;
  models: { model_id: string; response_text: string }[];
  claims: { claim_id: string; model_id: string; claim_text: string; span: { start: number; end: number } | null }[];
  clusters: { cluster_id: string; claim_ids: string[]; representative_claim_id: string; representative_text: string }[];
  coords3d: Record<string, [number, number, number]>;
  nli_results: { pair_id: string; claim_id: string; passage_id: string; label: string; probs: { entailment: number; contradiction: number; neutral: number } }[];
  cluster_scores: { cluster_id: string; trust_score: number; verdict: 'SAFE' | 'CAUTION' | 'REJECT'; agreement: { models_supporting: string[]; count: number }; verification: { best_entailment_prob: number; best_contradiction_prob: number; evidence_passage_id: string } }[];
  safe_answer: { text: string; supported_cluster_ids: string[]; rejected_cluster_ids: string[] };
  model_metrics: { model_id: string; claim_counts: { total: number; supported: number; caution: number; rejected: number } }[];
  warnings: string[];
  status: string;
}
