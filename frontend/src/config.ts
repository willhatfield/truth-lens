export const API_BASE = import.meta.env.VITE_API_URL ?? 'https://truth-lens-production.up.railway.app';
export const WS_BASE  = API_BASE.replace(/^http/, 'ws');
