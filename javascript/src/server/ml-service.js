import 'server-only';
import { classifyWithML } from './ml-client.js';

export function suggestTag(text, { signal } = {}) {
    return classifyWithML(text, { signal, baseURL: process.env.ML_API_URL || 'http://127.0.0.1:8000' });
}
