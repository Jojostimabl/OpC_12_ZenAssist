import { ALLOWED_TAGS } from '../constants/tags.js';
import { SuggestionError } from './suggestion-error.js';

const unavailable = () => new SuggestionError('ML_UNAVAILABLE', 'Automatic tagging is currently unavailable. Check that the ML service is running, or assign a tag manually.', 503);

export async function classifyWithML(text, { baseURL = 'http://127.0.0.1:8000', fetchImpl = fetch, signal } = {}) {
    if (typeof text !== 'string' || !text.trim()) throw new SuggestionError('EMPTY_CLAIM', 'This claim has no text to classify.', 422);
    if (text.length > 100_000) throw new SuggestionError('CLAIM_TOO_LONG', 'This claim is too long for automatic tagging. Please assign a tag manually.', 413);
    let endpoint;
    try {
        endpoint = new URL('/tags', baseURL);
        if (!['http:', 'https:'].includes(endpoint.protocol) || endpoint.username || endpoint.password) throw new Error('Invalid endpoint');
    } catch {
        throw unavailable();
    }
    const started = performance.now();
    try {
        const response = await fetchImpl(endpoint, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_claim: text }), cache: 'no-store', redirect: 'error',
            signal: AbortSignal.any([signal, AbortSignal.timeout(10_000)].filter(Boolean)),
        });
        if (!response.ok) {
            if (response.status === 422) throw new SuggestionError('INVALID_CLAIM', 'This claim does not contain usable text. Please assign a tag manually.', 422);
            if ([429, 503].includes(response.status)) throw unavailable();
            throw new SuggestionError('ML_SERVICE_ERROR', 'The model could not answer. Please try again or assign a tag manually.', 502);
        }
        let data;
        try { data = await response.json(); } catch { /* Reject non-JSON responses. */ }
        if (!data || !ALLOWED_TAGS.includes(data.tag) || data.model !== 'logistic_regression' ||
            typeof data.modelVersion !== 'string' || !data.modelVersion.trim() || data.modelVersion.length > 100 ||
            !Number.isFinite(data.elapsedMs) || data.elapsedMs < 0) {
            throw new SuggestionError('ML_INVALID_RESPONSE', 'The model returned an invalid suggestion. No tag was changed.', 502);
        }
        return { tag: data.tag, model: data.model, modelVersion: data.modelVersion,
            elapsedMs: Math.round(performance.now() - started), inferenceMs: data.elapsedMs };
    } catch (error) {
        if (error instanceof SuggestionError) throw error;
        if (error?.name === 'TimeoutError') throw new SuggestionError('ML_TIMEOUT', 'The model took too long. Please try again or assign a tag manually.', 504);
        throw unavailable();
    }
}
