import { createHash } from 'node:crypto';
import { SuggestionError } from './tag-suggestion.js';

export function parseClaimId(value) {
    if (typeof value !== 'string' || !/^[1-9]\d*$/.test(value)) return null;
    const id = Number(value);
    return Number.isSafeInteger(id) && id <= 2_147_483_647 ? id : null;
}

export function isSameOrigin(request) {
    const origin = request.headers.get('origin');
    // Next may normalise request.url to localhost even when the browser uses
    // 127.0.0.1. The original Host header retains the browser-facing authority.
    const expected = new URL(request.url);
    expected.host = request.headers.get('host') || expected.host;
    return request.headers.get('sec-fetch-site') !== 'cross-site' &&
        (!origin || origin === expected.origin);
}

const json = (body, status = 200) => Response.json(body, { status, headers: { 'Cache-Control': 'no-store' } });

export function createSuggestionHandler({ getClaim, classify }) {
    const pending = new Map();
    return async function handleSuggestion(request, { params }) {
        if (!isSameOrigin(request)) return json({ error: 'This request is not allowed.', code: 'INVALID_ORIGIN' }, 403);
        const id = parseClaimId((await params).id);
        if (id === null) return json({ error: 'Invalid claim ID.', code: 'INVALID_ID' }, 400);
        let claim;
        try {
            claim = await getClaim(id);
        } catch {
            return json({ error: 'The claim could not be loaded. Please try again.', code: 'DATABASE_UNAVAILABLE' }, 503);
        }
        if (!claim) return json({ error: 'This claim no longer exists.', code: 'CLAIM_NOT_FOUND' }, 404);
        const key = `${id}:${createHash('sha256').update(claim.content).digest('hex')}`;
        if (!pending.has(key) && pending.size >= 2) {
            return json({ error: 'The tagging service is busy. Please try again shortly.', code: 'LLM_RATE_LIMITED' }, 429);
        }
        let result = pending.get(key);
        if (!result) {
            result = Promise.resolve().then(() => classify(claim.content, { signal: request.signal }));
            pending.set(key, result);
        }
        try {
            // Suggest only: persistence is a separate, explicit confirmation request.
            return json(await result);
        } catch (error) {
            if (error instanceof SuggestionError) return json({ error: error.message, code: error.code }, error.status);
            return json({ error: 'Automatic tagging failed. You can assign a tag manually.', code: 'TAGGING_FAILED' }, 502);
        } finally {
            if (pending.get(key) === result) pending.delete(key);
        }
    };
}
