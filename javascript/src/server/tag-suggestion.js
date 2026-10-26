import { ALLOWED_TAGS } from '../constants/tags.js';

export class SuggestionError extends Error {
    constructor(code, message, status) {
        super(message);
        this.name = 'SuggestionError';
        this.code = code;
        this.status = status;
    }
}

export function providerError(error) {
    if (error instanceof SuggestionError) return error;
    if (error?.name === 'APIConnectionTimeoutError' || error?.name === 'TimeoutError') {
        return new SuggestionError('LLM_TIMEOUT', 'The tagging service took too long. Please try again.', 504);
    }
    if (error?.status === 429) {
        return new SuggestionError('LLM_RATE_LIMITED', 'The tagging service is busy. Please try again shortly.', 429);
    }
    if ([401, 402, 403, 404].includes(error?.status)) {
        return new SuggestionError('LLM_UNAVAILABLE', 'Automatic tagging is currently unavailable. You can assign a tag manually.', 503);
    }
    return new SuggestionError('LLM_PROVIDER_ERROR', 'The tagging service could not answer. Please try again or assign a tag manually.', 502);
}

export async function classifyComplaint(text, { client, model, config, signal }) {
    if (typeof text !== 'string' || !text.trim()) {
        throw new SuggestionError('EMPTY_CLAIM', 'This claim has no text to classify.', 422);
    }
    if (text.length > 100_000) {
        throw new SuggestionError('CLAIM_TOO_LONG', 'This claim is too long for automatic tagging. Please assign a tag manually.', 413);
    }
    const selected = config.models[model];
    if (!selected) {
        throw new SuggestionError('LLM_CONFIGURATION', 'Automatic tagging is currently unavailable. You can assign a tag manually.', 503);
    }
    const started = performance.now();
    const body = {
        model,
        messages: [{ role: 'system', content: selected.systemPrompt }, { role: 'user', content: text }],
        max_tokens: config.maxOutputTokens,
        reasoning: selected.reasoning,
        provider: { require_parameters: true, max_price: selected.maxPrice },
        usage: { include: true },
        response_format: {
            type: 'json_schema',
            json_schema: {
                name: 'complaint_label', strict: true,
                schema: {
                    type: 'object', properties: { label: { type: 'string', enum: ALLOWED_TAGS } },
                    required: ['label'], additionalProperties: false,
                },
            },
        },
    };
    if (selected.temperature !== null) body.temperature = selected.temperature;
    try {
        const response = await client.chat.completions.create(body, { signal });
        if (response?.error) throw providerError({ status: Number(response.error.code) });
        const choice = response?.choices?.[0];
        if (choice?.error || choice?.finish_reason === 'error') throw providerError({});
        let parsed;
        try {
            parsed = JSON.parse(choice?.message?.content);
        } catch {
            // A response with fences or explanations is not the required JSON contract.
        }
        if (!parsed || Array.isArray(parsed) || Object.keys(parsed).length !== 1 ||
            !ALLOWED_TAGS.includes(parsed.label) || choice?.finish_reason !== 'stop') {
            throw new SuggestionError('LLM_INVALID_RESPONSE', 'The tagging service returned an invalid suggestion. No tag was changed.', 502);
        }
        const cost = response.usage?.cost;
        return {
            tag: parsed.label, model, elapsedMs: Math.round(performance.now() - started),
            costUsd: Number.isFinite(cost) && cost >= 0 ? cost : null,
        };
    } catch (error) {
        throw providerError(error);
    }
}
