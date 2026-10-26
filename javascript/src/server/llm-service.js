import 'server-only';
import OpenAI from 'openai';
import config from './llm-config.json' with { type: 'json' };
import { classifyComplaint, SuggestionError } from './tag-suggestion.js';

export function suggestTag(text, { signal } = {}) {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
        throw new SuggestionError('LLM_CONFIGURATION', 'Automatic tagging is currently unavailable. You can assign a tag manually.', 503);
    }
    const client = new OpenAI({
        apiKey, baseURL: 'https://openrouter.ai/api/v1', timeout: 45_000, maxRetries: 0,
    });
    return classifyComplaint(text, {
        client, config, signal, model: process.env.OPENROUTER_MODEL || config.defaultModel,
    });
}
