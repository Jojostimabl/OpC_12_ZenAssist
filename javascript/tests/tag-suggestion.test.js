import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import config from '../src/server/llm-config.json' with { type: 'json' };
import { classifyComplaint } from '../src/server/tag-suggestion.js';
import { createSuggestionHandler } from '../src/server/suggestion-handler.js';
import { ALLOWED_TAGS } from '../src/constants/tags.js';

const model = config.defaultModel;
const valid = { choices: [{ finish_reason: 'stop', message: { content: '{"label":"Mortgage"}' } }], usage: { cost: 0.0001 } };
const fakeClient = create => ({ chat: { completions: { create } } });

test('the mission 1 prompts are intact and preserve 18 categories', () => {
    assert.equal(ALLOWED_TAGS.length, 18);
    assert.equal(new Set(ALLOWED_TAGS).size, 18);
    for (const item of Object.values(config.models)) {
        assert.equal(createHash('sha256').update(item.systemPrompt).digest('hex'), item.promptSha256);
        for (const label of ALLOWED_TAGS) assert.ok(item.systemPrompt.includes(label));
    }
});

test('a valid response preserves full input and the frozen request contract', async () => {
    let calls = 0;
    const text = 'My mortgage servicer has not applied my payment.';
    const client = fakeClient(async body => {
        calls++;
        assert.equal(body.model, model);
        assert.equal(body.messages[0].content, config.models[model].systemPrompt);
        assert.equal(body.messages[1].content, text);
        assert.equal(body.temperature, 0);
        assert.equal(body.max_tokens, 256);
        assert.deepEqual(body.reasoning, { enabled: false });
        assert.deepEqual(body.response_format.json_schema.schema.properties.label.enum, ALLOWED_TAGS);
        return valid;
    });
    const result = await classifyComplaint(text, { client, model, config });
    assert.equal(result.tag, 'Mortgage');
    assert.equal(result.costUsd, 0.0001);
    assert.ok(result.elapsedMs >= 0);
    assert.equal(calls, 1);
});

test('unsupported temperature is omitted for the Luna configuration', async () => {
    await classifyComplaint('A complaint', { model: 'openai/gpt-6-luna', config,
        client: fakeClient(async body => { assert.ok(!('temperature' in body)); return valid; }) });
});

for (const [name, response] of [
    ['invalid JSON', { choices: [{ finish_reason: 'stop', message: { content: 'not JSON' } }] }],
    ['unknown category', { choices: [{ finish_reason: 'stop', message: { content: '{"label":"invented"}' } }] }],
    ['additional properties', { choices: [{ finish_reason: 'stop', message: { content: '{"label":"Mortgage","reason":"x"}' } }] }],
    ['truncated response', { choices: [{ finish_reason: 'length', message: { content: '{"label":"Mortgage"}' } }] }],
    ['empty choices', { choices: [] }],
]) {
    test(`${name} never yields an applicable tag`, async () => {
        await assert.rejects(classifyComplaint('A complaint', { client: fakeClient(async () => response), model, config }),
            error => error.code === 'LLM_INVALID_RESPONSE' && error.status === 502);
    });
}

for (const [upstream, code, status] of [
    [{ status: 429 }, 'LLM_RATE_LIMITED', 429],
    [{ status: 401 }, 'LLM_UNAVAILABLE', 503],
    [{ status: 402 }, 'LLM_UNAVAILABLE', 503],
    [{ status: 503 }, 'LLM_PROVIDER_ERROR', 502],
    [{ name: 'APIConnectionTimeoutError' }, 'LLM_TIMEOUT', 504],
    [new Error('Transport failure with private detail'), 'LLM_PROVIDER_ERROR', 502],
]) {
    test(`provider error ${upstream.status ?? upstream.name} is handled without leaking raw details`, async () => {
        let calls = 0;
        await assert.rejects(classifyComplaint('A complaint', { model, config,
            client: fakeClient(async () => { calls++; throw upstream; }) }),
            error => error.code === code && error.status === status && !error.message.includes('private detail'));
        assert.equal(calls, 1);
    });
}

test('an error returned inside HTTP 200 is not interpreted as a tag', async () => {
    await assert.rejects(classifyComplaint('A complaint', { model, config,
        client: fakeClient(async () => ({ error: { code: 429 }, choices: [] })) }),
        error => error.code === 'LLM_RATE_LIMITED');
});

test('blank and oversized claims do not call the provider', async () => {
    const client = fakeClient(() => { assert.fail('No provider request expected'); });
    await assert.rejects(classifyComplaint('  ', { client, model, config }), error => error.status === 422);
    await assert.rejects(classifyComplaint('x'.repeat(100_001), { client, model, config }), error => error.status === 413);
});

const request = (id, origin) => new Request(`http://localhost:3100/api/claims/${id}/suggest-tag`, {
    method: 'POST', headers: origin ? { Origin: origin } : {},
});
const context = id => ({ params: Promise.resolve({ id }) });

test('Next normalised localhost URL accepts the matching browser-facing Host only', async () => {
    const handler = createSuggestionHandler({ getClaim: async () => ({ content: 'A complaint' }), classify: async () => ({ tag: 'Mortgage' }) });
    const local = new Request('http://localhost:3100/api/claims/1/suggest-tag', {
        method: 'POST', headers: { Host: '127.0.0.1:3100', Origin: 'http://127.0.0.1:3100' },
    });
    assert.equal((await handler(local, context('1'))).status, 200);
    const foreign = new Request(local.url, { method: 'POST', headers: { Host: '127.0.0.1:3100', Origin: 'https://unrelated.example' } });
    assert.equal((await handler(foreign, context('1'))).status, 403);
});

test('the handler reads the server record and only returns a suggestion', async () => {
    const original = { id: 1, content: 'Server-side text', tag: null };
    const handler = createSuggestionHandler({ getClaim: async id => { assert.equal(id, 1); return original; },
        classify: async text => { assert.equal(text, original.content); return { tag: 'Mortgage' }; } });
    const result = await handler(request('1'), context('1'));
    assert.equal(result.status, 200);
    assert.deepEqual(await result.json(), { tag: 'Mortgage' });
    assert.equal(original.tag, null);
});

test('invalid identifiers, missing records and cross-origin calls do not reach the LLM', async () => {
    const handler = createSuggestionHandler({ getClaim: async () => null, classify: () => assert.fail('No call expected') });
    assert.equal((await handler(request('1bad'), context('1bad'))).status, 400);
    assert.equal((await handler(request('2147483648'), context('2147483648'))).status, 400);
    assert.equal((await handler(request('1'), context('1'))).status, 404);
    assert.equal((await handler(request('1', 'https://unrelated.example'), context('1'))).status, 403);
});

test('concurrent requests for the same record share a single paid request', async () => {
    let calls = 0;
    let release;
    const answer = new Promise(resolve => { release = resolve; });
    const handler = createSuggestionHandler({ getClaim: async () => ({ content: 'A complaint' }),
        classify: async () => { calls++; return answer; } });
    const first = handler(request('1'), context('1'));
    const second = handler(request('1'), context('1'));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls, 1);
    release({ tag: 'Mortgage' });
    assert.equal((await first).status, 200);
    assert.equal((await second).status, 200);
});
