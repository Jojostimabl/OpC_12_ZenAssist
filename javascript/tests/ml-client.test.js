import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyWithML } from '../src/server/ml-client.js';
import { SuggestionError } from '../src/server/suggestion-error.js';
import { createSuggestionHandler } from '../src/server/suggestion-handler.js';

const valid = { tag: 'Mortgage', model: 'logistic_regression', modelVersion: 'test-version', elapsedMs: 1.2 };
const fake = data => async () => Response.json(data);

test('the internal request sends the full server text and returns quality metadata', async () => {
    const text = 'Full complaint '.repeat(300);
    const result = await classifyWithML(text, { fetchImpl: async (url, options) => {
        assert.equal(url.href, 'http://127.0.0.1:8000/tags');
        assert.equal(options.method, 'POST');
        assert.deepEqual(JSON.parse(options.body), { user_claim: text });
        assert.equal(options.redirect, 'error');
        return Response.json(valid);
    } });
    assert.equal(result.tag, 'Mortgage');
    assert.equal(result.inferenceMs, 1.2);
    assert.ok(result.elapsedMs >= 0);
});

for (const [name, data] of [
    ['unknown label', { ...valid, tag: 'invented' }], ['missing version', { ...valid, modelVersion: null }],
    ['wrong model', { ...valid, model: 'unknown' }], ['negative timing', { ...valid, elapsedMs: -1 }],
    ['missing timing', { ...valid, elapsedMs: undefined }], ['null', null],
]) {
    test(`${name} is not an applicable suggestion`, async () => {
        await assert.rejects(classifyWithML('Complaint', { fetchImpl: fake(data) }), error => error.code === 'ML_INVALID_RESPONSE');
    });
}

test('non-JSON success is rejected', async () => {
    await assert.rejects(classifyWithML('Complaint', { fetchImpl: async () => new Response('not-json') }), error => error.code === 'ML_INVALID_RESPONSE');
});

for (const [upstream, status] of [[422, 422], [429, 503], [503, 503], [500, 502], [404, 502]]) {
    test(`HTTP ${upstream} becomes a safe actionable error`, async () => {
        await assert.rejects(classifyWithML('Complaint', { fetchImpl: async () => new Response('private details', { status: upstream }) }),
            error => error.status === status && !error.message.includes('private details'));
    });
}

test('connection refused does not leak internal errors', async () => {
    await assert.rejects(classifyWithML('Complaint', { fetchImpl: async () => { throw new Error('private details'); } }), error => error.status === 503 && !error.message.includes('private details'));
});
test('timeout is distinguished from service unavailability', async () => {
    await assert.rejects(classifyWithML('Complaint', { fetchImpl: async () => { throw new DOMException('Timeout', 'TimeoutError'); } }), error => error.status === 504);
});
test('invalid input and configuration never reach the service', async () => {
    const fetchImpl = () => assert.fail('No request expected');
    await assert.rejects(classifyWithML(' ', { fetchImpl }), error => error.status === 422);
    await assert.rejects(classifyWithML('x'.repeat(100_001), { fetchImpl }), error => error.status === 413);
    await assert.rejects(classifyWithML('Complaint', { fetchImpl, baseURL: 'invalid' }), error => error.status === 503);
});

const request = (id, headers = {}) => new Request(`http://localhost:3100/api/claims/${id}/suggest-tag`, { method: 'POST', headers });
const context = id => ({ params: Promise.resolve({ id }) });
test('the route reads authoritative server text and does not save a tag', async () => {
    const original = { content: 'Server-side text', tag: null };
    const handler = createSuggestionHandler({ getClaim: async id => { assert.equal(id, 1); return original; },
        classify: async text => { assert.equal(text, original.content); return valid; } });
    assert.deepEqual(await (await handler(request('1'), context('1'))).json(), valid);
    assert.equal(original.tag, null);
});
test('bad identifiers, cross-origin and absent records do not reach ML', async () => {
    const handler = createSuggestionHandler({ getClaim: async () => null, classify: () => assert.fail('No prediction expected') });
    for (const id of ['bad', '1bad', '0', '2147483648']) assert.equal((await handler(request(id), context(id))).status, 400);
    assert.equal((await handler(request('1'), context('1'))).status, 404);
    assert.equal((await handler(request('1', { Origin: 'https://foreign.example' }), context('1'))).status, 403);
    assert.equal((await handler(request('1', { 'sec-fetch-site': 'cross-site' }), context('1'))).status, 403);
});
test('Next normalised URL uses the browser-facing Host for origin validation', async () => {
    const handler = createSuggestionHandler({ getClaim: async () => ({ content: 'Complaint' }), classify: async () => valid });
    assert.equal((await handler(request('1', { Host: '127.0.0.1:3100', Origin: 'http://127.0.0.1:3100' }), context('1'))).status, 200);
});
test('database and ML errors retain distinct statuses', async () => {
    const dbFail = createSuggestionHandler({ getClaim: async () => { throw new Error('private'); }, classify: () => assert.fail() });
    assert.equal((await dbFail(request('1'), context('1'))).status, 503);
    const mlFail = createSuggestionHandler({ getClaim: async () => ({ content: 'Complaint' }), classify: async () => { throw new SuggestionError('ML_TIMEOUT', 'Timed out', 504); } });
    assert.equal((await mlFail(request('1'), context('1'))).status, 504);
});
test('concurrent requests are deduplicated and additional work is bounded', async () => {
    let calls = 0;
    let release;
    const pending = new Promise(resolve => { release = resolve; });
    const handler = createSuggestionHandler({ getClaim: async () => ({ content: 'Complaint' }), classify: async () => { calls++; return pending; } });
    const first = handler(request('1'), context('1'));
    const duplicate = handler(request('1'), context('1'));
    const other = handler(request('2'), context('2'));
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(calls, 2);
    assert.equal((await handler(request('3'), context('3'))).status, 429);
    release(valid);
    assert.deepEqual(await Promise.all([first, duplicate, other].map(async result => (await result).status)), [200, 200, 200]);
});
