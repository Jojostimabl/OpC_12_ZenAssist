import { mkdir, writeFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';

const base = process.env.APP_URL || 'http://127.0.0.1:3100';
const modelURL = process.env.ML_API_URL || 'http://127.0.0.1:8000';
const claims = await (await fetch(`${base}/api/claims`)).json();
const claim = claims.find(item => item.id === 1);
if (!claim) throw new Error('Starter claim #1 is required for the integration check.');
const healthResponse = await fetch(`${modelURL}/health`);
if (!healthResponse.ok) throw new Error('The ML model is not ready.');
const health = await healthResponse.json();
const samples = [];
for (let i = 0; i < 10; i++) {
    const started = performance.now();
    const response = await fetch(`${base}/api/claims/1/suggest-tag`, { method: 'POST', headers: { Origin: base } });
    const result = await response.json();
    samples.push({ status: response.status, result, roundTripMs: Math.round(performance.now() - started) });
    if (!response.ok || result.modelVersion !== health.modelVersion) throw new Error('ML integration check failed.');
}
const direct = await (await fetch(`${modelURL}/tags`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_claim: claim.content }) })).json();
const after = (await (await fetch(`${base}/api/claims`)).json()).find(item => item.id === 1);
const timings = samples.map(sample => sample.roundTripMs).sort((a, b) => a - b);
const report = { checkedAt: new Date().toISOString(), appURL: base, health, samples, firstRoundTripMs: samples[0].roundTripMs,
    medianRoundTripMs: (timings[4] + timings[5]) / 2, p95RoundTripMs: timings[9],
    directPredictionMatches: samples.every(sample => sample.result.tag === direct.tag),
    tagUnchanged: after.tag === claim.tag, externalProviderCalls: 0 };
await mkdir('../../artifacts/mission3', { recursive: true });
const reportName = new URL(base).port === '3101' ? 'smoke-production.json' : 'smoke-live.json';
await writeFile(`../../artifacts/mission3/${reportName}`, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ model: health.model, version: health.modelVersion, tag: direct.tag,
    medianRoundTripMs: report.medianRoundTripMs, p95RoundTripMs: report.p95RoundTripMs,
    directPredictionMatches: report.directPredictionMatches, tagUnchanged: report.tagUnchanged }));
if (!report.directPredictionMatches || !report.tagUnchanged) process.exitCode = 1;
