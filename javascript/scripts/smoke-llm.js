import { mkdir, writeFile } from 'node:fs/promises';

const base = 'http://127.0.0.1:3100';
const claims = await (await fetch(`${base}/api/claims`)).json();
const claim = claims.find(item => item.id === 1);
if (!claim) throw new Error('Starter claim #1 is required for the live check.');
const response = await fetch(`${base}/api/claims/1/suggest-tag`, { method: 'POST', headers: { Origin: base } });
const result = await response.json();
const after = (await (await fetch(`${base}/api/claims`)).json()).find(item => item.id === 1);
const report = { checkedAt: new Date().toISOString(), status: response.status, result, tagUnchanged: after.tag === claim.tag };
await mkdir('../../artifacts/mission2', { recursive: true });
await writeFile('../../artifacts/mission2/smoke-live.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
if (!response.ok || !report.tagUnchanged) process.exitCode = 1;
