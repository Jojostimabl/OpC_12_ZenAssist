import { test, expect } from '@playwright/test';
import { Pool } from 'pg';

const pool = new Pool({ host: '127.0.0.1', port: 55432, database: 'zenassist_mission2', user: 'postgres', password: 'postgres' });
let claimId;
test.beforeAll(async () => {
    // Tests own only this synthetic record, leaving all existing claims untouched.
    const result = await pool.query('INSERT INTO claims(content, tag) VALUES ($1, NULL) RETURNING id', ['My bank charged an unexpected credit card fee. Please review it.']);
    claimId = result.rows[0].id;
});
test.beforeEach(async ({ page }) => {
    await pool.query('UPDATE claims SET tag = NULL WHERE id = $1', [claimId]);
    await page.goto('/');
    await page.getByRole('button', { name: `Select claim #${claimId}`, exact: true }).click();
});
test.afterAll(async () => {
    if (claimId) await pool.query('DELETE FROM claims WHERE id = $1', [claimId]);
    await pool.end();
});

test('loading, one request, human confirmation and persistent save', async ({ page, request }) => {
    let calls = 0;
    let release;
    const waiting = new Promise(resolve => { release = resolve; });
    await page.route(`**/api/claims/${claimId}/suggest-tag`, async route => {
        calls++;
        await waiting;
        await route.fulfill({ json: { tag: 'Credit card', model: 'test', elapsedMs: 100, costUsd: 0 } });
    });
    await page.getByRole('button', { name: 'Auto-tag', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Analysing…' })).toBeDisabled();
    await expect(page.getByRole('status')).toContainText('Finding a suggested tag');
    release();
    await expect(page.getByRole('status')).toContainText('Suggested tag: Credit card');
    expect(calls).toBe(1);
    const rows = await (await request.get('/api/claims')).json();
    expect(rows.find(row => row.id === claimId).tag).toBeNull();
    await page.getByRole('button', { name: 'Apply suggestion', exact: true }).click();
    await expect.poll(async () => (await pool.query('SELECT tag FROM claims WHERE id = $1', [claimId])).rows[0].tag).toBe('Credit card');
    await expect(page.getByRole('button', { name: `Select claim #${claimId}`, exact: true })).toHaveCount(0);
    await page.reload();
    await page.getByRole('button', { name: 'Credit card', exact: true }).click();
    await expect(page.getByRole('button', { name: `Select claim #${claimId}`, exact: true })).toBeVisible();
});

test('ML error, retry and dismiss preserve the database', async ({ page }) => {
    let calls = 0;
    await page.route(`**/api/claims/${claimId}/suggest-tag`, async route => {
        calls++;
        if (calls === 1) await route.fulfill({ status: 429, json: { error: 'The tagging service is busy. Please try again shortly.' } });
        else await route.fulfill({ json: { tag: 'Credit card' } });
    });
    await page.getByRole('button', { name: 'Auto-tag', exact: true }).click();
    await expect(page.getByRole('article', { name: `Claim #${claimId}`, exact: true }).getByRole('alert')).toContainText('busy');
    await expect(page.getByRole('button', { name: 'Apply suggestion' })).toHaveCount(0);
    await page.getByRole('button', { name: 'Auto-tag', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Credit card');
    await page.getByRole('button', { name: 'Dismiss', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Apply suggestion' })).toHaveCount(0);
    expect((await pool.query('SELECT tag FROM claims WHERE id = $1', [claimId])).rows[0].tag).toBeNull();
});

test('malformed response is rejected and manual tagging still works', async ({ page }) => {
    await page.route(`**/api/claims/${claimId}/suggest-tag`, route => route.fulfill({ json: { tag: 'Invented tag' } }));
    await page.getByRole('button', { name: 'Auto-tag', exact: true }).click();
    await expect(page.getByRole('article', { name: `Claim #${claimId}`, exact: true }).getByRole('alert')).toContainText('invalid suggestion');
    await expect(page.getByRole('button', { name: 'Apply suggestion' })).toHaveCount(0);
    await page.getByRole('combobox', { name: 'Assign tag' }).selectOption('Mortgage');
    await expect.poll(async () => (await pool.query('SELECT tag FROM claims WHERE id = $1', [claimId])).rows[0].tag).toBe('Mortgage');
});

test('a failed save keeps the suggestion and retries without a second ML request', async ({ page }) => {
    let suggestions = 0;
    let saves = 0;
    await page.route(`**/api/claims/${claimId}/suggest-tag`, route => {
        suggestions++;
        return route.fulfill({ json: { tag: 'Credit card' } });
    });
    await page.route(`**/api/claims/${claimId}/tag`, route => {
        saves++;
        return saves === 1 ? route.fulfill({ status: 503, json: { error: 'Save failed. Please retry.' } }) : route.continue();
    });
    await page.getByRole('button', { name: 'Auto-tag', exact: true }).click();
    await page.getByRole('button', { name: 'Apply suggestion' }).click();
    await expect(page.getByRole('article', { name: `Claim #${claimId}`, exact: true }).getByRole('alert')).toContainText('Save failed');
    await page.getByRole('button', { name: 'Apply suggestion' }).click();
    await expect.poll(async () => (await pool.query('SELECT tag FROM claims WHERE id = $1', [claimId])).rows[0].tag).toBe('Credit card');
    expect(suggestions).toBe(1);
});

test('the real Python model proposes a tag that is saved only after confirmation', async ({ page, request }) => {
    const content = (await pool.query('SELECT content FROM claims WHERE id = $1', [claimId])).rows[0].content;
    const prediction = await request.post('http://127.0.0.1:8000/tags', { data: { user_claim: content } });
    expect(prediction.ok()).toBeTruthy();
    const expected = await prediction.json();
    const response = page.waitForResponse(response => response.url().endsWith(`/api/claims/${claimId}/suggest-tag`));
    await page.getByRole('button', { name: 'Auto-tag', exact: true }).click();
    const suggestion = await (await response).json();
    expect(suggestion.model).toBe('logistic_regression');
    expect(suggestion.modelVersion).toBe(expected.modelVersion);
    await expect(page.getByRole('status')).toContainText(`Suggested tag: ${expected.tag}`);
    await expect(page.getByText(/ML suggestion · Response time: \d+ ms/)).toBeVisible();
    expect((await pool.query('SELECT tag FROM claims WHERE id = $1', [claimId])).rows[0].tag).toBeNull();
    await page.getByRole('button', { name: 'Apply suggestion', exact: true }).click();
    await expect.poll(async () => (await pool.query('SELECT tag FROM claims WHERE id = $1', [claimId])).rows[0].tag).toBe(expected.tag);
});
