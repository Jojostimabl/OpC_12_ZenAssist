import { ALLOWED_TAGS } from './constants/tags.js';

const BASE_URL = '/api/claims';

export async function fetchClaims(tag) {
  const url = tag
    ? `${BASE_URL}?tag=${encodeURIComponent(tag)}`
    : BASE_URL;

  const response = await fetch(url);

  if (!response.ok) {
    throw new Error('Failed to fetch claims', { cause: response });
  }

  return response.json();
}

export async function updateClaimTag(claimId, tag) {
  const response = await fetch(`${BASE_URL}/${claimId}/tag`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ tag }),
  });

  if (!response.ok) {
    throw new Error(await errorMessage(response, 'The tag could not be saved. Please try again.'));
  }
}

async function errorMessage(response, fallback) {
  try {
    const data = await response.json();
    return typeof data?.error === 'string' ? data.error : fallback;
  } catch {
    return fallback;
  }
}

export async function suggestClaimTag(claimId, { signal } = {}) {
  const response = await fetch(`${BASE_URL}/${claimId}/suggest-tag`, {
    method: 'POST',
    signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)].filter(Boolean)),
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response, 'Automatic tagging is unavailable. Please try again.'));
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('The tagging service returned an invalid response. No tag was changed.');
  }
  if (!ALLOWED_TAGS.includes(data?.tag)) {
    throw new Error('The tagging service returned an invalid suggestion. No tag was changed.');
  }
  return data;
}
