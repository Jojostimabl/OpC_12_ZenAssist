import { NextResponse } from 'next/server';
import { setClaimTag } from '@/database/queries.js';
import { ALLOWED_TAGS } from '@/constants/tags.js';
import { isSameOrigin, parseClaimId } from '@/server/suggestion-handler.js';

export async function PUT(request, { params }) {
  if (!isSameOrigin(request)) return NextResponse.json({ error: 'This request is not allowed.' }, { status: 403 });
  const claimId = parseClaimId((await params).id);
  if (claimId === null) return NextResponse.json({ error: 'Invalid claim ID.' }, { status: 400 });
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
  }
  try {
    const { tag } = body ?? {};
    if (!ALLOWED_TAGS.includes(tag)) {
      return NextResponse.json({ error: 'Select a valid product tag.' }, { status: 400 });
    }
    const updated = await setClaimTag(claimId, tag);
    if (!updated) return NextResponse.json({ error: 'This claim no longer exists.' }, { status: 404 });
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error updating claim tag:', error);
    return NextResponse.json({ error: 'Failed to update claim tag' }, { status: 500 });
  }
}
