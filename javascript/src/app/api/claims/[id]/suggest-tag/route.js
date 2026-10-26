import { getClaimById } from '@/database/queries.js';
import { suggestTag } from '@/server/llm-service.js';
import { createSuggestionHandler } from '@/server/suggestion-handler.js';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const handleSuggestion = createSuggestionHandler({ getClaim: getClaimById, classify: suggestTag });

export async function POST(request, context) {
    return handleSuggestion(request, context);
}
