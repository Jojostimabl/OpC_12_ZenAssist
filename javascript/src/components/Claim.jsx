'use client';

import { useEffect, useRef, useState } from 'react';
import { suggestClaimTag, updateClaimTag } from '@/api-client.js';
import styles from './Claim.module.css';

function AutoTagControls({ claim, onTagUpdate }) {
    const [status, setStatus] = useState('idle');
    const [suggestion, setSuggestion] = useState(null);
    const [error, setError] = useState('');
    const busy = useRef(false);
    const controller = useRef(null);

    useEffect(() => () => controller.current?.abort(), []);

    const handleSuggest = async () => {
        if (busy.current) return;
        busy.current = true;
        controller.current = new AbortController();
        setStatus('loading');
        setSuggestion(null);
        setError('');
        const started = performance.now();
        try {
            const result = await suggestClaimTag(claim.id, { signal: controller.current.signal });
            if (!controller.current.signal.aborted) setSuggestion({ ...result, clientElapsedMs: Math.round(performance.now() - started) });
        } catch (error) {
            if (!controller.current.signal.aborted) {
                setError(error.name === 'TimeoutError'
                    ? 'The tagging service took too long. Please try again.'
                    : error.message || 'Automatic tagging failed. You can assign a tag manually.');
            }
        } finally {
            busy.current = false;
            if (!controller.current.signal.aborted) setStatus('idle');
        }
    };

    const handleApply = async () => {
        if (busy.current || !suggestion) return;
        busy.current = true;
        setStatus('saving');
        setError('');
        try {
            await updateClaimTag(claim.id, suggestion.tag);
            onTagUpdate(claim.id, suggestion.tag);
            setSuggestion(null);
        } catch (error) {
            setError(error.message || 'The tag could not be saved. Please try again.');
        } finally {
            busy.current = false;
            setStatus('idle');
        }
    };

    const isBusy = status !== 'idle';
    return (
        <div className={styles.autoTagControls} aria-busy={isBusy}>
            <button type="button" className={styles.autoTagButton} onClick={handleSuggest} disabled={isBusy || !claim.content?.trim()}>
                {status === 'loading' ? 'Analysing…' : 'Auto-tag'}
            </button>
            {status === 'loading' && <p role="status">Finding a suggested tag. This may take a few seconds…</p>}
            {suggestion && (
                <div className={styles.suggestion}>
                    <p role="status">Suggested tag: <strong>{suggestion.tag}</strong></p>
                    <p>ML suggestion · Response time: {suggestion.clientElapsedMs} ms</p>
                    <p>Review before applying, or assign a different tag manually.</p>
                    <div className={styles.suggestionActions}>
                        <button type="button" className={styles.autoTagButton} onClick={handleApply} disabled={isBusy}>
                            {status === 'saving' ? 'Saving…' : 'Apply suggestion'}
                        </button>
                        <button type="button" className={styles.dismissButton} disabled={isBusy} onClick={() => { setSuggestion(null); setError(''); }}>
                            Dismiss
                        </button>
                    </div>
                </div>
            )}
            {error && <p className={styles.error} role="alert">{error}</p>}
        </div>
    );
}

export default function ClaimComponent({ claim, isSelected, onClick, onTagClick, onTagUpdate }) {
    const handleTagClick = (e) => {
        e.stopPropagation();

        if (claim.tag && onTagClick) {
            onTagClick(claim.tag);
        }
    };

    return (
        <article
            className={`${styles.container} ${isSelected ? styles.selected : ''}`}
            aria-label={`Claim #${claim.id}`}
        >
            <div className={styles.content}>
                <button
                    type="button"
                    className={styles.selectClaim}
                    onClick={onClick}
                    aria-pressed={isSelected}
                    aria-label={`Select claim #${claim.id}`}
                >
                    <span className={styles.text} id={`claim-content-${claim.id}`}>{claim.content}</span>
                </button>
                {claim.tag && (
                    <button
                        className={styles.tag}
                        onClick={handleTagClick}
                        aria-label={`Navigate to ${claim.tag} inbox`}
                        title={`Go to ${claim.tag} inbox`}
                    >
                        {claim.tag}
                    </button>
                )}
                {isSelected && <AutoTagControls key={`${claim.id}:${claim.tag ?? ''}`} claim={claim} onTagUpdate={onTagUpdate} />}
            </div>
        </article>
    );
}
