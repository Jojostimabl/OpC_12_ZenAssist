'use client';

import { useState } from 'react';
import styles from './TagSelector.module.css';
import { ALLOWED_TAGS } from '@/constants/tags.js';
import { updateClaimTag } from '@/api-client.js';

export default function TagSelector({ claim, onTagUpdate }) {
    const [isUpdating, setIsUpdating] = useState(false);
    const [error, setError] = useState('');

    const handleTagSelect = async (tag) => {
        if (!claim) return;

        setIsUpdating(true);
        setError('');
        try {
            await updateClaimTag(claim.id, tag);
            onTagUpdate(claim.id, tag);
        } catch (error) {
            setError(error.message || 'The tag could not be saved. Please try again.');
        } finally {
            setIsUpdating(false);
        }
    };

    if (!claim) {
        return (
            <div className={styles.container}>
                <div className={styles.placeholder}>
                    <h3>Select a claim to assign tags</h3>
                </div>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h2>Claim #{claim.id}</h2>
            </div>

            <div className={styles.content}>
                <div className={styles.claimContent}>
                    <h3>Content</h3>
                    <p>{claim.content}</p>
                </div>

                <div className={styles.tagSection}>
                    <h4>Assign tag</h4>

                    <div className={styles.tagSelector}>
                        <select
                            className={styles.select}
                            aria-label="Assign tag"
                            value={claim.tag ?? ''}
                            onChange={(e) => handleTagSelect(e.target.value)}
                            disabled={isUpdating}
                        >
                            <option value="" disabled>
                                Select a tag
                            </option>

                            {ALLOWED_TAGS.map((tag) => (
                                <option key={tag} value={tag}>
                                    {tag}
                                </option>
                            ))}
                        </select>

                        {(!claim.tag || claim.tag === '') && (
                            <svg
                                className={styles.chevronIcon}
                                width="12"
                                height="12"
                                viewBox="0 0 292.4 292.4"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    fill="var(--foreground-muted)"
                                    d="M287 69.4a17.6 17.6 0 0 0-13-5.4H18.4c-5 0-9.3 1.8-12.9 5.4A17.6 17.6 0 0 0 0 82.2c0 5 1.8 9.3 5.4 12.9l128 127.9c3.6 3.6 7.8 5.4 12.8 5.4s9.2-1.8 12.8-5.4L287 95c3.5-3.5 5.4-7.8 5.4-12.8 0-5-1.9-9.2-5.5-12.8z"
                                />
                            </svg>
                        )}

                        {isUpdating && (
                            <div className={styles.updating}>
                                Updating...
                            </div>
                        )}
                    </div>
                    {error && <p role="alert">{error}</p>}
                </div>
            </div>
        </div>
    );
}
