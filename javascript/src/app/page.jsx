'use client';

import { useState, useEffect } from 'react';
import styles from "./page.module.css";
import Inboxes from '@/components/Inboxes.jsx';
import Inbox from '@/components/Inbox.jsx';
import TagSelector from '@/components/TagSelector.jsx';
import { fetchClaims } from '@/api-client.js';

export default function Home() {
    const [selectedInboxId, setSelectedInboxId] = useState('untagged');
    const [selectedClaim, setSelectedClaim] = useState(null);
    const [claims, setClaims] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadClaims = async () => {
            setLoading(true);
            try {
                const tagParam = selectedInboxId === 'untagged' ? 'untagged' : selectedInboxId;
                const fetchedClaims = await fetchClaims(tagParam);
                setClaims(fetchedClaims);
            } catch (error) {
                console.error('Error fetching claims:', error);
                setClaims([]);
            } finally {
                setLoading(false);
            }
        };

        loadClaims();
    }, [selectedInboxId]);

    const handleSelectInbox = (inboxId) => {
        setSelectedInboxId(inboxId);
        setSelectedClaim(null);
    };

    const handleSelectClaim = (claim) => {
        setSelectedClaim(claim);
    };

    const handleTagClick = (tag) => {
        setSelectedInboxId(tag);
        setSelectedClaim(null);
    };

    const handleTagUpdate = (claimId, tag) => {
        setSelectedClaim(current => current?.id === claimId ? { ...current, tag } : current);
        const tagParam = selectedInboxId === 'untagged' ? 'untagged' : selectedInboxId;
        fetchClaims(tagParam).then(setClaims).catch(() => {
            // The saved tag remains visible if a subsequent list refresh fails.
            setClaims(current => current.map(claim => claim.id === claimId ? { ...claim, tag } : claim));
        });
    };

    return (
        <div className={styles.page}>
            <header className={styles.header}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                    src="/za-logo.svg"
                    alt="ZA Logo"
                    className={styles.logo}
                />

                <h1>Claims Management System</h1>
            </header>

            <main id="main-content" className={styles.mainContent}>
                <nav className={styles.sidebar}>
                    <Inboxes
                        onSelectInbox={handleSelectInbox}
                        selectedInboxId={selectedInboxId}
                    />
                </nav>

                <section className={styles.contentArea}>
                    <Inbox
                        inboxId={selectedInboxId}
                        onSelectClaim={handleSelectClaim}
                        selectedClaimId={selectedClaim?.id}
                        claims={claims}
                        loading={loading}
                        onTagClick={handleTagClick}
                        onTagUpdate={handleTagUpdate}
                    />
                </section>

                <aside className={styles.detailsPanel}>
                    <TagSelector
                        claim={selectedClaim}
                        onTagUpdate={handleTagUpdate}
                    />
                </aside>
            </main>
        </div>
    );
}
