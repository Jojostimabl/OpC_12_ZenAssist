import { db } from '../src/database/client.js';

if (process.env.DB_NAME !== 'zenassist_mission2') throw new Error('Only the isolated mission 2 demo database can be initialised.');
try {
    const existing = await db.query("SELECT to_regclass('public.claims') AS table_name");
    if (existing.rows[0].table_name) console.log('Claims table already exists; data preserved.');
    else {
        await import('../src/database/seed.js');
        console.log('New demo database initialised with the 10 starter claims.');
    }
} finally {
    if (!db.ended) await db.end();
}
