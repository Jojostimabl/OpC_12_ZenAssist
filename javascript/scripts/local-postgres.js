import EmbeddedPostgres from 'embedded-postgres';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { Client } from 'pg';

// Optional local alternative to Docker; never touches an existing cluster.
if (process.env.DB_NAME !== 'zenassist_mission2' || process.env.DB_HOST !== '127.0.0.1' || process.env.DB_PORT !== '55432') {
    throw new Error('Local PostgreSQL requires the isolated .env.example database settings.');
}
const databaseDir = resolve('.local-postgres');
const postgres = new EmbeddedPostgres({
    databaseDir, user: process.env.DB_USER, password: process.env.DB_PASSWORD,
    port: 55432, persistent: true, authMethod: 'scram-sha-256',
    initdbFlags: ['--encoding=UTF8', '--locale=C'],
    postgresFlags: ['-h', '127.0.0.1'],
});
if (!existsSync(resolve(databaseDir, 'PG_VERSION'))) await postgres.initialise();
await postgres.start();
const client = new Client({ host: '127.0.0.1', port: 55432, database: 'postgres', user: process.env.DB_USER, password: process.env.DB_PASSWORD });
await client.connect();
try {
    const existing = await client.query('SELECT 1 FROM pg_database WHERE datname = $1', ['zenassist_mission2']);
    if (!existing.rowCount) await client.query("CREATE DATABASE zenassist_mission2 ENCODING 'UTF8' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0");
} finally {
    await client.end();
}
console.log('Local PostgreSQL ready on 127.0.0.1:55432. Ctrl+C to stop; data is preserved.');
// Keep the parent alive so the package can shut down PostgreSQL on exit.
setInterval(() => {}, 60_000);
