/** @type {import('next').NextConfig} */
const nextConfig = {
  // Permit production verification alongside a running development server.
  distDir: process.env.NEXT_DIST_DIR || '.next',
};

export default nextConfig;
