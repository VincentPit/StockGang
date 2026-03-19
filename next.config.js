/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output produces a self-contained server.js — required for the
  // multi-stage Docker build and for minimal image size.
  output: "standalone",

  async rewrites() {
    // In Docker the Next.js server proxies to the api container name.
    // Locally it falls back to localhost:8000.
    const apiBase = process.env.API_BASE_URL ?? "http://localhost:8000";
    return [
      {
        source:      "/api/:path*",
        destination: `${apiBase}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
