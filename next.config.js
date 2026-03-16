/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:5343/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig
