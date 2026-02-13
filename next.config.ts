import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    // Ignore node-specific modules when bundling for the browser
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    // Handle TensorFlow.js and MediaPipe modules
    config.module.rules.push({
      test: /\.node$/,
      use: 'node-loader',
    });

    return config;
  },
  // Set empty turbopack config to acknowledge webpack config
  turbopack: {},
};

export default nextConfig;
