FROM node:20-alpine
WORKDIR /app

LABEL org.opencontainers.image.title="9router"

# Install 9router from npm so we always get the latest version
RUN npm install -g 9router --no-audit --no-fund

# Copy the prebuilt app from the npm package into /app
RUN cp -r /usr/local/lib/node_modules/9router/app/* /app/ \
    && cp -r /usr/local/lib/node_modules/9router/app/.next /app/.next \
    && cp -r /usr/local/lib/node_modules/9router/src /app/src

# Ensure node-forge is available for MITM
RUN cp -r /usr/local/lib/node_modules/9router/app/node_modules/node-forge /app/node_modules/node-forge 2>/dev/null || true

ENV NODE_ENV=production
ENV PORT=20128
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1

RUN mkdir -p /app/data

# Fix permissions at runtime (handles mounted volumes)
RUN printf '#!/bin/sh\nchown -R node:node /app/data 2>/dev/null; exec su-exec node "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh
RUN apk add --no-cache su-exec

EXPOSE 20128

ENTRYPOINT ["/entrypoint.sh"]
CMD ["node", "server.js"]
