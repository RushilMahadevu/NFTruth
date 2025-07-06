#!/bin/bash

echo "🔧 NFTruth API Key Configuration"
echo "================================"

# Check for API keys
OPENSEA_KEY=${OPENSEA_API_KEY:-""}
REDDIT_CLIENT_ID=${REDDIT_CLIENT_ID:-""}
REDDIT_CLIENT_SECRET=${REDDIT_CLIENT_SECRET:-""}

if [ -z "$OPENSEA_KEY" ]; then
    echo "⚠️  OPENSEA_API_KEY not set"
    echo "💡 Get your free API key from: https://docs.opensea.io/reference/api-keys"
    echo ""
    read -p "🔑 Enter your OpenSea API key (or press Enter to skip): " OPENSEA_KEY
fi

if [ -z "$REDDIT_CLIENT_ID" ]; then
    echo "⚠️  REDDIT_CLIENT_ID not set"
    echo "💡 Create Reddit app at: https://www.reddit.com/prefs/apps"
    echo ""
    read -p "🔑 Enter your Reddit Client ID (or press Enter to skip): " REDDIT_CLIENT_ID
fi

if [ -z "$REDDIT_CLIENT_SECRET" ]; then
    echo "⚠️  REDDIT_CLIENT_SECRET not set"
    echo ""
    read -p "🔑 Enter your Reddit Client Secret (or press Enter to skip): " REDDIT_CLIENT_SECRET
fi

# Update the deployment
echo "🚀 Updating deployment with API keys..."

# Delete existing deployment if it exists
kubectl delete deployment nftruth-api -n nftruth &> /dev/null || true

# Create deployment with API keys
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nftruth-api
  namespace: nftruth
  labels:
    app: nftruth-api
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  selector:
    matchLabels:
      app: nftruth-api
  template:
    metadata:
      labels:
        app: nftruth-api
    spec:
      containers:
      - name: nftruth-api
        image: nftruth-api:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: PORT
          value: "8000"
        - name: HOST
          value: "0.0.0.0"
        - name: OPENSEA_API_KEY
          value: "${OPENSEA_KEY}"
        - name: REDDIT_CLIENT_ID
          value: "${REDDIT_CLIENT_ID}"
        - name: REDDIT_CLIENT_SECRET
          value: "${REDDIT_CLIENT_SECRET}"
        resources:
          requests:
            memory: "128Mi"
            cpu: "50m"
          limits:
            memory: "256Mi"
            cpu: "150m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      restartPolicy: Always
EOF

# Wait for deployment
echo "⏳ Waiting for deployment..."
kubectl wait --for=condition=available --timeout=60s deployment/nftruth-api -n nftruth

if [ $? -eq 0 ]; then
    echo "✅ API Keys configured successfully!"
    echo ""
    echo "🔑 Configured API Keys:"
    [ ! -z "$OPENSEA_KEY" ] && echo "  ✅ OpenSea API Key: ${OPENSEA_KEY:0:8}..." || echo "  ❌ OpenSea API Key: Not set"
    [ ! -z "$REDDIT_CLIENT_ID" ] && echo "  ✅ Reddit Client ID: ${REDDIT_CLIENT_ID:0:8}..." || echo "  ❌ Reddit Client ID: Not set"
    [ ! -z "$REDDIT_CLIENT_SECRET" ] && echo "  ✅ Reddit Client Secret: ${REDDIT_CLIENT_SECRET:0:8}..." || echo "  ❌ Reddit Client Secret: Not set"
    echo ""
    echo "🚀 Test your API:"
    echo "  curl http://localhost:8000/health"
    echo ""
    echo "📖 Get API Keys:"
    echo "  OpenSea: https://docs.opensea.io/reference/api-keys"
    echo "  Reddit: https://www.reddit.com/prefs/apps"
else
    echo "❌ Deployment failed!"
    kubectl get pods -n nftruth
fi
