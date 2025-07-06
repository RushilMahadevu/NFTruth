#!/bin/bash

echo "⚡ ULTRA-FAST NFTruth Deploy (No Storage)"
echo "========================================"

# Complete cleanup first
echo "🧹 Complete cleanup..."
kubectl delete namespace nftruth --grace-period=0 --force &> /dev/null || true
kubectl delete pv nftruth-pv --grace-period=0 --force &> /dev/null || true

# Wait a moment for cleanup
sleep 2

# Create namespace
echo "📁 Creating namespace..."
kubectl create namespace nftruth

# Create configmap
echo "⚙️  Creating configmap..."
kubectl create configmap nftruth-config \
  --from-literal=ENVIRONMENT=production \
  --from-literal=LOG_LEVEL=INFO \
  --from-literal=PORT=8000 \
  --from-literal=HOST=0.0.0.0 \
  -n nftruth

# Create super lightweight deployment WITHOUT persistent storage
echo "🚀 Creating lightweight deployment (no storage)..."
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

# Create service
echo "🌐 Creating service..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: nftruth-service
  namespace: nftruth
  labels:
    app: nftruth-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: nftruth-api
EOF

# Wait for deployment with progress
echo "⏳ Waiting for pod to start..."
kubectl wait --for=condition=available --timeout=60s deployment/nftruth-api -n nftruth

if [ $? -eq 0 ]; then
    echo "✅ DEPLOYMENT SUCCESSFUL!"
    echo ""
    echo "📊 Status:"
    kubectl get pods -n nftruth
    echo ""
    echo "🚀 IMMEDIATE ACCESS:"
    echo "  kubectl port-forward service/nftruth-service 8000:80 -n nftruth"
    echo ""
    echo "🌐 Your API will be at: http://localhost:8000"
    echo ""
    echo "⚡ This deployment:"
    echo "  • No persistent storage (faster startup)"
    echo "  • Single replica (minimal resources)"
    echo "  • Logs stored in container (temporary)"
    echo ""
    echo "💡 To add storage later: ./k8s/deploy.sh"
else
    echo "❌ Deployment failed!"
    kubectl get pods -n nftruth
    kubectl describe pod -n nftruth
fi
