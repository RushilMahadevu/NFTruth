#!/bin/bash

echo "🚀 Deploying NFTruth to Kubernetes..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check for lightweight deployment flag
LIGHTWEIGHT_MODE=false
if [[ "$1" == "--lightweight" ]] || [[ "$1" == "-l" ]]; then
    LIGHTWEIGHT_MODE=true
    echo "💡 Using lightweight deployment configuration"
fi

# Auto-detect if we should use lightweight mode based on available resources
if [ "$LIGHTWEIGHT_MODE" = false ]; then
    DOCKER_MEMORY=$(docker system info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
    DOCKER_MEMORY_MB=$((DOCKER_MEMORY / 1024 / 1024))
    
    if [ $DOCKER_MEMORY_MB -lt 8000 ] && [ $DOCKER_MEMORY_MB -gt 0 ]; then
        echo "💡 Auto-detected limited resources (${DOCKER_MEMORY_MB}MB), using lightweight mode"
        LIGHTWEIGHT_MODE=true
    fi
fi

# Build the Docker image (make sure Docker is running)
echo "📦 Building Docker image..."
docker build -t nftruth-api:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

# For Minikube, load the image
if command -v minikube &> /dev/null && minikube status &> /dev/null; then
    echo "📥 Loading image into Minikube..."
    minikube image load nftruth-api:latest
fi

# Apply Kubernetes manifests in order
echo "🔧 Applying Kubernetes manifests..."

echo "  📁 Creating namespace..."
kubectl apply -f k8s/namespace.yaml

echo "  ⚙️  Creating configmap..."
kubectl apply -f k8s/configmap.yaml

echo "  💾 Creating persistent volume..."
kubectl apply -f k8s/persistent-volume.yaml

echo "  🚀 Creating deployment..."
if [ "$LIGHTWEIGHT_MODE" = true ]; then
    kubectl apply -f k8s/deployment-lightweight.yaml
else
    kubectl apply -f k8s/deployment.yaml
fi

echo "  🌐 Creating service..."
kubectl apply -f k8s/service.yaml

echo "  📊 Creating HPA..."
if [ "$LIGHTWEIGHT_MODE" = true ]; then
    kubectl apply -f k8s/hpa-lightweight.yaml
else
    kubectl apply -f k8s/hpa.yaml
fi

# Optionally apply ingress (uncomment if you have ingress controller)
# echo "  🔗 Creating ingress..."
# kubectl apply -f k8s/ingress.yaml

echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/nftruth-api -n nftruth

if [ $? -eq 0 ]; then
    echo "✅ Deployment complete!"
    echo ""
    if [ "$LIGHTWEIGHT_MODE" = true ]; then
        echo "💡 Deployed in lightweight mode (optimized for limited resources)"
        echo "   - 1 replica with 256Mi memory limit"
        echo "   - Auto-scaling: 1-2 replicas"
    else
        echo "💪 Deployed in standard mode (full resources)"
        echo "   - 2 replicas with 512Mi memory limit each"
        echo "   - Auto-scaling: 2-5 replicas"
    fi
    echo ""
    echo "📊 Current status:"
    kubectl get pods -n nftruth
    echo ""
    echo "🔍 Useful commands:"
    echo "  Check status: ./k8s/monitor.sh"
    echo "  View logs: kubectl logs -f deployment/nftruth-api -n nftruth"
    echo "  Port forward: kubectl port-forward service/nftruth-service 8000:80 -n nftruth"
    echo "  Access API: http://localhost:8000 (after port-forward)"
    echo ""
    echo "💡 To increase Docker memory: Docker Desktop > Preferences > Resources"
else
    echo "❌ Deployment failed or timed out!"
    echo "Check status with: kubectl get pods -n nftruth"
    echo "View logs with: kubectl logs -n nftruth <pod-name>"
    exit 1
fi
