#!/bin/bash

echo "🚀 Quick Deploy NFTruth (Optimized for Speed)"
echo "============================================="

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

# Step 1: Build Docker image (skip if exists and --skip-build flag is used)
if [[ "$1" == "--skip-build" ]] || [[ "$2" == "--skip-build" ]]; then
    echo "⏭️  Skipping Docker build (using existing image)"
else
    echo "📦 Building Docker image (this may take 2-5 minutes first time)..."
    docker build -t nftruth-api:latest . --quiet
    
    if [ $? -ne 0 ]; then
        echo "❌ Docker build failed!"
        exit 1
    fi
    
    # For Minikube, load the image
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        echo "📥 Loading image into Minikube..."
        minikube image load nftruth-api:latest
    fi
fi

# Step 2: Clean up any existing failed deployments
echo "🧹 Cleaning up any existing failed deployments..."
kubectl delete pod --field-selector=status.phase=Failed -n nftruth &> /dev/null || true

# Step 3: Apply Kubernetes manifests with progress
echo "🔧 Applying Kubernetes manifests..."

echo "  📁 Creating namespace..."
kubectl apply -f k8s/namespace.yaml > /dev/null

echo "  ⚙️  Creating configmap..."
kubectl apply -f k8s/configmap.yaml > /dev/null

echo "  💾 Creating persistent volume..."
kubectl apply -f k8s/persistent-volume.yaml > /dev/null

echo "  🚀 Creating deployment..."
if [ "$LIGHTWEIGHT_MODE" = true ]; then
    kubectl apply -f k8s/deployment-lightweight.yaml > /dev/null
else
    kubectl apply -f k8s/deployment.yaml > /dev/null
fi

echo "  🌐 Creating service..."
kubectl apply -f k8s/service.yaml > /dev/null

echo "  📊 Creating HPA..."
if [ "$LIGHTWEIGHT_MODE" = true ]; then
    kubectl apply -f k8s/hpa-lightweight.yaml > /dev/null
else
    kubectl apply -f k8s/hpa.yaml > /dev/null
fi

# Step 4: Wait for deployment with progress indicator
echo "⏳ Waiting for pods to start (this can take 1-3 minutes)..."
echo "   💡 Tip: Use --skip-build next time to deploy faster"

# Wait for pod to be scheduled
timeout=60
while [ $timeout -gt 0 ]; do
    POD_STATUS=$(kubectl get pods -n nftruth --no-headers 2>/dev/null | awk '{print $3}' | head -1)
    if [ "$POD_STATUS" != "" ] && [ "$POD_STATUS" != "Pending" ]; then
        break
    fi
    echo -n "."
    sleep 2
    timeout=$((timeout-2))
done
echo ""

# Check if deployment was successful
kubectl wait --for=condition=available --timeout=180s deployment/nftruth-api -n nftruth

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
    echo "🚀 Quick access commands:"
    echo "  Port forward: kubectl port-forward service/nftruth-service 8000:80 -n nftruth"
    echo "  View logs: kubectl logs -f deployment/nftruth-api -n nftruth"
    echo "  Monitor: ./k8s/monitor.sh"
    echo ""
    echo "🌐 Your API will be available at: http://localhost:8000 (after port-forward)"
    echo ""
    echo "⚡ Next deployment tip: Use './k8s/deploy-fast.sh --skip-build' to skip Docker build"
else
    echo "❌ Deployment failed or timed out!"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "  Check pod status: kubectl get pods -n nftruth"
    echo "  View pod details: kubectl describe pod -n nftruth"
    echo "  View logs: kubectl logs -n nftruth <pod-name>"
    echo ""
    echo "💡 Common issues:"
    echo "  - Docker image build failed"
    echo "  - Insufficient resources"
    echo "  - Health check endpoints not responding"
    exit 1
fi
