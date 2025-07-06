#!/bin/bash

echo "🔧 Docker Desktop Memory Configuration Helper"
echo "============================================="

# Get current Docker memory
DOCKER_MEMORY=$(docker system info --format '{{.MemTotal}}' 2>/dev/null || echo "0")
DOCKER_MEMORY_MB=$((DOCKER_MEMORY / 1024 / 1024))

if [ $DOCKER_MEMORY -eq 0 ]; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "📊 Current Docker memory allocation: ${DOCKER_MEMORY_MB}MB"

# Check if memory is sufficient
if [ $DOCKER_MEMORY_MB -lt 4000 ]; then
    echo "❌ Insufficient memory for Kubernetes deployment"
    echo ""
    echo "🔧 To fix this:"
    echo "1. Open Docker Desktop"
    echo "2. Go to Preferences (⚙️) > Resources"
    echo "3. Increase Memory to at least 4GB (recommended: 6-8GB)"
    echo "4. Click 'Apply & Restart'"
    echo ""
    echo "💡 Minimum recommendations:"
    echo "   - Memory: 4GB (for lightweight mode)"
    echo "   - Memory: 6-8GB (for standard mode)"
    echo "   - Swap: 1GB"
    echo "   - Disk: 10GB"
elif [ $DOCKER_MEMORY_MB -lt 6000 ]; then
    echo "⚠️  Limited memory detected - will use lightweight deployment"
    echo ""
    echo "💡 For better performance, consider increasing to 6-8GB:"
    echo "1. Docker Desktop > Preferences > Resources"
    echo "2. Increase Memory slider"
    echo "3. Apply & Restart"
    echo ""
    echo "🚀 You can still deploy with current settings using:"
    echo "   ./k8s/deploy.sh --lightweight"
else
    echo "✅ Memory allocation is sufficient for standard deployment"
    echo ""
    echo "🚀 Ready to deploy:"
    echo "   ./k8s/setup-minikube.sh"
    echo "   ./k8s/deploy.sh"
fi

echo ""
echo "📋 Memory usage recommendations:"
echo "   < 4GB:  Not recommended for Kubernetes"
echo "   4-6GB:  Lightweight mode (1-2 pods)"
echo "   6-8GB:  Standard mode (2-3 pods)"
echo "   8GB+:   Full mode (2-5 pods with auto-scaling)"
echo ""
echo "🔍 Current Docker resource allocation:"
docker system info | grep -E "(Total Memory|CPUs)" 2>/dev/null || echo "Unable to get detailed Docker info"
