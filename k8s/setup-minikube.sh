#!/bin/bash

echo "🔧 Setting up Minikube for NFTruth..."

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "❌ Minikube not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install minikube
    else
        echo "Please install Homebrew first or install Minikube manually:"
        echo "https://minikube.sigs.k8s.io/docs/start/"
        exit 1
    fi
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check available Docker memory
DOCKER_MEMORY=$(docker system info --format '{{.MemTotal}}' 2>/dev/null)
DOCKER_MEMORY_MB=$((DOCKER_MEMORY / 1024 / 1024))

echo "📊 Available Docker memory: ${DOCKER_MEMORY_MB}MB"

# Set memory based on available resources
if [ $DOCKER_MEMORY_MB -gt 12000 ]; then
    MINIKUBE_MEMORY=8192
    MINIKUBE_CPUS=4
    echo "🚀 Using high-resource configuration (8GB RAM, 4 CPUs)"
elif [ $DOCKER_MEMORY_MB -gt 8000 ]; then
    MINIKUBE_MEMORY=6144
    MINIKUBE_CPUS=3
    echo "🚀 Using medium-resource configuration (6GB RAM, 3 CPUs)"
elif [ $DOCKER_MEMORY_MB -gt 6000 ]; then
    MINIKUBE_MEMORY=4096
    MINIKUBE_CPUS=2
    echo "🚀 Using low-resource configuration (4GB RAM, 2 CPUs)"
else
    MINIKUBE_MEMORY=3072
    MINIKUBE_CPUS=2
    echo "🚀 Using minimal-resource configuration (3GB RAM, 2 CPUs)"
fi

# Start minikube with appropriate resources
echo "🚀 Starting Minikube with ${MINIKUBE_MEMORY}MB memory and ${MINIKUBE_CPUS} CPUs..."
minikube start --cpus=$MINIKUBE_CPUS --memory=$MINIKUBE_MEMORY --disk-size=20g --driver=docker

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Minikube!"
    echo "💡 Try increasing Docker Desktop memory in Preferences > Resources"
    echo "💡 Or manually start with lower resources:"
    echo "   minikube start --cpus=2 --memory=3072 --disk-size=10g"
    exit 1
fi

# Enable addons
echo "🔌 Enabling addons..."
minikube addons enable ingress
minikube addons enable metrics-server

echo "✅ Minikube setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Make scripts executable: chmod +x k8s/*.sh"
echo "2. Deploy the application: ./k8s/deploy.sh"
echo "3. Monitor the deployment: ./k8s/monitor.sh"
echo ""
echo "🔍 Minikube dashboard (optional):"
echo "  minikube dashboard"
