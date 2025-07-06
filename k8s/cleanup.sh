#!/bin/bash

echo "🧹 Cleaning up NFTruth Kubernetes resources..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace nftruth &> /dev/null; then
    echo "⚠️  Namespace 'nftruth' not found. Nothing to clean up."
    exit 0
fi

# Show what will be deleted
echo "📋 Resources to be deleted:"
kubectl get all -n nftruth

echo ""
read -p "🗑️  Are you sure you want to delete all NFTruth resources? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 0
fi

# Delete all resources in the namespace
echo "🗑️  Deleting namespace and all resources..."
kubectl delete namespace nftruth

# Clean up persistent volumes (they might not be automatically deleted)
echo ""
read -p "🗑️  Do you want to delete persistent volumes? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete pv nftruth-pv 2>/dev/null && echo "✅ Persistent volume deleted" || echo "⚠️  No persistent volume found"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "🔄 To redeploy:"
echo "  ./k8s/deploy.sh"
