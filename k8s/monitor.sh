#!/bin/bash

echo "📊 NFTruth Kubernetes Monitoring Dashboard"
echo "=========================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Function to print section headers
print_header() {
    echo ""
    echo "🔹 $1"
    echo "----------------------------------------"
}

# Check namespace
print_header "Namespace Status"
kubectl get namespace nftruth 2>/dev/null || echo "❌ Namespace 'nftruth' not found"

# Check pods
print_header "Pod Status"
kubectl get pods -n nftruth -o wide 2>/dev/null || echo "❌ No pods found in namespace 'nftruth'"

# Check deployments
print_header "Deployment Status"
kubectl get deployments -n nftruth 2>/dev/null || echo "❌ No deployments found"

# Check services
print_header "Service Status"
kubectl get services -n nftruth 2>/dev/null || echo "❌ No services found"

# Check HPA
print_header "Horizontal Pod Autoscaler Status"
kubectl get hpa -n nftruth 2>/dev/null || echo "❌ No HPA found"

# Check storage
print_header "Storage Status"
kubectl get pv,pvc -n nftruth 2>/dev/null || echo "❌ No storage resources found"

# Check recent events
print_header "Recent Events (Last 10)"
kubectl get events -n nftruth --sort-by='.lastTimestamp' 2>/dev/null | tail -10 || echo "❌ No events found"

# Resource usage (if metrics-server is available)
print_header "Resource Usage"
kubectl top pods -n nftruth 2>/dev/null || echo "⚠️  Metrics not available (metrics-server may not be running)"

echo ""
echo "🛠️  Useful Commands:"
echo "----------------------------------------"
echo "📋 View logs:           kubectl logs -f deployment/nftruth-api -n nftruth"
echo "🌐 Port forward:        kubectl port-forward service/nftruth-service 8000:80 -n nftruth"
echo "📈 Scale manually:      kubectl scale deployment nftruth-api --replicas=3 -n nftruth"
echo "🔄 Restart deployment:  kubectl rollout restart deployment/nftruth-api -n nftruth"
echo "🗑️  Delete everything:   ./k8s/cleanup.sh"
echo "🎛️  Minikube dashboard:  minikube dashboard"
echo ""
echo "🌍 Access your API:"
echo "1. Run: kubectl port-forward service/nftruth-service 8000:80 -n nftruth"
echo "2. Visit: http://localhost:8000"
