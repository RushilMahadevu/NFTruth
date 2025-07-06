# NFTruth Kubernetes Deployment

This directory contains Kubernetes manifests and scripts to deploy NFTruth as a highly available, 24/7 running service.

## 🎯 Features

- **24/7 Operation**: Automatic restarts and health monitoring
- **High Availability**: Multiple replicas with rolling updates
- **Auto-scaling**: Horizontal Pod Autoscaler based on CPU/memory usage
- **Persistent Storage**: Logs and model outputs preserved across restarts
- **Health Monitoring**: Liveness and readiness probes
- **Resource Management**: CPU and memory limits/requests

## 🚀 Quick Start (Local with Minikube)

### Prerequisites
- Docker Desktop running with **at least 4GB memory**
- kubectl installed
- Minikube (will be installed automatically if using Homebrew)

### 0. Check Your System Resources
```bash
./k8s/check-resources.sh
```

### 1. Setup Minikube
```bash
chmod +x k8s/*.sh
./k8s/setup-minikube.sh
```

### 2. Deploy NFTruth
```bash
# FIRST: Set up your API keys (required for full functionality)
export OPENSEA_API_KEY="your_opensea_api_key_here"
export REDDIT_CLIENT_ID="your_reddit_client_id_here"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret_here"

# Auto-detects your system resources
./k8s/deploy.sh

# Or force lightweight mode for limited resources
./k8s/deploy.sh --lightweight

# For quick testing without API keys (limited functionality)
./k8s/deploy-quick.sh
```

### 3. Monitor Deployment
```bash
./k8s/monitor.sh
```

### 4. Access Your API
```bash
# Port forward to access locally
kubectl port-forward service/nftruth-service 8000:80 -n nftruth

# Visit http://localhost:8000 in your browser
```

## 📁 File Structure

```
k8s/
├── namespace.yaml          # Kubernetes namespace
├── configmap.yaml         # Application configuration
├── persistent-volume.yaml # Storage for logs and models
├── deployment.yaml        # Main application deployment (standard)
├── deployment-lightweight.yaml # Lightweight deployment (limited resources)
├── service.yaml           # Internal service
├── ingress.yaml           # External access (optional)
├── hpa.yaml              # Auto-scaling configuration (standard)
├── hpa-lightweight.yaml  # Auto-scaling configuration (lightweight)
├── setup-minikube.sh     # Local cluster setup
├── deploy.sh             # Deployment script (auto-detects resources)
├── deploy-quick.sh       # Ultra-fast deployment (no storage)
├── deploy-fast.sh        # Fast deployment with optimizations
├── configure-api-keys.sh # API key configuration
├── monitor.sh            # Monitoring dashboard
├── cleanup.sh            # Cleanup script
├── check-resources.sh    # Resource requirement checker
└── README.md             # This file
```

## 🔧 Configuration

### Resource Modes

#### Standard Mode (6GB+ Docker memory)
- **Memory**: 256Mi request, 512Mi limit per pod
- **CPU**: 100m request, 250m limit per pod
- **Replicas**: 2-5 (auto-scaling based on load)

#### Lightweight Mode (4-6GB Docker memory)
- **Memory**: 128Mi request, 256Mi limit per pod
- **CPU**: 50m request, 150m limit per pod
- **Replicas**: 1-2 (auto-scaling based on load)

### Storage
- **Volume Size**: 10Gi persistent storage
- **Mount Points**: `/app/logs` and `/app/model_outputs`

### Health Checks
- **Readiness Probe**: Checks `/health` endpoint every 5-10s
- **Liveness Probe**: Checks `/health` endpoint every 10-15s

## 🌐 Deployment Options

### Local Development (Minikube)
```bash
# Already covered in Quick Start
./k8s/setup-minikube.sh
./k8s/deploy.sh
```

### Google Cloud Platform (GKE) - $300 Free Credit
```bash
# Install gcloud CLI first
gcloud container clusters create nftruth-cluster \
    --zone=us-central1-a \
    --num-nodes=2 \
    --machine-type=e2-small \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=3

# Get credentials
gcloud container clusters get-credentials nftruth-cluster --zone=us-central1-a

# Deploy
./k8s/deploy.sh
```

### Oracle Cloud (Always Free)
```bash
# Create OKE cluster using Oracle Cloud Console
# Download kubeconfig
# Then deploy
./k8s/deploy.sh
```

### Azure (AKS) - $200 Free Credit
```bash
# Install Azure CLI first
az aks create \
    --resource-group myResourceGroup \
    --name nftruth-cluster \
    --node-count 2 \
    --node-vm-size Standard_B2s \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 3

# Get credentials
az aks get-credentials --resource-group myResourceGroup --name nftruth-cluster

# Deploy
./k8s/deploy.sh
```

## 📊 Monitoring & Management

### View Logs
```bash
kubectl logs -f deployment/nftruth-api -n nftruth
```

### Scale Manually
```bash
kubectl scale deployment nftruth-api --replicas=3 -n nftruth
```

### Update Deployment
```bash
# After making changes to your code
docker build -t nftruth-api:latest .
kubectl rollout restart deployment/nftruth-api -n nftruth
```

### Access Minikube Dashboard
```bash
minikube dashboard
```

## 🔍 Troubleshooting

### API Key Issues
```bash
# If you see "Missing an API Key" errors:
./k8s/configure-api-keys.sh

# Check current API key configuration:
kubectl get deployment nftruth-api -n nftruth -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="OPENSEA_API_KEY")].value}'

# Update API keys without redeployment:
kubectl set env deployment/nftruth-api OPENSEA_API_KEY=your_new_key -n nftruth
```

### Get Free API Keys
- **OpenSea**: https://docs.opensea.io/reference/api-keys (Free tier: 100 requests/minute)
- **Reddit**: https://www.reddit.com/prefs/apps (Free - create "script" app)

### Docker Memory Issues
```bash
# Check your Docker memory allocation
./k8s/check-resources.sh

# If you get "Docker Desktop has only XMB memory but you specified YMB"
# 1. Open Docker Desktop
# 2. Go to Preferences > Resources
# 3. Increase Memory to at least 4GB (recommended: 6-8GB)
# 4. Click Apply & Restart
# 5. Try again: ./k8s/setup-minikube.sh
```

### Minikube Startup Issues
```bash
# If Minikube fails to start, try manual configuration:
minikube start --cpus=2 --memory=3072 --disk-size=10g --driver=docker

# For very limited resources:
minikube start --cpus=1 --memory=2048 --disk-size=5g --driver=docker
```

### Pod Not Starting
```bash
kubectl describe pod -n nftruth
kubectl logs -n nftruth <pod-name>
```

### Service Not Accessible
```bash
kubectl get services -n nftruth
kubectl describe service nftruth-service -n nftruth
```

### Storage Issues
```bash
kubectl get pv,pvc -n nftruth
kubectl describe pvc nftruth-pvc -n nftruth
```

### Check Resource Usage
```bash
kubectl top pods -n nftruth
kubectl get hpa -n nftruth
```

## 🧹 Cleanup

### Remove Everything
```bash
./k8s/cleanup.sh
```

### Stop Minikube
```bash
minikube stop
minikube delete  # Complete removal
```

## 💰 Cost Optimization

### Free Tier Limits
- **GCP**: $300 credit (12 months)
- **Azure**: $200 credit (12 months)  
- **Oracle**: Always free (2 micro instances)
- **Minikube**: Completely free (local)

### Resource Efficiency
- Uses small resource requests/limits
- Auto-scaling prevents over-provisioning
- Non-root container for security

## 🔒 Security Features

- Non-root user (UID 1000)
- Resource limits prevent resource exhaustion
- Network policies can be added
- Health checks prevent unhealthy pods from receiving traffic

## 📈 Auto-scaling

The HPA (Horizontal Pod Autoscaler) will:
- Scale up when CPU > 70% or Memory > 80%
- Scale down when usage is low
- Maintain 2-5 replicas
- Prevent thrashing with stabilization windows

## 🚨 High Availability

- **Multiple replicas**: Always 2+ instances running
- **Rolling updates**: Zero-downtime deployments
- **Health checks**: Automatic restart of failed containers
- **Persistent storage**: Data survives pod restarts
- **Load balancing**: Traffic distributed across healthy pods

Your NFTruth application will run 24/7 with automatic recovery and scaling!
