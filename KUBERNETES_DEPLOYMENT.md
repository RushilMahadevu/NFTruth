# 🚀 NFTruth Kubernetes Deployment Guide

## Complete Setup for 24/7 Operation

Your Kubernetes setup is now ready! Here's how to deploy your NFTruth application to run 24/7 with high availability.

## 🎯 What You Get

✅ **24/7 Operation** - Automatic restarts and monitoring  
✅ **High Availability** - Multiple replicas running simultaneously  
✅ **Auto-scaling** - Scales from 2-5 pods based on load  
✅ **Persistent Storage** - Logs and models survive restarts  
✅ **Health Monitoring** - Automatic failure detection and recovery  
✅ **Zero Downtime Updates** - Rolling deployment strategy  

## 🚀 Quick Deployment (3 Commands)

```bash
# 1. Setup local Kubernetes cluster
./k8s/setup-minikube.sh

# 2. Deploy your application
./k8s/deploy.sh

# 3. Monitor your deployment
./k8s/monitor.sh
```

## 🌐 Access Your API

```bash
# Port forward to access locally
kubectl port-forward service/nftruth-service 8000:80 -n nftruth

# Your API will be available at:
# http://localhost:8000
```

## 📊 Real-time Monitoring

```bash
# View live logs
kubectl logs -f deployment/nftruth-api -n nftruth

# Monitor pod status
watch kubectl get pods -n nftruth

# Check resource usage
kubectl top pods -n nftruth
```

## 🔄 Management Commands

```bash
# Scale manually (if needed)
kubectl scale deployment nftruth-api --replicas=3 -n nftruth

# Update deployment after code changes
docker build -t nftruth-api:latest .
kubectl rollout restart deployment/nftruth-api -n nftruth

# Clean up everything
./k8s/cleanup.sh
```

## 🌍 Free Cloud Deployment Options

### Option 1: Google Cloud (Recommended)
- **$300 free credit** for 12 months
- **Best for**: Production use
- **Setup**: See k8s/README.md for GKE commands

### Option 2: Oracle Cloud Always Free
- **Permanently free** 2 micro instances
- **Best for**: Long-term free hosting
- **Limitations**: Smaller resources

### Option 3: Local Minikube
- **Completely free**
- **Best for**: Development and testing
- **Requirements**: Your computer stays on

## 📋 Deployment Status

After running `./k8s/deploy.sh`, you should see:

```
✅ Deployment complete!

📊 Current status:
NAME                           READY   STATUS    RESTARTS   AGE
nftruth-api-xxxxxxxxx-xxxxx    1/1     Running   0          1m
nftruth-api-xxxxxxxxx-xxxxx    1/1     Running   0          1m

🔍 Useful commands:
  Check status: ./k8s/monitor.sh
  View logs: kubectl logs -f deployment/nftruth-api -n nftruth
  Port forward: kubectl port-forward service/nftruth-service 8000:80 -n nftruth
  Access API: http://localhost:8000 (after port-forward)
```

## 🎛️ Dashboard Access

```bash
# Open Kubernetes dashboard
minikube dashboard

# View your deployment in a web interface
```

## 🔧 Troubleshooting

### If pods are not starting:
```bash
kubectl describe pod -n nftruth
kubectl logs -n nftruth <pod-name>
```

### If can't access the API:
```bash
kubectl get services -n nftruth
kubectl port-forward service/nftruth-service 8000:80 -n nftruth
```

### If running out of resources:
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n nftruth

# Scale down if needed
kubectl scale deployment nftruth-api --replicas=1 -n nftruth
```

## 💰 Cost Estimation

### Local (Minikube): $0/month
- Completely free
- Uses your computer's resources

### Google Cloud: ~$20-50/month after free credits
- 2 e2-small instances
- Free for first 12 months with $300 credit

### Oracle Always Free: $0/month forever
- 2 micro instances (1/8 OCPU, 1GB RAM each)
- Permanently free tier

## 🛡️ Security Features

- ✅ Non-root containers (security best practice)
- ✅ Resource limits (prevent resource exhaustion)
- ✅ Health checks (only healthy pods receive traffic)
- ✅ Network isolation (namespace-based)

## 📈 Auto-scaling Behavior

Your deployment will automatically:
- **Scale UP** when CPU > 70% or Memory > 80%
- **Scale DOWN** when resource usage is low
- **Maintain** 2-5 replicas at all times
- **Prevent** rapid scaling changes (stabilization)

## 🚨 High Availability Features

1. **Multiple Replicas**: Always 2+ instances running
2. **Rolling Updates**: Updates happen without downtime
3. **Health Checks**: Failed containers restart automatically
4. **Load Balancing**: Traffic distributed across healthy pods
5. **Persistent Storage**: Data survives pod crashes

## 🎉 Your Application is Now:

- ✅ **Running 24/7** with automatic restarts
- ✅ **Highly Available** with multiple replicas
- ✅ **Auto-scaling** based on demand
- ✅ **Monitored** with health checks
- ✅ **Persistent** with saved data
- ✅ **Production Ready** with best practices

**Ready to go live!** 🚀
