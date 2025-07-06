#!/bin/bash

# NFTruth Docker Management Scripts

case "$1" in
    "build")
        echo "🔨 Building NFTruth API Docker image..."
        docker build -t nftruth-api .
        ;;
    "run")
        echo "🚀 Starting NFTruth API..."
        docker-compose up -d
        echo "✅ API running at http://localhost:8000"
        echo "📚 API docs at http://localhost:8000/docs"
        ;;
    "stop")
        echo "🛑 Stopping NFTruth API..."
        docker-compose down
        ;;
    "logs")
        echo "📋 Showing logs..."
        docker-compose logs -f nftruth-api
        ;;
    "restart")
        echo "🔄 Restarting NFTruth API..."
        docker-compose restart nftruth-api
        ;;
    "shell")
        echo "🐚 Opening shell in container..."
        docker-compose exec nftruth-api /bin/bash
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        docker-compose down --volumes --remove-orphans
        docker image rm nftruth-api 2>/dev/null || true
        ;;
    "status")
        echo "📊 Container status:"
        docker-compose ps
        ;;
    "fix")
        echo "🔧 Fixing model compatibility issues..."
        docker-compose down
        echo "🏗️ Rebuilding with updated dependencies..."
        docker build --no-cache -t nftruth-api .
        echo "🚀 Starting with fresh build..."
        docker-compose up -d
        echo "✅ API should be running with fixed dependencies"
        ;;
    *)
        echo "NFTruth Docker Management"
        echo "Usage: $0 {build|run|stop|logs|restart|shell|clean|status}"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image"
        echo "  run     - Start the API in background"
        echo "  stop    - Stop the API"
        echo "  logs    - Show API logs"
        echo "  restart - Restart the API"
        echo "  shell   - Open shell in container"
        echo "  clean   - Clean up all Docker resources"
        echo "  status  - Show container status"
        ;;
esac