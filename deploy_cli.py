#!/usr/bin/env python3
"""
ML Models Deployment CLI
Simple command-line interface for deploying ML models
"""
import sys
import subprocess
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_menu():
    """Display main menu"""
    print_header("ML Models Deployment CLI")
    print("Choose a deployment option:\n")
    print("  [1] 🏠 Deploy House Price API locally (Docker)")
    print("  [2] 🏠 Test House Price API")
    print("  [3] 🏠 View House Price API logs")
    print("  [4] 🏠 Stop House Price API")
    print()
    print("  [5] ❤️  Deploy Heart Disease to AWS SageMaker")
    print("  [6] ❤️  Test SageMaker Endpoint")
    print("  [7] ❤️  Delete SageMaker Endpoint")
    print()
    print("  [8] 📚 View Documentation")
    print("  [9] 🔧 Install Dependencies")
    print()
    print("  [0] Exit")
    print()


def deploy_api_local():
    """Deploy House Price API locally with Docker"""
    print_header("Deploying House Price API Locally")
    
    print("Starting Docker containers...")
    try:
        subprocess.run([
            "docker-compose", "-f", "docker-compose-api.yml", "up", "--build", "-d"
        ], check=True)
        
        print("\n✅ API deployed successfully!")
        print("\n📍 Access points:")
        print("   - API: http://localhost:8000")
        print("   - Swagger UI: http://localhost:8000")
        print("   - Health Check: http://localhost:8000/health")
        print("\nTo test: python test_api.py")
        print("To view logs: docker-compose -f docker-compose-api.yml logs -f")
        
    except subprocess.CalledProcessError:
        print("\n❌ Deployment failed!")
        print("Make sure Docker is installed and running.")
    except FileNotFoundError:
        print("\n❌ docker-compose not found!")
        print("Please install Docker Desktop or docker-compose.")


def test_api():
    """Test House Price API"""
    print_header("Testing House Price API")
    
    if not os.path.exists("test_api.py"):
        print("❌ test_api.py not found!")
        return
    
    try:
        subprocess.run(["python", "test_api.py"], check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Tests failed!")
    except FileNotFoundError:
        print("\n❌ Python not found!")


def view_api_logs():
    """View API logs"""
    print_header("House Price API Logs")
    
    try:
        subprocess.run([
            "docker-compose", "-f", "docker-compose-api.yml", "logs", "-f"
        ])
    except KeyboardInterrupt:
        print("\n\nLog streaming stopped.")
    except subprocess.CalledProcessError:
        print("\n❌ Could not view logs!")


def stop_api():
    """Stop House Price API"""
    print_header("Stopping House Price API")
    
    try:
        subprocess.run([
            "docker-compose", "-f", "docker-compose-api.yml", "down"
        ], check=True)
        print("\n✅ API stopped successfully!")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to stop API!")


def deploy_sagemaker():
    """Deploy to AWS SageMaker"""
    print_header("Deploy Heart Disease Model to AWS SageMaker")
    
    if not os.path.exists("sagemaker-deployment/deploy_sagemaker.py"):
        print("❌ SageMaker deployment script not found!")
        return
    
    print("⚠️  Make sure you have:")
    print("  1. AWS account and credentials configured")
    print("  2. IAM role with SageMaker permissions")
    print("  3. Installed: boto3, sagemaker\n")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    try:
        os.chdir("sagemaker-deployment")
        subprocess.run(["python", "deploy_sagemaker.py"], check=True)
        os.chdir("..")
    except subprocess.CalledProcessError:
        print("\n❌ Deployment failed!")
        os.chdir("..")
    except FileNotFoundError:
        print("\n❌ Python not found!")


def test_sagemaker():
    """Test SageMaker endpoint"""
    print_header("Testing SageMaker Endpoint")
    
    if not os.path.exists("sagemaker-deployment/test_sagemaker.py"):
        print("❌ Test script not found!")
        return
    
    try:
        os.chdir("sagemaker-deployment")
        subprocess.run(["python", "test_sagemaker.py"], check=True)
        os.chdir("..")
    except subprocess.CalledProcessError:
        print("\n❌ Tests failed!")
        os.chdir("..")
    except FileNotFoundError:
        print("\n❌ Python not found!")


def delete_sagemaker_endpoint():
    """Delete SageMaker endpoint"""
    print_header("Delete SageMaker Endpoint")
    
    endpoint_name = input("Enter endpoint name (heart-disease-endpoint): ").strip()
    if not endpoint_name:
        endpoint_name = "heart-disease-endpoint"
    
    print(f"\n⚠️  This will delete endpoint: {endpoint_name}")
    print("⚠️  You will stop being charged for this endpoint.\n")
    
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    try:
        subprocess.run([
            "aws", "sagemaker", "delete-endpoint",
            "--endpoint-name", endpoint_name
        ], check=True)
        print(f"\n✅ Endpoint '{endpoint_name}' deleted successfully!")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to delete endpoint!")
        print("Make sure AWS CLI is configured and the endpoint exists.")
    except FileNotFoundError:
        print("\n❌ AWS CLI not found!")
        print("Please install AWS CLI: https://aws.amazon.com/cli/")


def view_documentation():
    """View documentation menu"""
    print_header("Documentation")
    
    docs = {
        "1": ("DEPLOYMENT_MASTER.md", "Complete Deployment Guide"),
        "2": ("API_DEPLOYMENT_GUIDE.md", "House Price API Deployment"),
        "3": ("SAGEMAKER_DEPLOYMENT_GUIDE.md", "AWS SageMaker Deployment"),
        "4": ("QUICKSTART.md", "Quick Start Guide"),
        "5": ("README_DEPLOYMENT.md", "Docker Deployment Guide"),
    }
    
    print("Available documentation:\n")
    for key, (file, desc) in docs.items():
        print(f"  [{key}] {desc}")
    print("\n  [0] Back to main menu\n")
    
    choice = input("Select documentation to view: ").strip()
    
    if choice == "0":
        return
    
    if choice in docs:
        file, _ = docs[choice]
        if os.path.exists(file):
            # Try to open with default viewer
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", file])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["start", file], shell=True)
                else:  # Linux
                    subprocess.run(["xdg-open", file])
                print(f"\n✅ Opening {file}...")
            except:
                # Fallback: print first 50 lines
                print(f"\n📄 {file}:")
                print("-" * 70)
                with open(file, 'r') as f:
                    lines = f.readlines()[:50]
                    print(''.join(lines))
                    if len(lines) == 50:
                        print("\n... (view full file for more)")
        else:
            print(f"\n❌ {file} not found!")
    else:
        print("\n❌ Invalid choice!")


def install_dependencies():
    """Install dependencies"""
    print_header("Install Dependencies")
    
    print("Choose what to install:\n")
    print("  [1] API dependencies (fastapi, uvicorn)")
    print("  [2] SageMaker dependencies (boto3, sagemaker)")
    print("  [3] All dependencies")
    print("  [0] Cancel\n")
    
    choice = input("Your choice: ").strip()
    
    try:
        if choice == "1":
            print("\nInstalling API dependencies...")
            subprocess.run([
                "pip", "install", "fastapi", "uvicorn[standard]", "requests"
            ], check=True)
            print("\n✅ API dependencies installed!")
            
        elif choice == "2":
            print("\nInstalling SageMaker dependencies...")
            subprocess.run([
                "pip", "install", "-r", "sagemaker-deployment/requirements-deploy.txt"
            ], check=True)
            print("\n✅ SageMaker dependencies installed!")
            
        elif choice == "3":
            print("\nInstalling all dependencies...")
            subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
            subprocess.run(["pip", "install", "fastapi", "uvicorn[standard]", "requests"], check=True)
            subprocess.run([
                "pip", "install", "-r", "sagemaker-deployment/requirements-deploy.txt"
            ], check=True)
            print("\n✅ All dependencies installed!")
            
        elif choice == "0":
            print("Cancelled.")
        else:
            print("❌ Invalid choice!")
            
    except subprocess.CalledProcessError:
        print("\n❌ Installation failed!")
    except FileNotFoundError:
        print("\n❌ pip not found!")


def main():
    """Main CLI loop"""
    while True:
        print_menu()
        choice = input("Enter your choice [0-9]: ").strip()
        
        if choice == "0":
            print("\n👋 Goodbye!\n")
            break
        elif choice == "1":
            deploy_api_local()
        elif choice == "2":
            test_api()
        elif choice == "3":
            view_api_logs()
        elif choice == "4":
            stop_api()
        elif choice == "5":
            deploy_sagemaker()
        elif choice == "6":
            test_sagemaker()
        elif choice == "7":
            delete_sagemaker_endpoint()
        elif choice == "8":
            view_documentation()
        elif choice == "9":
            install_dependencies()
        else:
            print("\n❌ Invalid choice! Please select 0-9.\n")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
