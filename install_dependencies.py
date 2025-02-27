import subprocess
import sys
import os

def main():
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"Error: {req_file} not found. Please create it first.")
        sys.exit(1)
    
    print(f"Installing dependencies from {req_file}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print("Failed to install dependencies.")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
