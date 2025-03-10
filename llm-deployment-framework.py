import os
import subprocess
import json
import platform
import psutil
import torch
import logging
import requests
import zipfile
import tarfile
import time
import signal
import threading
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import huggingface_hub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM-Deployer")

class ModelDownloader:
    """Handles downloading models from various sources."""
    
    def __init__(self, models_directory: str = "./models"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.models_directory / "model_cache.json"
        self.model_cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load the model cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model cache: {str(e)}")
        return {"models": {}}
        
    def _save_cache(self) -> None:
        """Save the model cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.model_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model cache: {str(e)}")
    
    def download_model(self, model_name: str, model_specs: Dict) -> str:
        """Download a model from the specified source."""
        source = model_specs.get("source", "").lower()
        model_id = model_specs.get("model_id", model_name)
        
        # Check if model is already downloaded
        model_path = self.models_directory / model_name
        if model_name in self.model_cache.get("models", {}):
            cache_info = self.model_cache["models"][model_name]
            if model_path.exists():
                logger.info(f"Model {model_name} already exists at {model_path}")
                # Verify checksum if available
                if "checksum" in cache_info:
                    if self._verify_checksum(model_path, cache_info["checksum"]):
                        return str(model_path)
                    else:
                        logger.warning(f"Checksum verification failed for {model_name}, re-downloading")
                else:
                    return str(model_path)
        
        # Create model directory if it doesn't exist
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if source == "huggingface":
                return self._download_from_huggingface(model_name, model_id, model_path, model_specs)
            elif source == "direct":
                return self._download_direct(model_name, model_specs["url"], model_path, model_specs)
            else:
                raise ValueError(f"Unsupported model source: {source}")
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            raise
            
    def _download_from_huggingface(self, model_name: str, model_id: str, model_path: Path, model_specs: Dict) -> str:
        """Download a model from Hugging Face."""
        logger.info(f"Downloading {model_name} from Hugging Face ({model_id})...")
        
        try:
            # Check if we need a specific file or the whole repo
            specific_file = model_specs.get("specific_file")
            quantized = model_specs.get("quantized", False)
            
            if specific_file:
                # Download just the specific file
                huggingface_hub.hf_hub_download(
                    repo_id=model_id,
                    filename=specific_file,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                result_path = model_path / specific_file
            elif quantized:
                # For quantized models, get the specified format
                quant_format = model_specs.get("quant_format", "gguf")
                quant_size = model_specs.get("quant_size", "Q4_K_M")
                
                # This handles the pattern used by TheBloke and other quantizers
                filename = f"{model_name.split('-')[0]}-{model_specs.get('model_version', 'latest')}.{quant_format}"
                if quant_size:
                    filename = f"{model_name.split('-')[0]}-{model_specs.get('model_version', 'latest')}.{quant_size}.{quant_format}"
                
                huggingface_hub.hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                result_path = model_path / filename
            else:
                # Download the whole model
                huggingface_hub.snapshot_download(
                    repo_id=model_id,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                result_path = model_path
            
            # Update cache
            self.model_cache.setdefault("models", {})[model_name] = {
                "path": str(result_path),
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "huggingface",
                "model_id": model_id
            }
            self._save_cache()
            
            logger.info(f"Successfully downloaded {model_name} to {result_path}")
            return str(result_path)
            
        except Exception as e:
            logger.error(f"Failed to download from Hugging Face: {str(e)}")
            raise
    
    def _download_direct(self, model_name: str, url: str, model_path: Path, model_specs: Dict) -> str:
        """Download a model from a direct URL."""
        logger.info(f"Downloading {model_name} from {url}...")
        
        try:
            # Download file
            local_filename = model_path / url.split('/')[-1]
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(local_filename, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, 
                    desc=f"Downloading {local_filename.name}"
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract if it's an archive
            file_type = model_specs.get("file_type", "").lower()
            extract_dir = model_path / "extracted"
            
            if file_type == "zip" or local_filename.suffix == ".zip":
                logger.info(f"Extracting ZIP file: {local_filename}")
                with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                result_path = extract_dir
            elif file_type in ["tar", "tgz", "tar.gz"] or any(local_filename.suffixes[i:] == [".tar", ".gz"] for i in range(len(local_filename.suffixes))):
                logger.info(f"Extracting TAR file: {local_filename}")
                with tarfile.open(local_filename) as tar:
                    tar.extractall(path=extract_dir)
                result_path = extract_dir
            else:
                result_path = local_filename
            
            # Update cache
            self.model_cache.setdefault("models", {})[model_name] = {
                "path": str(result_path),
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "direct",
                "url": url
            }
            self._save_cache()
            
            logger.info(f"Successfully downloaded {model_name} to {result_path}")
            return str(result_path)
            
        except Exception as e:
            logger.error(f"Failed to download from URL: {str(e)}")
            raise
            
    def _verify_checksum(self, path: Path, expected_checksum: str) -> bool:
        """Verify the checksum of a downloaded model."""
        # Implement checksum verification (e.g., SHA-256) if needed.
        # For now, we simply log that it's not implemented and return True.
        logger.info(f"Checksum verification for {path} skipped (not implemented)")
        return True


class ModelDeployer:
    """Handles deploying models as local services or API clients."""
    
    def __init__(self, models_directory: str = "./models", deployments_directory: str = "./deployments"):
        self.models_directory = Path(models_directory)
        self.deployments_directory = Path(deployments_directory)
        self.deployments_directory.mkdir(parents=True, exist_ok=True)
        self.downloader = ModelDownloader(models_directory)
        self.running_deployments = {}
    
    def deploy_model(self, model_info: Dict, api_keys: Optional[Dict] = None, 
                    port: int = 8000, host: str = "127.0.0.1") -> Dict:
        """Deploy the selected model."""
        model_name = model_info["name"]
        model_specs = model_info["specs"]
        
        result = {
            "model": model_name,
            "status": "pending",
            "deployment_id": f"{model_name}-{int(time.time())}"
        }
        
        try:
            if model_specs["type"] == "open_source":
                # Download the model if needed
                model_path = self.downloader.download_model(model_name, model_specs)
                if not model_path:
                    raise Exception(f"Failed to download or locate model {model_name}")
                
                # Deploy the model
                deployment_result = self._deploy_local_model(
                    model_name, model_specs, model_path, 
                    port=port, host=host, 
                    quantization=model_info.get("quantization")
                )
                
                result.update(deployment_result)
                
            else:  # API model
                # Set up API client
                api_key = api_keys.get(model_name.split("-")[0]) if api_keys else None
                if not api_key and model_specs.get("api_key_required", False):
                    raise Exception(f"API key for {model_name} not provided")
                
                deployment_result = self._deploy_api_client(
                    model_name, model_specs, api_key, 
                    port=port, host=host
                )
                
                result.update(deployment_result)
        
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
    
    def _deploy_local_model(self, model_name: str, model_specs: Dict, model_path: str, 
                            port: int = 8000, host: str = "127.0.0.1", 
                            quantization: Optional[str] = None) -> Dict:
        """Deploy a local model as a service."""
        logger.info(f"Deploying local model {model_name} from {model_path}")
        
        # Determine deployment type and prepare command
        framework = model_specs.get("framework", "llama.cpp")
        result = {
            "model_path": model_path,
            "host": host,
            "port": port,
            "framework": framework
        }
        
        try:
            # Create a deployment directory for this instance
            deployment_id = f"{model_name}-{int(time.time())}"
            deployment_dir = self.deployments_directory / deployment_id
            deployment_dir.mkdir(parents=True, exist_ok=True)
            result["deployment_id"] = deployment_id
            result["deployment_dir"] = str(deployment_dir)
            
            # Generate deployment script and config based on framework
            if framework == "llama.cpp":
                cmd, interface = self._prepare_llamacpp_deployment(
                    model_name, model_path, deployment_dir, port, host, model_specs, quantization
                )
            elif framework == "transformers":
                cmd, interface = self._prepare_transformers_deployment(
                    model_name, model_path, deployment_dir, port, host, model_specs
                )
            elif framework == "vllm":
                cmd, interface = self._prepare_vllm_deployment(
                    model_name, model_path, deployment_dir, port, host, model_specs, quantization
                )
            else:
                raise ValueError(f"Unsupported framework: {framework}")
                
            # Start the process
            logger.info(f"Starting model server with command: {cmd}")
            
            # Create log files
            log_file = deployment_dir / "deployment.log"
            err_file = deployment_dir / "deployment.err"
            
            with open(log_file, "w") as log, open(err_file, "w") as err:
                process = subprocess.Popen(
                    cmd, 
                    shell=True,
                    stdout=log,
                    stderr=err,
                    cwd=str(deployment_dir)
                    
                )
            
            # Store process info
            self.running_deployments[deployment_id] = {
                "process": process,
                "cmd": cmd,
                "model": model_name,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "port": port,
                "host": host
            }
            
            # Wait for server to start (simple implementation)
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is not None:
                # Process terminated
                with open(err_file, "r") as f:
                    error_log = f.read()
                raise Exception(f"Server process terminated unexpectedly. Exit code: {process.returncode}. Error: {error_log}")
            
            result["status"] = "deployed"
            result["process_id"] = process.pid
            result["command"] = cmd
            result["interface"] = interface
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to deploy local model: {str(e)}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result
    
    def _prepare_llamacpp_deployment(self, model_name: str, model_path: str, 
                                     deployment_dir: Path, port: int, host: str, 
                                     model_specs: Dict, quantization: Optional[str] = None) -> Tuple[str, str]:
        """Prepare deployment for llama.cpp models."""
        # Check if model path is a directory and find the model file
        model_path_obj = Path(model_path)
        if model_path_obj.is_dir():
            # Search for model files with common extensions
            model_files = list(model_path_obj.glob("*.bin")) + list(model_path_obj.glob("*.gguf"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in {model_path}")
            model_file = str(model_files[0])
        else:
            model_file = model_path
            
        # Determine context length
        ctx_len = model_specs.get("ctx_len", 4096)
        
        # Prepare command
        cmd_parts = [
            "llama-cpp-python",
            f"--model {model_file}",
            f"--ctx_len {ctx_len}",
            f"--host {host}",
            f"--port {port}",
            "--n_gpu_layers -1"  # Use all available GPU layers if possible
        ]
        
        # Add quantization if specified
        if quantization:
            if quantization == "4-bit":
                cmd_parts.append("--quant Q4_K_M")
            elif quantization == "8-bit":
                cmd_parts.append("--quant Q8_0")
        
        # Create a server configuration file
        config = {
            "model": model_file,
            "ctx_len": ctx_len,
            "host": host,
            "port": port,
            "n_gpu_layers": -1,
            "quantization": quantization
        }
        
        with open(deployment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create a shell script to start the server
        cmd = " ".join(cmd_parts)
        with open(deployment_dir / "start_server.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)
        
        # Make the script executable
        os.chmod(deployment_dir / "start_server.sh", 0o755)
        
        # Interface URL
        interface = f"http://{host}:{port}/v1"
        
        return cmd, interface
    
    def _prepare_transformers_deployment(self, model_name: str, model_path: str, 
                                         deployment_dir: Path, port: int, host: str, 
                                         model_specs: Dict) -> Tuple[str, str]:
        """Prepare deployment for Transformers models using text-generation-server."""
        cmd_parts = [
            "text-generation-launcher",
            f"--model-id {model_path}",
            f"--port {port}",
            f"--host {host}",
            "--trust-remote-code"
        ]
        
        # Add quantization if needed
        if model_specs.get("quantize", False):
            cmd_parts.append("--quantize bitsandbytes")
        
        # Create a server configuration file
        config = {
            "model_id": model_path,
            "port": port,
            "host": host
        }
        
        with open(deployment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create a shell script to start the server
        cmd = " ".join(cmd_parts)
        with open(deployment_dir / "start_server.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)
        
        # Make the script executable
        os.chmod(deployment_dir / "start_server.sh", 0o755)
        
        # Interface URL
        interface = f"http://{host}:{port}/generate"
        
        return cmd, interface
    
    def _prepare_vllm_deployment(self, model_name: str, model_path: str, 
                                 deployment_dir: Path, port: int, host: str, 
                                 model_specs: Dict, quantization: Optional[str] = None) -> Tuple[str, str]:
        """Prepare deployment for vLLM models."""
        # Prepare command
        cmd_parts = [
            "python -m vllm.entrypoints.api_server",
            f"--model {model_path}",
            f"--port {port}",
            f"--host {host}",
            "--trust-remote-code"
        ]
        
        # Detect how many GPUs are available via torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                cmd_parts.append(f"--tensor-parallel-size {gpu_count}")
        
        # Add quantization if needed
        if quantization == "4-bit":
            cmd_parts.append("--quantization awq")
        elif quantization == "8-bit":
            cmd_parts.append("--quantization int8")
        
        # Create a server configuration file
        config = {
            "model": model_path,
            "port": port,
            "host": host,
            "tensor_parallel_size": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "quantization": quantization
        }
        
        with open(deployment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create a shell script to start the server
        cmd = " ".join(cmd_parts)
        with open(deployment_dir / "start_server.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(cmd)
        
        # Make the script executable
        os.chmod(deployment_dir / "start_server.sh", 0o755)
        
        # Interface URL
        interface = f"http://{host}:{port}/v1"
        
        return cmd, interface
    
    def _deploy_api_client(self, model_name: str, model_specs: Dict, api_key: Optional[str], 
                           port: int = 8000, host: str = "127.0.0.1") -> Dict:
        """Set up an API client for remote LLM services."""
        logger.info(f"Setting up API client for {model_name}")
        
        # Create deployment directory
        deployment_id = f"{model_name}-{int(time.time())}"
        deployment_dir = self.deployments_directory / deployment_id
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "deployment_id": deployment_id,
            "deployment_dir": str(deployment_dir),
            "host": host,
            "port": port
        }
        
        try:
            # Determine API type and prepare configuration
            api_type = model_name.split("-")[0].lower()
            
            # Create API configuration file
            config = {
                "api_endpoint": model_specs["api_endpoint"],
                "model": model_name,
                "port": port,
                "host": host
            }
            
            if api_key:
                # Store API key with minimal exposure
                config["api_key_file"] = str("api_key.txt")
                with open(deployment_dir / "api_key.txt", "w") as f:
                    f.write(api_key)
                os.chmod(deployment_dir / "api_key.txt", 0o600)  # Restrict permissions
            
            # Write config (without the actual API key)
            with open(deployment_dir / "config.json", "w") as f:
                safe_config = config.copy()
                if "api_key_file" in safe_config:
                    safe_config["api_key"] = "********"  # Don't write actual key to config
                json.dump(safe_config, f, indent=2)
            
            # Determine proxy server type and create script
            if api_type == "claude" or api_type == "anthropic":
                cmd, interface = self._setup_claude_proxy(deployment_dir, config, port, host)
            elif api_type == "gpt" or api_type == "openai":
                cmd, interface = self._setup_openai_proxy(deployment_dir, config, port, host)
            else:
                raise ValueError(f"Unsupported API type: {api_type}")
            
            # Start the proxy server
            logger.info(f"Starting API proxy with command: {cmd}")
            
            # Create log files
            log_file = deployment_dir / "deployment.log"
            err_file = deployment_dir / "deployment.err"
            
            with open(log_file, "w") as log, open(err_file, "w") as err:
                process = subprocess.Popen(
                    cmd, 
                    shell=True,
                    stdout=log,
                    stderr=err,
                    cwd=str(deployment_dir)
                )
            
            # Store process info
            self.running_deployments[deployment_id] = {
                "process": process,
                "cmd": cmd,
                "model": model_name,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "port": port,
                "host": host
            }
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is not None:
                # Process terminated
                with open(err_file, "r") as f:
                    error_log = f.read()
                raise Exception(f"Proxy process terminated unexpectedly. Exit code: {process.returncode}. Error: {error_log}")
            
            result["status"] = "configured"
            result["process_id"] = process.pid
            result["command"] = cmd
            result["interface"] = interface
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to set up API client: {str(e)}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result
    
    def _setup_claude_proxy(self, deployment_dir: Path, config: Dict, port: int, host: str) -> Tuple[str, str]:
        """Set up a local proxy for Claude API."""
        proxy_file = deployment_dir / "claude_proxy.py"
        
        with open(proxy_file, "w") as f:
            f.write("""
from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Get API key
api_key = None
if 'api_key_file' in config:
    with open(config['api_key_file'], 'r') as f:
        api_key = f.read().strip()

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if not api_key:
        return jsonify({"error": "API key not configured"}), 401
        
    data = request.json
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Transform to Claude's expected payload (simple pass-through in many cases)
    claude_data = {}
    if 'model' in data:
        claude_data['model'] = data['model'].replace('gpt', 'claude')
    else:
        claude_data['model'] = 'claude-1'  # Fallback or default
    
    if 'messages' in data:
        claude_data['messages'] = data['messages']
        
    if 'temperature' in data:
        claude_data['temperature'] = data['temperature']
        
    if 'max_tokens' in data:
        claude_data['max_tokens'] = data['max_tokens']
    elif 'max_tokens_to_sample' in data:
        claude_data['max_tokens'] = data['max_tokens_to_sample']
    
    response = requests.post(
        config['api_endpoint'],
        headers=headers,
        json=claude_data
    )
    
    return response.json(), response.status_code

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model": config.get("model", "claude")})

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
""".replace("HOST", repr(host)).replace("PORT", str(port)))
        
        # Create start script
        with open(deployment_dir / "start_proxy.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("python claude_proxy.py")
        
        os.chmod(deployment_dir / "start_proxy.sh", 0o755)
        
        cmd = "./start_proxy.sh"
        interface = f"http://{host}:{port}/v1/chat/completions"
        
        return cmd, interface
    
    def _setup_openai_proxy(self, deployment_dir: Path, config: Dict, port: int, host: str) -> Tuple[str, str]:
        """Set up a local proxy for OpenAI API."""
        proxy_file = deployment_dir / "openai_proxy.py"
        
        with open(proxy_file, "w") as f:
            f.write("""
from flask import Flask, request, jsonify
import requests
import os
import json

app = Flask(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Get API key
api_key = None
if 'api_key_file' in config:
    with open(config['api_key_file'], 'r') as f:
        api_key = f.read().strip()

@app.route('/v1/<path:subpath>', methods=['GET','POST','PUT','DELETE'])
def proxy(subpath):
    if not api_key:
        return jsonify({"error": "API key not configured"}), 401
    
    url = f"{config['api_endpoint'].rstrip('/')}/{subpath}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": request.headers.get("Content-Type", "application/json")
    }
    
    if request.method == 'GET':
        response = requests.get(url, headers=headers, params=request.args)
    elif request.method == 'POST':
        response = requests.post(url, headers=headers, json=request.json)
    elif request.method == 'PUT':
        response = requests.put(url, headers=headers, json=request.json)
    elif request.method == 'DELETE':
        response = requests.delete(url, headers=headers)
    else:
        return jsonify({"error": "Method not supported"}), 405
    
    return response.json(), response.status_code

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "model": config.get("model", "gpt-4")})

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
""".replace("HOST", repr(host)).replace("PORT", str(port)))
        
        # Create start script
        with open(deployment_dir / "start_proxy.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("python openai_proxy.py")
        
        os.chmod(deployment_dir / "start_proxy.sh", 0o755)
        
        cmd = "./start_proxy.sh"
        interface = f"http://{host}:{port}/v1"
        
        return cmd, interface
    
    def stop_deployment(self, deployment_id: str) -> Dict:
        """Stop a running model deployment."""
        logger.info(f"Stopping deployment: {deployment_id}")
        
        result = {
            "deployment_id": deployment_id,
            "status": "unknown"
        }
        
        if deployment_id not in self.running_deployments:
            result["status"] = "not_found"
            result["error"] = f"Deployment {deployment_id} not found"
            return result
        
        try:
            # Get process info
            deployment_info = self.running_deployments[deployment_id]
            process = deployment_info["process"]
            
            # Try graceful termination first
            process.terminate()
            
            # Wait for process to terminate
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if process doesn't terminate
                logger.warning(f"Process {process.pid} did not terminate gracefully, sending SIGKILL")
                process.kill()
            
            self.running_deployments[deployment_id]["status"] = "stopped"
            self.running_deployments[deployment_id]["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            result["status"] = "stopped"
            
            return result
            
        except Exception as e:
            logger.error(f"Error stopping deployment: {str(e)}")
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    def list_deployments(self) -> List[Dict]:
        """List all current deployments and their status."""
        deployments = []
        
        for deployment_id, info in self.running_deployments.items():
            process = info["process"]
            status = "running" if process.poll() is None else "stopped"
            
            deployment = {
                "deployment_id": deployment_id,
                "model": info["model"],
                "status": status,
                "started_at": info["started_at"],
                "port": info["port"],
                "host": info["host"],
                "process_id": process.pid
            }
            
            if "stopped_at" in info:
                deployment["stopped_at"] = info["stopped_at"]
                
            deployments.append(deployment)
        
        return deployments
    
    def get_deployment_info(self, deployment_id: str) -> Dict:
        """Get detailed information about a specific deployment."""
        if deployment_id not in self.running_deployments:
            return {"status": "not_found", "error": f"Deployment {deployment_id} not found"}
        
        info = self.running_deployments[deployment_id].copy()
        process = info.pop("process")  # Remove the actual process object
        
        info["status"] = "running" if process.poll() is None else "stopped"
        info["process_id"] = process.pid
        
        # Add resource usage information
        try:
            if process.poll() is None:  # Only if process is running
                proc = psutil.Process(process.pid)
                info["memory_usage"] = proc.memory_info().rss / (1024 * 1024)  # MB
                info["cpu_percent"] = proc.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {str(e)}")
        
        return info
    
    def restart_deployment(self, deployment_id: str) -> Dict:
        """Restart a deployed model."""
        logger.info(f"Restarting deployment: {deployment_id}")
        
        result = {
            "deployment_id": deployment_id,
            "status": "unknown"
        }
        
        if deployment_id not in self.running_deployments:
            result["status"] = "not_found"
            result["error"] = f"Deployment {deployment_id} not found"
            return result
        
        try:
            # Stop the deployment
            stop_result = self.stop_deployment(deployment_id)
            if stop_result["status"] != "stopped":
                raise Exception(f"Failed to stop deployment: {stop_result.get('error', 'Unknown error')}")
            
            # Get original deployment info
            deployment_info = self.running_deployments[deployment_id]
            deployment_dir = Path(deployment_info.get("deployment_dir", ""))
            
            if not deployment_dir.exists():
                raise Exception(f"Deployment directory {deployment_dir} not found")
            
            cmd = deployment_info["cmd"]
            
            log_file = deployment_dir / "deployment.log"
            err_file = deployment_dir / "deployment.err"
            
            with open(log_file, "a") as log, open(err_file, "a") as err:
                log.write(f"\n\n--- Restart at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                err.write(f"\n\n--- Restart at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                
                process = subprocess.Popen(
                    cmd, 
                    shell=True,
                    stdout=log,
                    stderr=err,
                    cwd=str(deployment_dir)
                )
            
            self.running_deployments[deployment_id]["process"] = process
            self.running_deployments[deployment_id]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.running_deployments[deployment_id].pop("stopped_at", None)
            
            time.sleep(5)
            
            if process.poll() is not None:
                with open(err_file, "r") as f:
                    error_log = f.read()
                raise Exception(f"Server process terminated unexpectedly. Exit code: {process.returncode}. Error: {error_log}")
            
            result["status"] = "restarted"
            result["process_id"] = process.pid
            
            return result
            
        except Exception as e:
            logger.error(f"Error restarting deployment: {str(e)}")
            result["status"] = "error"
            result["error"] = str(e)
            return result


class HardwareManager:
    """Handles hardware detection and resource management for model deployment."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        
    def _get_system_info(self) -> Dict:
        """Get basic system hardware information."""
        info = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total": psutil.virtual_memory().total / (1024 ** 3),  # GB
                "available": psutil.virtual_memory().available / (1024 ** 3)  # GB
            },
            "os": platform.system(),
            "python_version": platform.python_version()
        }
        return info
    
    def _get_gpu_info(self) -> List[Dict]:
        """Get GPU information (NVIDIA, AMD via ROCm, or Apple MPS) if available through PyTorch."""
        gpus = []
        
        # 1) Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # On Apple Silicon with MPS support, you won't get full memory details via PyTorch
            gpus.append({
                "name": "Apple MPS Device",
                "memory_total_gb": None,
                "memory_free_gb": None,
                "memory_allocated_gb": None
            })
            return gpus
        
        # 2) Check for CUDA or ROCm
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            for i in range(num_devices):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                
                # This should work for both NVIDIA (CUDA) and AMD (ROCm) builds:
                total_mem = props.total_memory / (1024 ** 3)
                allocated_mem = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved_mem = torch.cuda.memory_reserved(i) / (1024 ** 3)
                free_mem = total_mem - reserved_mem
                
                gpus.append({
                    "name": gpu_name,
                    "memory_total_gb": total_mem,
                    "memory_free_gb": free_mem,
                    "memory_allocated_gb": allocated_mem
                })
        
        return gpus
    
    def recommend_model_config(self, model_specs: Dict) -> Dict:
        """Recommend optimal configuration for a model based on available hardware."""
        recommendation = {
            "can_run": False,
            "quantization": None,
            "framework": None,
            "tensor_parallel": 1
        }
        
        # Get memory requirements from the model specs
        model_memory_req = model_specs.get("memory_requirements", {})
        min_memory = model_memory_req.get("min_gb", 8)  # Minimum memory required in GB
        recommended_memory = model_memory_req.get("recommended_gb", 16)  # Recommended memory in GB
        
        if self.gpu_info:
            # Sum up GPU memory
            total_gpu_memory = sum(g["memory_total_gb"] for g in self.gpu_info if g["memory_total_gb"] is not None)
            free_gpu_memory = sum(g["memory_free_gb"] for g in self.gpu_info if g["memory_free_gb"] is not None)
            num_gpus = len(self.gpu_info)
            
            logger.info(f"Detected {num_gpus} GPU(s). Total GPU memory: ~{total_gpu_memory:.2f} GB, Free GPU memory: ~{free_gpu_memory:.2f} GB")
            
            # If we have enough free GPU memory for recommended usage
            if free_gpu_memory >= recommended_memory:
                recommendation["can_run"] = True
                recommendation["quantization"] = None
                recommendation["tensor_parallel"] = min(num_gpus, 2)
                
                # Auto-select a good framework if not specified
                if model_specs.get("framework") == "auto":
                    # vLLM can be more memory efficient, but llama.cpp can be simpler
                    # You can decide your own logic; here's a simple heuristic:
                    if free_gpu_memory >= recommended_memory * 1.5:
                        recommendation["framework"] = "vllm"
                    else:
                        recommendation["framework"] = "llama.cpp"
                else:
                    recommendation["framework"] = model_specs.get("framework", "llama.cpp")
            
            # Try 8-bit quant if not enough memory
            elif free_gpu_memory >= min_memory:
                recommendation["can_run"] = True
                recommendation["quantization"] = "8-bit"
                recommendation["framework"] = model_specs.get("framework", "llama.cpp")
            
            # Try 4-bit quant if that's still too large
            elif free_gpu_memory >= min_memory * 0.5:
                recommendation["can_run"] = True
                recommendation["quantization"] = "4-bit"
                recommendation["framework"] = model_specs.get("framework", "llama.cpp")
            
            else:
                # Not enough GPU memory, fallback to CPU check
                logger.warning("Insufficient GPU memory detected, checking CPU memory.")
                return self._check_cpu_capabilities(model_specs, min_memory, recommended_memory)
            print(model_specs)
        else:
            logger.info("No GPU detected, using CPU fallback.")
            return self._check_cpu_capabilities(model_specs, min_memory, recommended_memory)
        
        return recommendation
    
    def _check_cpu_capabilities(self, model_specs: Dict, min_memory: float, recommended_memory: float) -> Dict:
        """Check if CPU can run the model."""
        recommendation = {
            "can_run": False,
            "quantization": None,
            "framework": None,
            "cpu_only": True
        }
        
        available_memory = self.system_info["memory"]["available"]
        logger.info(f"Available CPU memory: {available_memory:.2f} GB")
        
        if available_memory >= recommended_memory:
            recommendation["can_run"] = True
            recommendation["quantization"] = None
            recommendation["framework"] = "llama.cpp"
        elif available_memory >= min_memory:
            recommendation["can_run"] = True
            recommendation["quantization"] = "8-bit"
            recommendation["framework"] = "llama.cpp"
        elif available_memory >= min_memory * 0.5:
            recommendation["can_run"] = True
            recommendation["quantization"] = "4-bit"
            recommendation["framework"] = "llama.cpp"
        else:
            recommendation["can_run"] = False
            recommendation["error"] = (
                f"Insufficient memory. Model requires ~{min_memory} GB but only "
                f"{available_memory:.2f} GB is available."
            )
        
        return recommendation
    
    def get_optimal_batch_size(self, model_specs: Dict, deployment_config: Dict) -> int:
        """Determine an approximate batch size based on hardware and model."""
        if deployment_config.get("quantization"):
            return 4  # more conservative for quantized models
        
        if self.gpu_info:
            # Higher batch size for GPU
            return 8
        else:
            # CPU fallback
            return 1
    
    def get_system_status(self) -> Dict:
        """Get current system resource usage."""
        status = {
            "cpu_usage": psutil.cpu_percent(interval=0.1, percpu=True),
            "memory_usage": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 ** 3),
                "total_gb": psutil.virtual_memory().total / (1024 ** 3)
            },
            "disk_usage": {
                "percent": psutil.disk_usage('/').percent,
                "used_gb": psutil.disk_usage('/').used / (1024 ** 3),
                "total_gb": psutil.disk_usage('/').total / (1024 ** 3)
            }
        }
        
        # Refresh GPU info (in case of dynamic changes)
        status["gpu"] = self._get_gpu_info()
        return status


class DeploymentManager:
    """Main class for managing model deployments and configurations."""
    
    def __init__(self, config_file: str = "./config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.hardware_manager = HardwareManager()
        self.model_deployer = ModelDeployer(
            models_directory=self.config.get("models_directory", "./models"),
            deployments_directory=self.config.get("deployments_directory", "./deployments")
        )
        self.shutdown_event = threading.Event()
        
    def _load_config(self) -> Dict:
        """Load or create the main configuration file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {str(e)}. Creating default config.")
        
        # Default configuration
        default_config = {
            "models_directory": "./models",
            "deployments_directory": "./deployments",
            "default_host": "127.0.0.1",
            "starting_port": 8000,
            "port_range": 1000,
            "api_keys": {},
            "models": {
                "open_source": [
                    {
        "name": "llama-3-8b",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "meta-llama/Meta-Llama-3-8B",
          "framework": "auto",
          "memory_requirements": {
            "min_gb": 8,
            "recommended_gb": 16
          }
        }
      },
      {
        "name": "mistral-7b",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "mistralai/Mistral-7B-v0.1",
          "framework": "llama.cpp",
          "memory_requirements": {
            "min_gb": 8,
            "recommended_gb": 16
          }
        }
      },
      {
        "name": "tiny-llama-1.1b",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          "framework": "llama.cpp",
          "memory_requirements": {
            "min_gb": 2,
            "recommended_gb": 4
          }
        }
      },
      {
        "name": "phi-2",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "microsoft/phi-2",
          "framework": "transformers",
          "memory_requirements": {
            "min_gb": 3,
            "recommended_gb": 6
          }
        }
      },
      {
        "name": "gemma-2b",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "google/gemma-2b",
          "framework": "transformers",
          "memory_requirements": {
            "min_gb": 2,
            "recommended_gb": 4
          }
        }
      },
      {
        "name": "stablelm-zephyr-3b",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "stabilityai/stablelm-zephyr-3b",
          "framework": "transformers",
          "memory_requirements": {
            "min_gb": 3,
            "recommended_gb": 6
          }
        }
      },
      {
        "name": "bloom-560m",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "bigscience/bloom-560m",
          "framework": "transformers",
          "memory_requirements": {
            "min_gb": 1,
            "recommended_gb": 2
          }
        }
      },
      {
        "name": "distilgpt2",
        "specs": {
          "type": "open_source",
          "source": "huggingface",
          "model_id": "distilgpt2",
          "framework": "transformers",
          "memory_requirements": {
            "min_gb": 1,
            "recommended_gb": 2
          }
        }
      }
                ],
                "api": [
                    {
                        "name": "claude-3-opus",
                        "specs": {
                            "type": "api",
                            "api_endpoint": "https://api.anthropic.com/v1/messages",
                            "api_key_required": True
                        }
                    }
                ]
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def save_config(self) -> None:
        """Save the current configuration to disk."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def add_api_key(self, provider: str, api_key: str) -> None:
        """Add or update an API key for a provider."""
        self.config.setdefault("api_keys", {})[provider] = api_key
        self.save_config()
        logger.info(f"API key for {provider} updated")
    
    def get_available_port(self) -> int:
        """Find an available port for deployment."""
        start_port = self.config.get("starting_port", 8000)
        port_range = self.config.get("port_range", 1000)
        
        used_ports = {d["port"] for d in self.model_deployer.list_deployments()}
        for port in range(start_port, start_port + port_range):
            if port not in used_ports:
                return port
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port+port_range}")
    
    def list_available_models(self) -> Dict:
        """List all available models from configuration, with hardware recommendations."""
        result = {
            "open_source": self.config.get("models", {}).get("open_source", []),
            "api": self.config.get("models", {}).get("api", [])
        }
        
        for model in result["open_source"]:
            model["hardware_recommendation"] = self.hardware_manager.recommend_model_config(model["specs"])
        
        return result
    
    def deploy_model(self, model_name: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict:
        """Deploy a model by name from the config."""
        # Search in open_source
        model_info = None
        for m in self.config.get("models", {}).get("open_source", []):
            if m["name"] == model_name:
                model_info = m
                break
        
        # Or search in API
        if not model_info:
            for m in self.config.get("models", {}).get("api", []):
                if m["name"] == model_name:
                    model_info = m
                    break
        
        if not model_info:
            return {"status": "error", "error": f"Model {model_name} not found in config"}
        
        if host is None:
            host = self.config.get("default_host", "127.0.0.1")
        if port is None:
            port = self.get_available_port()
        
        # If open_source, apply hardware recommendations
        if model_info["specs"]["type"] == "open_source":
            recommendation = self.hardware_manager.recommend_model_config(model_info["specs"])
            if not recommendation["can_run"]:
                return {"status": "error", "error": f"Hardware insufficient. {recommendation.get('error', '')}"}
            
            model_info["quantization"] = recommendation["quantization"]
            if recommendation.get("framework"):
                model_info["specs"]["framework"] = recommendation["framework"]
        
        api_keys = self.config.get("api_keys", {})
        result = self.model_deployer.deploy_model(
            model_info=model_info,
            api_keys=api_keys,
            port=port,
            host=host
        )
        
        return result
    
    def stop_all_deployments(self) -> Dict:
        """Stop all running deployments."""
        results = {}
        for dep in self.model_deployer.list_deployments():
            dep_id = dep["deployment_id"]
            results[dep_id] = self.model_deployer.stop_deployment(dep_id)
        return results
    
    def start_monitoring(self, interval: int = 60) -> None:
        """Start background monitoring of system & deployments."""
        def monitor_loop():
            while not self.shutdown_event.is_set():
                try:
                    system_status = self.hardware_manager.get_system_status()
                    deployments = self.model_deployer.list_deployments()
                    
                    for d in deployments:
                        if d["status"] != "running":
                            logger.warning(f"Deployment {d['deployment_id']} is not running. Status: {d['status']}")
                    
                    mem_used = system_status["memory_usage"]["used_gb"]
                    mem_total = system_status["memory_usage"]["total_gb"]
                    logger.info(f"System: CPU Usage={system_status['cpu_usage']}, "
                                f"Memory Used={mem_used:.2f} GB / {mem_total:.2f} GB")
                except Exception as e:
                    logger.error(f"Error in monitoring thread: {str(e)}")
                
                self.shutdown_event.wait(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.shutdown_event.set()
        logger.info("System monitoring stopped")


# Simple CLI for demonstration/testing
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Deployment Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # list
    list_parser = subparsers.add_parser("list", help="List models or deployments")
    list_parser.add_argument("--type", choices=["models", "deployments"], required=True)
    
    # deploy
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model from config")
    deploy_parser.add_argument("--model", required=True)
    deploy_parser.add_argument("--host", default=None)
    deploy_parser.add_argument("--port", type=int, default=None)
    
    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop a deployment")
    stop_parser.add_argument("--id", required=True)
    
    # add-key
    key_parser = subparsers.add_parser("add-key", help="Add or update an API key")
    key_parser.add_argument("--provider", required=True)
    key_parser.add_argument("--key", required=True)
    
    # system-info
    subparsers.add_parser("system-info", help="Show hardware info")
    
    args = parser.parse_args()
    
    manager = DeploymentManager()
    
    if args.command == "list":
        if args.type == "models":
            models = manager.list_available_models()
            print("Open Source Models:")
            for m in models["open_source"]:
                print(f"  - {m['name']}")
                hr = m.get("hardware_recommendation", {})
                can_run = "Yes" if hr.get("can_run") else "No"
                print(f"     Can run? {can_run}, recommended quant: {hr.get('quantization')}, framework: {hr.get('framework')}")
            print("\nAPI Models:")
            for m in models["api"]:
                print(f"  - {m['name']} (endpoint: {m['specs'].get('api_endpoint')})")
        
        elif args.type == "deployments":
            deps = manager.model_deployer.list_deployments()
            if not deps:
                print("No active deployments.")
            else:
                for d in deps:
                    print(f"{d['deployment_id']}: {d['model']} at {d['host']}:{d['port']} - {d['status']}")
    
    elif args.command == "deploy":
        result = manager.deploy_model(model_name=args.model, host=args.host, port=args.port)
        if result.get("status") in ["deployed", "configured"]:
            print(f"Deployment succeeded! ID: {result['deployment_id']}")
            print(f"Interface URL: {result.get('interface')}")
        else:
            print(f"Deployment failed: {result.get('error')}")
    
    elif args.command == "stop":
        r = manager.model_deployer.stop_deployment(args.id)
        if r["status"] == "stopped":
            print(f"Deployment {args.id} stopped.")
        else:
            print(f"Failed to stop deployment: {r.get('error')}")
    
    elif args.command == "add-key":
        manager.add_api_key(args.provider, args.key)
        print(f"API key for {args.provider} set.")
    
    elif args.command == "system-info":
        info = manager.hardware_manager.system_info
        print("System Info:")
        print(f"  OS: {info['os']}")
        print(f"  CPU Cores: {info['cpu']['cores']}, Threads: {info['cpu']['threads']}")
        print(f"  Memory: {info['memory']['total']:.2f} GB total, {info['memory']['available']:.2f} GB available")
        if manager.hardware_manager.gpu_info:
            print("\nDetected GPUs:")
            for i, gpu in enumerate(manager.hardware_manager.gpu_info):
                print(f"  GPU {i + 1}: {gpu['name']}")
                if gpu["memory_total_gb"] is not None:
                    print(f"     Memory Total: {gpu['memory_total_gb']:.2f} GB, Free: {gpu['memory_free_gb']:.2f} GB")
        else:
            print("\nNo GPU detected.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
