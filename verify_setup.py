"""Verify that the ADK Ollama setup is correct."""

import os
import sys


def check_ollama_running():
    """Check if Ollama server is accessible."""
    import requests
    
    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        response = requests.get(f"{api_base}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return True, model_names
        return False, []
    except Exception as e:
        return False, str(e)


def check_model_available(model_name="qwen2.5:14b"):
    """Check if the specified model is available in Ollama."""
    import requests
    
    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        response = requests.get(f"{api_base}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check for exact match or partial match
            if model_name in model_names:
                return True, "exact match"
            # Check for partial matches (UPDATED for Qwen)
            for name in model_names:
                if "qwen" in name.lower():
                    return True, f"similar model found: {name}"
            return False, f"Model '{model_name}' not found. Available models: {', '.join(model_names)}"
    except Exception as e:
        return False, str(e)


def check_dependencies():
    """Check if required Python packages are installed."""
    missing = []
    try:
        import google.adk
    except ImportError:
        missing.append("google-adk")
    
    try:
        import litellm
    except ImportError:
        missing.append("litellm")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    return len(missing) == 0, missing


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("ADK Ollama Demo - Setup Verification")
    print("=" * 60)
    
    all_ok = True
    
    # Check dependencies
    print("\n1. Checking Python dependencies...")
    deps_ok, missing = check_dependencies()
    if deps_ok:
        print("   ✓ All dependencies installed")
    else:
        print(f"   ✗ Missing dependencies: {', '.join(missing)}")
        print(f"   Run: pip install -r requirements.txt")
        all_ok = False
    
    # Check environment variable
    print("\n2. Checking environment variables...")
    ollama_base = os.getenv("OLLAMA_API_BASE")
    if ollama_base:
        print(f"   ✓ OLLAMA_API_BASE is set: {ollama_base}")
    else:
        print("   ⚠ OLLAMA_API_BASE not set (using default: http://localhost:11434)")
        print("   You can set it with: export OLLAMA_API_BASE=http://localhost:11434")
    
    # Check Ollama server
    print("\n3. Checking Ollama server connection...")
    ollama_ok, result = check_ollama_running()
    if ollama_ok:
        print(f"   ✓ Ollama server is running")
        if isinstance(result, list) and result:
            print(f"   Available models: {', '.join(result[:5])}")
            if len(result) > 5:
                print(f"   ... and {len(result) - 5} more")
    else:
        print(f"   ✗ Cannot connect to Ollama server: {result}")
        print("   Make sure Ollama is running: ollama serve")
        all_ok = False
    
    # Check model availability
    if ollama_ok:
        print("\n4. Checking model availability...")
        model_ok, message = check_model_available("qwen2.5:14b")
        if model_ok:
            print(f"   ✓ Model found: {message}")
        else:
            print(f"   ✗ {message}")
            print("   Pull the model with: ollama pull qwen2.5:14b")
            all_ok = False
    
    # Check agent file
    print("\n5. Checking agent configuration...")
    if os.path.exists("src/agent.py"):
        print("   ✓ Agent file found: src/agent.py")
    else:
        print("   ✗ Agent file not found: src/agent.py")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! You're ready to use the agent.")
        print("\nTry running:")
        print("  adk web        # Start web interface")
        print("  adk run .      # Start CLI interface")
        print("  python example.py  # Run example script")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

