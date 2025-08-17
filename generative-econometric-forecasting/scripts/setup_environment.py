#!/usr/bin/env python3
"""
🛠️ Environment Setup Script
Sets up API keys and environment configuration
"""

import os
from pathlib import Path

def setup_env_file():
    """Create .env file from template"""
    print("🔑 Setting up environment configuration...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("❌ .env.example not found")
        return False
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            return True
    
    # Copy template
    with open(env_example, 'r') as f:
        template = f.read()
    
    with open(env_file, 'w') as f:
        f.write(template)
    
    print("✅ .env file created from template")
    print("\n📝 Next steps:")
    print("   1. Edit .env file with your API keys")
    print("   2. Get FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("   3. Get OpenAI API key: https://platform.openai.com/api-keys")
    
    return True

if __name__ == "__main__":
    print("🛠️  ENVIRONMENT SETUP")
    print("=" * 30)
    setup_env_file()