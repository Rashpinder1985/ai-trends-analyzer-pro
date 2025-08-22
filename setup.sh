#!/bin/bash

# AI Trends Analyzer Pro - Setup Script
# This script sets up the development environment

set -e

echo "🚀 Setting up AI Trends Analyzer Pro..."

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' || echo "0.0")
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data output logs config/env

# Copy environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "✏️ Please edit .env file with your configuration"
fi

# Initialize git hooks (if in git repo)
if [ -d ".git" ]; then
    echo "🪝 Setting up git hooks..."
    # Create pre-commit hook for code formatting
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run code formatting and linting before commit
echo "Running pre-commit checks..."

# Format code with black
black --check ai_trends/ tests/ || {
    echo "❌ Code formatting issues found. Run 'black ai_trends/ tests/' to fix."
    exit 1
}

# Sort imports with isort
isort --check-only ai_trends/ tests/ || {
    echo "❌ Import sorting issues found. Run 'isort ai_trends/ tests/' to fix."
    exit 1
}

# Run linting with flake8
flake8 ai_trends/ tests/ || {
    echo "❌ Linting issues found. Please fix before committing."
    exit 1
}

echo "✅ Pre-commit checks passed!"
EOF

    chmod +x .git/hooks/pre-commit
fi

# Run initial tests
echo "🧪 Running initial tests..."
python -m pytest tests/ -v --tb=short || {
    echo "⚠️ Some tests failed. This is expected for initial setup."
}

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Docker is available for containerized deployment"
    echo "💡 Run 'docker-compose up' to start the full stack"
else
    echo "⚠️ Docker not found. Install Docker for containerized deployment"
fi

# Check if Redis is available
if command -v redis-cli &> /dev/null; then
    echo "🔴 Redis CLI available for cache management"
else
    echo "⚠️ Redis not found. Install Redis for caching (optional)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run 'source venv/bin/activate' to activate the environment"
echo "3. Run 'python -m ai_trends.api.main' to start the API server"
echo "4. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "🔧 Development commands:"
echo "- Run tests: pytest tests/"
echo "- Format code: black ai_trends/ tests/"
echo "- Sort imports: isort ai_trends/ tests/"
echo "- Lint code: flake8 ai_trends/ tests/"
echo "- Type check: mypy ai_trends/"
echo ""
echo "📚 Documentation: Check README.md for detailed usage"