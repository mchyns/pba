#!/bin/bash

# Git repository setup for Streamlit Cloud deployment
echo "🚀 Setting up Git repository for deployment..."

# Initialize git if not already initialized
if [ ! -d .git ]; then
    git init
    echo "✅ Git repository initialized"
fi

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/

# MacOS
.DS_Store

# Temporary files
*.tmp
*.log

# Don't ignore model files - they're needed for deployment
!*.pkl
!*.csv
!*.json
EOF

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Sentiment Analysis App for Hospital Reviews

Features:
- Naive Bayes sentiment classification
- TF-IDF vectorization
- Sastrawi stemming for Indonesian text
- Negation and context handling
- Mixed sentiment detection
- Clean Streamlit UI
- Ready for production deployment"

echo "✅ Git repository ready for deployment!"
echo ""
echo "Next steps for Streamlit Cloud deployment:"
echo "1. Create GitHub repository"
echo "2. Add remote: git remote add origin https://github.com/yourusername/repo-name.git"
echo "3. Push code: git push -u origin main"
echo "4. Deploy on share.streamlit.io"
echo ""
echo "Your app is ready! 🎉"
