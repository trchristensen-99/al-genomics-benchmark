#!/bin/bash
# Script to set up HashFrag for the project
#
# HashFrag creates homology-aware data splits by:
# 1. Using BLAST to find similar sequences
# 2. Computing Smith-Waterman alignment scores
# 3. Clustering homologous sequences together
# 4. Ensuring clusters don't span train/val/test splits

set -e  # Exit on any error

echo "=== Setting up HashFrag ==="
echo ""

# Check if BLAST+ is installed
if ! command -v blastn &> /dev/null; then
    echo "ERROR: BLAST+ not found!"
    echo ""
    echo "Please install BLAST+ from:"
    echo "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/"
    echo ""
    echo "On Ubuntu/Debian: sudo apt-get install ncbi-blast+"
    echo "On macOS: brew install blast"
    echo "On HPC: module load blast-plus (or similar)"
    exit 1
fi

echo "✓ BLAST+ found: $(blastn -version | head -1)"
echo ""

# Create external tools directory
mkdir -p external

# Clone HashFrag if not already present
if [ -d "external/hashFrag" ]; then
    echo "✓ HashFrag already cloned at external/hashFrag"
    
    # Check if it's up to date
    cd external/hashFrag
    echo "  Current commit: $(git rev-parse --short HEAD)"
    cd ../..
else
    echo "Cloning HashFrag repository..."
    git clone https://github.com/de-Boer-Lab/hashFrag.git external/hashFrag
    echo "✓ HashFrag cloned successfully"
fi

echo ""

# Check if HashFrag scripts are executable
if [ -f "external/hashFrag/src/hashFrag" ]; then
    chmod +x external/hashFrag/src/hashFrag
    echo "✓ Made HashFrag executable"
fi

# Add HashFrag to PATH for this session
export PATH="$PATH:$(pwd)/external/hashFrag/src"

# Test HashFrag installation
if command -v hashFrag &> /dev/null; then
    echo "✓ HashFrag is accessible in PATH"
else
    echo "⚠ HashFrag not in PATH yet (will work after exporting)"
fi

echo ""
echo "=== HashFrag Setup Complete ==="
echo ""
echo "To use HashFrag in your current shell, run:"
echo "  export PATH=\"\$PATH:$(pwd)/external/hashFrag/src\""
echo ""
echo "To make this permanent, add to ~/.bashrc or ~/.zshrc:"
echo "  echo 'export PATH=\"\$PATH:$(pwd)/external/hashFrag/src\"' >> ~/.bashrc"
echo ""
echo "Next steps:"
echo "  1. Test with: hashFrag --help"
echo "  2. Create K562 splits: python scripts/create_hashfrag_splits.py"
echo ""
