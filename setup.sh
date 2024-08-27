#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
STRIDE_DIR="${SCRIPT_DIR}/stride"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install build tools
install_build_tools() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install -y build-essential
        elif command_exists yum; then
            sudo yum groupinstall -y "Development Tools"
        else
            echo "Error: Unsupported package manager. Please install build tools manually."
            return 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command_exists gcc || ! command_exists make; then
            echo "Installing Xcode Command Line Tools..."
            xcode-select --install
            read -p "Press enter after Xcode Command Line Tools installation is complete"
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash or Cygwin)
        if ! command_exists gcc || ! command_exists make; then
            echo "Please install MinGW-w64 or Cygwin with gcc and make."
            echo "Visit: https://sourceforge.net/projects/mingw-w64/ or https://www.cygwin.com/"
            return 1
        fi
    else
        echo "Unsupported operating system"
        return 1
    fi
    return 0
}

# Function to compile stride
compile_stride() {
    echo "Starting stride compilation process..."

    # Decompress stride
    if [ -f "$STRIDE_DIR/stride.tgz" ]; then
        echo "Decompressing stride..."
        tar -zxf "$STRIDE_DIR/stride.tgz" -C "$STRIDE_DIR" || return 1
    else
        echo "Error: stride.tgz not found in $STRIDE_DIR"
        return 1
    fi

    # Install build tools if necessary
    echo "Checking and installing build tools if necessary..."
    install_build_tools || return 1

    # Compile stride
    echo "Compiling stride..."
    cd "$STRIDE_DIR" || return 1
    make || return 1

    # Change permissions
    echo "Changing stride permissions to executable..."
    chmod +x stride || return 1

    # Test run
    echo "Performing a test run of stride..."
    ./stride --help || return 1

    # Change back to the original directory
    cd - || return 1

    echo "Stride compilation successful!"
    return 0
}

# Main script execution starts here

# Check if venv directory chswEnv exists
if [ ! -d "$SCRIPT_DIR/chswEnv" ]; then
    echo "Creating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
       python -m venv "$SCRIPT_DIR/chswEnv"
    # Try Python 3.11, then 3.10, then 3.9
    elif command_exists python3.11; then
        python3.11 -m venv "$SCRIPT_DIR/chswEnv"
        echo "Created virtual environment with Python 3.11"
    elif command_exists python3.10; then
        python3.10 -m venv "$SCRIPT_DIR/chswEnv"
        echo "Created virtual environment with Python 3.10"
    elif command_exists python3.9; then
        python3.9 -m venv "$SCRIPT_DIR/chswEnv"
        echo "Created virtual environment with Python 3.9"
    elif command_exists python3.8; then
        python3.9 -m venv "$SCRIPT_DIR/chswEnv"
        echo "Created virtual environment with Python 3.8"
    else
        echo "Error: Python 3.8, 3.9, 3.10, or 3.11 is not installed. Please install one of these versions and try again."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source "$SCRIPT_DIR/chswEnv/Scripts/activate"
else
    source "$SCRIPT_DIR/chswEnv/bin/activate"
fi

echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing dependencies from requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Compile stride
if compile_stride; then
    echo "Stride compilation and setup completed successfully."
    echo "Chainsaw setup complete. Test the installation by running the following command:"
    echo "python get_predictions.py --structure_file example_files/AF-A0A1W2PQ64-F1-model_v4.pdb --output results/test.tsv"
else
    echo "Error occurred during stride compilation."
    echo "Please compile stride manually by following these steps:"
    echo "1. Navigate to the stride directory: cd $STRIDE_DIR"
    echo "2. Decompress the source: tar -zxf stride.tgz"
    echo "3. Compile the source: make"
    echo "4. Make the binary executable: chmod +x stride"
    echo "5. Test the binary: ./stride --help"
fi