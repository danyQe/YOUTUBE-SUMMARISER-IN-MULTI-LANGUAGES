# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
# Ask user to start the application
$user_input = Read-Host "Do you want to start the application? (yes/no)"

# If user input is 'yes', then start the application
if ($user_input -eq "yes") {
    python main.py
}