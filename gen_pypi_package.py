import os
from distutils.dir_util import copy_tree
import shutil

# Get the current working directory
cwd = os.getcwd()

# Create a new temporary directory for the package
if not os.path.exists("tmp"):
    os.makedirs("tmp")
  
# Copy the python files of the memory_gym package to the new directory
copy_tree("memory_gym", dst="tmp\\memory_gym")

# Optional: Add extra directories which should be included in the package  
pypi_folders = []

# Setup files to create the package
setup_files = ["setup.py", "LICENSE", "README.md"]

# Build the package directories
for folder in pypi_folders:
    copy_tree(folder, dst="tmp\\memory_gym\\" + folder)

for file in setup_files:
    shutil.copy(file, "tmp")

# Deletes all pycache folders
for dname, dirs, files in os.walk("tmp\\memory_gym"):
    if "__pycache__" in dname:
        shutil.rmtree(dname)

# Build the package
os.chdir("./tmp")
os.system("python -m build")
os.chdir(cwd)

# Copy the package to the current directory
copy_tree("tmp\\dist", "dist")

# Clean up
shutil.rmtree("tmp")

# Upload the package to PyPI
os.system("python -m twine upload dist/*")

# Clean up
shutil.rmtree("dist")