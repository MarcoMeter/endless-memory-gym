from setuptools import setup, find_packages
import os
import sys
sys.path.insert(0, os.getcwd())

# Get current working directory
cwd = os.getcwd()

# Get long description from README.md
long_description = ""
with open(cwd + "//README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
      name="memory-gym",
      description="A gym that contains the memory benchmarks Mortar Mayhem, Mystery Maze and Searing Spotlights",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/MarcoMeter/drl-memory-gym",
      keywords = ["Deep Reinforcement Learning", "gym", "POMDP", "Imperfect Information", "Partial Observation"],
      project_urls={
            "Github": "https://github.com/MarcoMeter/drl-memory-gym",
            "Bug Tracker": "https://github.com/MarcoMeter/drl-memory-gym/issues"
      },
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      author="Marco Pleines",
      package_dir={'': '.'},
      packages=find_packages(where='.', include="memory_gym*"),
      python_requires=">=3.8",
      include_package_data=True,
      install_requires=["gymnasium==0.29.0",
                        "pygame==2.4.0"],
      entry_points={
            "console_scripts": [
            "searing_spotlights=memory_gym.searing_spotlights:main",
            "gt_searing_spotlights=memory_gym.searing_spotlights_gt:main",
            "endless_searing_spotlights=memory_gym.endless_searing_spotlights:main",
            "mortar_mayhem=memory_gym.mortar_mayhem:main",
            "endless_mortar_mayhem=memory_gym.endless_mortar_mayhem:main",
            "mortar_mayhem_b=memory_gym.mortar_mayhem_b:main",
            "mortar_mayhem_grid=memory_gym.mortar_mayhem_grid:main",
            "mortar_mayhem_b_grid=memory_gym.mortar_mayhem_b_grid:main",
            "mystery_path=memory_gym.mystery_path:main",
            "endless_mystery_path=memory_gym.endless_mystery_path:main",
            "mystery_path_grid=memory_gym.mystery_path_grid:main",
            ],
      },
      version="1.0.3",
)