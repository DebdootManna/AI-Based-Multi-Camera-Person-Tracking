from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi-camera-tracking",
    version="0.1.0",
    author="Debdoot Manna",
    author_email="debdootmanna@example.com",
    description="Multi-Camera Person Tracking System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DebdootManna/multi-camera-tracking",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.6.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "filterpy>=1.4.5",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
        "lap>=0.4.0",
        "PyYAML>=6.0",
        "pillow>=9.0.0",
        "thop>=0.1.0",
        "seaborn>=0.11.0",
        "av>=9.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "track=multi_camera_tracking.main:main",
        ],
    },
)
