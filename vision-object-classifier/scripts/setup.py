from setuptools import setup, find_packages

setup(
    name="vision-object-classifier",
    version="1.0.0",
    description="Computer vision system for classifying household objects as clean or dirty",
    author="Vision Object Classifier Team",
    author_email="contact@vision-classifier.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "timm>=0.9.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0"
        ],
        "kaggle": [
            "kaggle>=1.5.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "dish-classifier-train=src.train:main",
            "dish-classifier-predict=src.predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)