from setuptools import setup

from setuptools import setup, find_packages

setup(
    name="nodding",
    version="0.1.1",
    description="Detect nodding in a video.",
    author="Keyne Oei",
    author_email="keynekassapa13@gmail.com",
    packages=find_packages(include=["nodding"]),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
        "matplotlib",
        "scipy",
        "loguru",
        "python-box",
        "fastdtw"
    ],
    entry_points={
        "console_scripts": [
            "nodding = nodding.cli:main"
        ]
    },
    python_requires=">=3.8",
)

