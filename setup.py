from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

long_description = (ROOT / "README.md").read_text(encoding="utf-8")
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]


setup(
    name="loc-gs",
    version="0.1.0",
    description="Localization-oriented Gaussian feature fields for accurate camera relocalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["output", "dataset", "reference", "third_party*"]),
    include_package_data=True,
    package_data={"loc_gs": ["configs/*.yaml"]},
    python_requires=">=3.9",
    install_requires=requirements,
)
