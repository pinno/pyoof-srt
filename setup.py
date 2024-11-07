import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyoof-srt",
    version="1.0",
    author="Andrea Pinna",
    author_email="andreapinna@gmail.com",
    description="Out-of-focus holography on astronomical beam maps for the Sardinia Radio Telescope",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pinno/pyoof-srt.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
    python_requires='>=3.9.0',
    include_package_data=True
)

