from setuptools import setup, find_packages

setup(
    name="ldm",
    py_modules=["ldm"],
    version="1.0",
    description="",
    author="CompVis",
    packages=find_packages(),
    # install_requires=[
    #     "fairscale>=0.4.4"
    # ],
    include_package_data=True,
)