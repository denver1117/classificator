"""
Run setup
"""

from setuptools import setup, find_packages
from classificator import __version__

setup(name='classificator',
      version=__version__,
      description='Text classification automation tool',
      url='https://github.com/denver1117/classificator',
      download_url='https://pypi.org/project/classificator/#files',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pandas>=0.18.0',
          'numpy>=1.13.1',
          'scipy>=0.17.0',
          'scikit-learn>=0.21.0',
          'boto3>=1.4.0'
      ])
