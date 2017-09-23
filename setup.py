"""
Run setup
"""

from distutils.core import setup

setup(name='classificator',
      version='0.1.0',
      description='Text classification automation tool',
      url='https://github.com/denver1117/classificator',
      download_url='https://github.com/denver1117/classificator/archive/0.1.tar.gz',
      author='Evan Harris',
      author_email='emitchellh@gmail.com',
      license='MIT',
      packages=['classificator'],
      install_requires=[
          'pandas==0.18.0',
          'numpy==1.13.1',
          'scipy==0.17.0',
          'scikit-learn==0.18.2',
          'boto3==1.4.0'
      ],
      zip_safe=False)
