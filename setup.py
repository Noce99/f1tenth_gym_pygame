from setuptools import setup

setup(name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      package_dir={'': 'gym'},
      install_requires=[
          'setuptools==56.0.0',
          'opencv-python==4.10.0.84',
          'cloudpickle==1.6.0',
          'future==1.0.0',
          'gym==0.19.0',
          'importlib_metadata==8.0.0',
          'llvmlite==0.41.1',
          'numba==0.58.1',
          'numpy==1.22.0',
          'pillow==10.3.0',
          'pygame==2.6.0',
          'pyglet==1.4.11',
          'PyOpenGL==3.1.7',
          'PyYAML==6.0.1',
          'scipy==1.10.1',
          'zipp==3.19.2'
      ]
      )
