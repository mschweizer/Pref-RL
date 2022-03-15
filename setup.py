from setuptools import setup, find_packages

setup(name='pref_rl',
      version='0.1',
      description='Provides ready-to-use preference-based reinforcement learning agents that are easily extensible.',
      url='https://github.com/mschweizer/Pref-RL',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'stable-baselines3[extra]',
          'torch',
          'gym[atari]<0.22',
          'numpy',
          'scipy',
      ],
      include_package_data=True,
      python_requires='>=3.9',
      )
