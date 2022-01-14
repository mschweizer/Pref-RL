from setuptools import setup

setup(name='pref_rl',
      version='0.1',
      description='Provides ready-to-use PbRL agents that are easily extensible.',
      url='https://github.com/mschweizer/Pref-RL',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=['pref_rl'],
      install_requires=[
          'stable-baselines3[extra]==1.3.0',
          'torch==1.10.0',
          'gym[atari]==0.19.0',
          'numpy==1.21.4',
          'scipy==1.7.2',
      ],
      include_package_data=True,
      python_requires='>=3.9',
      )
