from setuptools import setup

setup(name='gym_fastag',
      version='0.0.2',
      description='Fast version of AlphaGardenSim with interface for domain randomization. Env supports RLLib.',
      author='Sebastian Oehme',
      author_email='sebastian.oehme@tum.de',
      install_requires=['gym', 'numpy', 'scipy', 'pandas', 'matplotlib', 'pyyaml', 'ray[rllib]', 'tensorflow', 'torch']
)