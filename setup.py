from setuptools import setup

setup(
  
  name = "inferno",
  description='useful tools for inference',  
  version = "1.0",
  author='Neale Gibson',
  author_email='n.gibson@tcd.ie',
  python_requires='>=3',
  
  packages=['inferno'],
  package_dir={'inferno':'src'},
  install_requires=['numpy','scipy','tqdm','dill'],
  )
