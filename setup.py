from distutils.core import setup

setup(
    name='FlowStats',
    version='0.8',
    packages=['flowstats'],
    package_data={'': []},
    description='Flow Cytometry Standard Statistical Functions',
    requires=[
        'numpy (>=1.6)',
        'scipy',
        'dpmix_exp (==0.5)',
        'matplotlib'
    ],
)
