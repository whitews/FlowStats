from distutils.core import setup

setup(
    name='FlowStats',
    version='0.6',
    packages=['flowstats'],
    package_data={'': []},
    description='Flow Cytometry Standard Statistical Functions',
    requires=[
        'numpy (>=1.6)',
        'scipy',
        'dpmix_exp (==0.5a)',
        'matplotlib'
    ],
)
