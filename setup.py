from distutils.core import setup

setup(
    name='FlowStats',
    version='0.2',
    packages=['flowstats'],
    package_data={'': []},
    description='Flow Cytometry Standard Statistical Functions',
    requires=['numpy', 'scipy', 'dpmix', 'matplotlib'],
)
