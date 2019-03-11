import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#==============================================================================
# Utilities
#==============================================================================

def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def get_install_requirements(path):
    content = read(path)
    return [
        req
        for req in content.split("\n")
        if req != '' and not req.startswith('#')
    ]


def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py

    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

HERE = os.path.abspath(os.path.dirname(__file__))

# From https://github.com/jupyterlab/jupyterlab/blob/master/setupbase.py, BSD licensed
def find_packages(top=HERE):
    """
    Find all of the packages.
    """
    packages = []
    for d, dirs, _ in os.walk(top, followlinks=True):
        if os.path.exists(os.path.join(d, '__init__.py')):
            packages.append(os.path.relpath(d, top).replace(os.path.sep, '.'))
        elif d != top:
            # Do not look for packages in subfolders if current is not a package
            dirs[:] = []
    return packages

#==============================================================================
# Variables
#==============================================================================


DESCRIPTION         = "A python engine for evaluating Altair transforms."
LONG_DESCRIPTION    = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
NAME                = "altair_transform"
PACKAGES            = find_packages()
AUTHOR              = "Jake VanderPlas"
AUTHOR_EMAIL        = "jakevdp@gmail.com"
URL                 = 'http://github.com/jakevdp/altair-transform/'
DOWNLOAD_URL        = 'http://github.com/jakevdp/altair-transform/'
LICENSE             = 'MIT'
INSTALL_REQUIRES    = get_install_requirements("requirements.txt")
PYTHON_REQUIRES     = ">=3.7"
VERSION             = version('altair_transform/__init__.py')


setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=PACKAGES,
      include_package_data=True,
      install_requires=INSTALL_REQUIRES,
      python_requires=PYTHON_REQUIRES,
      classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7'],
     )
