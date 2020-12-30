from distutils.core import setup, Extension
import numpy.distutils.misc_util

pyjack = Extension("pyjack", ["_pyjack.c"])


setup(
name='pyjack',
      version='0.03',
      description='SPH Rendering C extension',
      author='Jack Humphries',
      author_email='rjh73@le.ac.uk',
      url='',
      ext_modules=[pyjack],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
