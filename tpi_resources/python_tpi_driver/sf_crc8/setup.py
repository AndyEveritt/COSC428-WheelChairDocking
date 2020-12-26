# For an in-place build:
# python3 setup.py build_ext --inplace


from distutils.core import setup, Extension

# define the extension module
crc8_module = Extension('crc8', sources=['pycrc8.c', 'src/SF_CRC8.c'],
                        include_dirs=['src'])

# run the setup
setup(ext_modules=[crc8_module])

