#!/usr/bin/env python

import os
from distutils.core import setup, Extension
from subprocess import Popen, PIPE

this_dir = os.path.dirname(os.path.realpath(__file__))

cv2gpumodule = Extension('cv2gpu', 
                  define_macros = [('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                  sources = [os.path.join('src', 'face_detector.cpp'), os.path.join('src', 'cv2gpu.cpp')],
                  extra_compile_args = ['-std=c++11'])

setup (name = 'cv2gpu',
       version = '1.0',
       description = 'OpenCV GPU Bindings',
       author = 'Alexander Koumis and Matthew Carlis',
       author_email = 'alexander.koumis@sjsu.edu, matthew.carlis@sjsu.edu',
       url = 'https://docs.python.org/extending/building',
       long_description = '''
OpenCV GPU Bindings
''',
       ext_modules = [cv2gpumodule])
