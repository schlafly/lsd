#!/usr/bin/env python

import os, os.path

def suppress_keyboard_interrupt_message():
	old_excepthook = sys.excepthook

	def new_hook(type, value, traceback):
		if type != exceptions.KeyboardInterrupt:
			old_excepthook(type, value, traceback)
		else:
			pass

	sys.excepthook = new_hook


# Use NUMPY_INCLUDE environment variable to set where to find NumPy
numpy_include=os.getenv('NUMPY_INCLUDE', '/opt/python2.7/lib/python2.7/site-packages/numpy/core/include/')
if not os.path.isfile(numpy_include + '/numpy/arrayobject.h'):
	print >> sys.stderr, "Failed to find " . numpy_include + '/numpy/arrayobject.h'
	print >> sys.stderr, "Error: could not find arrayobject.h. Please set the NumPy include path using NUMPY_INCLUDE environment variable"
	exit(-1)

# ------ no changes below! If you need to change, it's a bug! -------
from distutils.core import setup, Extension
from sys import platform

import numpy
inc = [numpy_include]

longdesc = """Large Survey Database"""

args = { 
	'name'			: "skysurvey",
	'version'		: "0.1",
	'description'	 	: "Large Survey Database Python Module",
	'long_description'	: longdesc,
	'license'		: "GPLv2",
	'author'		: "Mario Juric",
	'author_email'		: "mjuric@cfa.harvard.edu",
	'maintainer'		: "Mario Juric",
	'maintainer_email'	: "mjuric@cfa.harvard.edu",
	'url'			: "http://mwscience.net/lsd",
	'download_url'		: "http://mwscience.net/lsd/download",
	'classifiers'		: [
					'Development Status :: 3 - Alpha',
					'Intended Audience :: Science/Research', 
					'Intended Audience :: Developers',
					'License :: OSI Approved :: GNU General Public License (GPL)', 
					'Programming Language :: C++', 
					'Programming Language :: Python :: 2', 
					'Programming Language :: Python :: 2.7',
					'Operating System :: POSIX :: Linux',
					'Topic :: Database',
					'Topic :: Scientific/Engineering :: Astronomy'
	],
	'scripts'	: ['src/lsd-footprint', 'src/lsd-import-sdss', 'src/lsd-make-object-catalog',
	 			'src/lsd-import-dvo', 'src/lsd-import-smf',
	 			'src/lsd-query', 'src/lsd-xmatch'],
	'packages'	: ['skysurvey'],
	'package_dir'	: {'': 'src'},
	'ext_modules'	: [Extension('skysurvey.native', ['src/native/main.cpp'], include_dirs=inc)]
}

setup(**args)