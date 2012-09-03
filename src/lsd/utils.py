import subprocess, os, errno
import numpy as np
import contextlib

class NamedList(list):
	def __init__(self, *items):
		self.names = [ name for name, _ in items ]
		list.__init__(self, [ col for _, col in items ])

class Namespace:
	def __init__(self, **kwargs):
		for k, v in kwargs.iteritems():
			setattr(self, k, v)

class ModuleProxy(object):
	""" A proxy class used to wrap modules imported as UDFs
	    to make them pickleable.
	"""
	def __init__(self, obj):
		object.__setattr__(self, '_obj_', obj)

	## Forward non-pickle calls to the real module
	def __getattribute__(self, name):
		#print "GET:", name
		if name in ['__reduce_ex__', '__reduce__', '__getnewargs__', '__getstate__', '__class__', '__setstate__']:
			return object.__getattribute__(self, name)
		obj = object.__getattribute__(self, '_obj_')
		return getattr(obj, name)
	
	def __setattr__(self, name, value):
		obj = object.__getattribute__(self, '_obj_')
		return setattr(obj, name, value)

	def __delattr__(self, name):
		obj = object.__getattribute__(self, '_obj_')
		return delattr(obj, name)


	## Pickling support: just store the module name, for reconstruction later on
	def __getstate__(self):
		obj = object.__getattribute__(self, '_obj_')
		return obj.__name__,

	def __setstate__(self, state):
		name, = state
		__import__(name)
		object.__setattr__(self, '_obj_', sys.modules[name])

class LazyCreate(object):
	""" Lazy (on-demand) creation of objects.
	
	    This class allows the object to appear instantiated, while
	    in fact it will be instantiated on first use. This is useful
	    for "heavy" (== memory consuming) objects that may or may
	    not be called by the user -- if they're never called, they
	    are never created.
	    
	    Example usage:
	    ====
	    I instead of:
	    	obj = HeavyObject(arg1, arg2, ...)
	    replace it by:
	        obj = LazyCreate(HeavyObject, arg1, arg2, ...)

	    The object's constructor will not be called until it's
	    actually accessed (e.g., a method is called on it, or
	    an attribute is accessed). All of this is transparent
	    to users of obj -- it behaves exactly like an instance
	    of HeavyObject.
	    
	    Special kwargs read by LazyCreate:
	    ==================================
	    _LC_del_on_pickle -- (default: True)
	    	Destroy the inner object when pickled, if it was already
	    	instantiated. When unpickled, the LazyCreate instance will
	    	start with the inner object uncreated.
	"""
	def __init__(self, cls, *args, **kwargs):
		# Extract arguments intended for LazyCreate
		lcargs = Namespace()
		lcargs.del_on_pickle = True
		for k, v in kwargs.iteritems():
			if k.startswith('_LC_'): setattr(lcargs, k[4:], v)

		kwargs = { k: v for k, v in kwargs.iteritems() if not k.startswith('_LC_') }

		# Store the attributes
		object.__setattr__(self, '__LazyCreate__data__', (cls, args, kwargs))
		object.__setattr__(self, '__LazyCreate__obj__', None)
		object.__setattr__(self, '__LazyCreate__lcargs__', lcargs)

	def __getattribute__(self, name):
		if name in ['_LazyCreate__get_internal', '__reduce_ex__', '__reduce__', '__getnewargs__', '__getstate__', '__class__', '__setstate__']:
			return object.__getattribute__(self, name)
		return getattr(self.__get_internal(), name)

	def __setattr__(self, name, value):
		return setattr(self.__get_internal(), name, value)

	def __delattr__(self, name):
		return delattr(self.__get_internal(), name)

	def __call__(self, *args, **kwargs):
		return self.__get_internal()(*args, **kwargs)

	## Object auto-creation
	def __get_internal(self):
		obj = object.__getattribute__(self, '__LazyCreate__obj__')
		if obj is None:
			cls, args, kwargs = object.__getattribute__(self, '__LazyCreate__data__')
			obj = cls(*args, **kwargs)
			object.__setattr__(self, '__LazyCreate__obj__', obj)

		return obj

	## Pickling support: if the object hasn't been constructed,
	## pickling this class won't trigger its construction. But if the object
	## has been constructed, the constructed instance will be pickled and
	## passed along.
	def __getstate__(self):
		objdef = object.__getattribute__(self, '__LazyCreate__data__')
		obj    = object.__getattribute__(self, '__LazyCreate__obj__')
		lcargs = object.__getattribute__(self, '__LazyCreate__lcargs__')

		if lcargs.del_on_pickle:
			obj = None

		return lcargs, objdef, obj

	def __setstate__(self, state):
		lcargs, objdef, obj = state
		object.__setattr__(self, '__LazyCreate__lcargs__', lcargs)
		object.__setattr__(self, '__LazyCreate__data__', objdef)
		object.__setattr__(self, '__LazyCreate__obj__', obj)

def open_ex(fname):
	""" Transparently open bzipped/gzipped/raw file, based on suffix """
	# lifted from numpy.loadtxt
	if fname.endswith('.gz'):
		import gzip
		fh = gzip.GzipFile(fname)
	elif fname.endswith('.bz2'):
		import bz2
		fh = bz2.BZ2File(fname)
	else:
		fh = file(fname)

	return fh

def isiterable(x):
	try:
		iter(x)
		return True
	except TypeError:
		return False

def unpack_callable(func):
	""" Unpack a (function, function_args) tuple
	"""
	func, func_args = (func, ()) if callable(func) or func is None else (func[0], func[1:])
	return func, func_args

def gnomonic(lon, lat, clon, clat):
	from numpy import sin, cos

	phi  = np.radians(lat)
	l    = np.radians(lon)
	phi1 = np.radians(clat)
	l0   = np.radians(clon)

	cosc = sin(phi1)*sin(phi) + cos(phi1)*cos(phi)*cos(l-l0)
	x = cos(phi)*sin(l-l0) / cosc
	y = (cos(phi1)*sin(phi) - sin(phi1)*cos(phi)*cos(l-l0)) / cosc

	return (np.degrees(x), np.degrees(y))

def gc_dist(lon1, lat1, lon2, lat2):
	from numpy import sin, cos, arcsin, sqrt

	lon1 = np.radians(lon1); lat1 = np.radians(lat1)
	lon2 = np.radians(lon2); lat2 = np.radians(lat2)

	return np.degrees(2*arcsin(sqrt( (sin((lat1-lat2)*0.5))**2 + cos(lat1)*cos(lat2)*(sin((lon1-lon2)*0.5))**2 )));

_fmt_map = {
	'int8':         '%4d',
	'int16':	'%6d',
	'int32':	'%11d',
	'int64':	'%21d',
	'float32':	'%7.3f',
	'float64':	'%12.8f',
	'uint8':        '%3s',
	'uint16':	'%5s',
	'uint32':	'%10s',
	'uint64':	'%20s',
	'bool':		'%1d'
}

def get_fmt(dtype):
	#
	# Note: there's a bug with formatting long integers (they're formatted as signed), that will be fixed in numpy 1.5.1
	#       Once it's is fixed, change the format chars for uints back to 'd'
	#
	# http://projects.scipy.org/numpy/ticket/1287
	#
	if dtype.kind == 'S':
		return '%' + str(dtype.itemsize) + 's'
	if dtype == np.dtype(np.object_):
		return '%s'

	# Note: returning %s by default for unknown types
	stype = str(dtype)
	return _fmt_map[stype] if stype in _fmt_map else '%s'

def make_printf_string(row):
	fmt = ' '.join( [ get_fmt(row.dtype.fields[name][0]) for name in row.dtype.names ] )
	return fmt

def is_scalar_of_type(v, t):
	s = np.array([], v).dtype.type
	return s == t

def str_dtype(dtype):
	""" Return a comma-separated dtype string given a dtype
	    object.

	    Note: This will NOT work for any dtype object. Example of one:

	    	dtype(('i8,f4', (64,)))

	    See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html for details
	"""
	if not dtype.subdtype:
		if dtype.fields is None:
			kind = dtype.kind if dtype.kind != 'S' else 'a'
			itemsize = str(dtype.itemsize)	
			assert len(dtype.shape) == 0

			return kind + itemsize
		else:
			s = ''
			for f in dtype.names:
				if len(s): s += ','
				s += str_dtype(dtype[f])
			return s
	else:
		# Fetch info from subtype
		s = str_dtype(dtype.subdtype[0])

		assert len(dtype.shape) != 0
		assert s.find(',') == -1, "Arrays of structured arrays cannot be represented as comma-separated strings"

		if len(dtype.shape) == 1:
			s = str(dtype.shape[0]) + s
		else:
			s = str(dtype.shape) + s

		return s

def full_dtype(arr):
	""" Return the dtype string of the ndarray that includes
	    the array shape. Useful when merging multiple ndarray
	    columns into a single structured array table

	    See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html for details
	"""
	shape = arr.shape[1:]
	dtype = str(shape) + str_dtype(arr.dtype) if len(shape) else str_dtype(arr.dtype)
	return dtype

def as_tuple(row):
	return tuple((row[col] for col in row.dtype.names))

def as_columns(rows, start=None, stop=None, stride=None):
	# Emulate slice syntax: only one index present
	if stop == None and stride == None:
		stop = start
		start = None
	return tuple((rows[col] for col in rows.dtype.names[slice(start, stop, stride)]))

def shell(cmd):
	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	(out, err) = p.communicate();
	if p.returncode != 0:
		err = subprocess.CalledProcessError(p.returncode, cmd)
		raise err
	return out;

def mkdir_p(path):
	''' Recursively create a directory, but don't fail if it already exists. '''
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST:
			pass
		else:
			raise

def chunks(l, n):
	""" Yield successive n-sized chunks from l.
	    From http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
	"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def astype(v, t):
	""" Typecasting that works for arrays as well as scalars.
	    Note: arrays not being 1st order types in python is truly
	          annoying for scientific applications....
	"""
	if type(v) == np.ndarray:
		return v.astype(t)
	return t(v)

# extract/compact functions by David Zaslavsky from 
# http://stackoverflow.com/questions/783781/python-equivalent-of-phps-compact-and-extract
#
# -- mjuric: modification to extract to ensure variable names are legal
import inspect

legal_variable_characters = ''
for i in xrange(256):
	c = chr(i)
	legal_variable_characters = legal_variable_characters + (c if c.isalnum() else '_')

def compact(*names):
	caller = inspect.stack()[1][0] # caller of compact()
	vars = {}
	for n in names:
		if n in caller.f_locals:
			vars[n] = caller.f_locals[n]
		elif n in caller.f_globals:
			vars[n] = caller.f_globals[n]
	return vars

def extract(vars, level=1):
	caller = inspect.stack()[level][0] # caller of extract()
	for n, v in vars.items():
		n = n.translate(legal_variable_characters)
		caller.f_locals[n] = v   # NEVER DO THIS ;-)

def extract_row(row, level=1):
	caller = inspect.stack()[level][0] # caller of extract()
	for n in row.dtype.names:
		v = row[n]
		n = n.translate(legal_variable_characters)
		caller.f_locals[n] = v   # NEVER DO THIS ;-)

def xhistogram(data, bin_edges):
	""" Bin the points in 'data' into bins whose edges are given by bin_edges.
	    The output array at location i will contain the number of points pts
	    satisfying bin_edges[i-1] < pts < bin_edges[i]
	    
	    Points less than bin_edges[0] and greater than bin_edges[-1] will be
	    at indices 0 and len(bin_edges) in the output array, respectively.
	"""
	bins = np.empty(len(bin_edges)+2, dtype='f8')
	bins[0]    = -np.inf
	bins[1:-1] = bin_edges
	bins[-1]   =  np.inf
	hist, _ = np.histogram(data, bins)
	return hist

## YAML related utils

if __name__ == "__main__":
	import smf, yaml_ex as yaml, json
	import surveys.galex
	schema = surveys.galex.schema
	text = yaml.safe_dump(schema)
	schema2 = yaml.load(text)
	print schema2
	print yaml.safe_dump(schema2)
	#v = tuples_to_lists(schema)
	#print yaml.dump(v)
