import os
import sys

_dirname = os.path.dirname(__file__)
_newpath = os.path.abspath(os.path.join(_dirname, '..'))
sys.path.insert(0, _newpath)

_newpath = os.path.abspath(os.path.join(_dirname, '..', '..', 'python-rptree'))
sys.path.insert(0, _newpath)
