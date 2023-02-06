import sys, os

current = os.path.dirname(os.path.realpath(__file__))

# Get the project root path (manually)
rootdir = os.path.dirname(os.path.dirname(current))

# Add root dir into path
sys.path.append(rootdir)
