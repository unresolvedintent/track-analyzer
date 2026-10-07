import os
import sys

# Make the track_analyzer package importable when pytest runs from a checkout
# without the package installed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
