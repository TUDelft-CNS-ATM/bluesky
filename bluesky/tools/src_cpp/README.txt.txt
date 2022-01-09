Note, to make use of this you currently still have to manually compile
the extension by running:
python setup.py build_ext --inplace
in bluesky/tools/src_cpp
and copying/moving the resulting cgeo.[so/dll] to bluesky/tools.

In bluesky/tools/__init__, change the variable prefer_compiled 
to True if not read correctly from settings.cfg