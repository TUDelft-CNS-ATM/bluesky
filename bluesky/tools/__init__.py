from bluesky import settings
# Register settings defaults
settings.set_variable_defaults(prefer_compiled=False)

if settings.prefer_compiled:
    try:
        import cgeo as geo
    except ImportError:
        import geo
else:
    import geo

from dynamicarrays import RegisterElementParameters, DynamicArrays
