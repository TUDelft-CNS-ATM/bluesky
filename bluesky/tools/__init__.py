from ..settings import prefer_compiled
if prefer_compiled:
    try:
        import cgeo as geo
    except ImportError:
        import geo
else:
    import geo
