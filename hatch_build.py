from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

from setuptools import Distribution, Extension
from setuptools.command import build_ext

def get_numpy_include():
    import numpy
    return [numpy.get_include()]


class CustomHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print(version, build_data)
        extensions = [
            Extension('bluesky.tools.geo._cgeo', ['bluesky/tools/geo/src_cpp/_cgeo.cpp']),
            Extension('bluesky.traffic.asas.cstatebased', ['bluesky/traffic/asas/src_cpp/cstatebased.cpp'], include_dirs=['bluesky/tools/geo/src_cpp'])]

        dist = Distribution(dict(name='extended', include_dirs=get_numpy_include(),
                                 ext_modules=extensions))
        dist.package_dir = "extended"
        cmd = build_ext.build_ext(dist)
        cmd.verbose = True  # type: ignore
        cmd.ensure_finalized()  # type: ignore
        cmd.run()
        buildpath = Path(cmd.build_lib)

        # Provide locations of compiled modules
        force_include = {(buildpath / cmd.get_ext_filename('bluesky.tools.geo._cgeo')).as_posix(): cmd.get_ext_filename('bluesky.tools.geo._cgeo'),
                         (buildpath / cmd.get_ext_filename('bluesky.traffic.asas.cstatebased')).as_posix(): cmd.get_ext_filename('bluesky.traffic.asas.cstatebased')}

        build_data['pure_python'] = False
        build_data['infer_tag'] = True
        build_data['force_include'].update(force_include)