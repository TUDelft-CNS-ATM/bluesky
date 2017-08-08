# -*- mode: python -*-

# For QtWebEngine to properly install on mac:
# use macports version of python and pyinstaller 3.2.1
# build with: /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/pyinstaller -w --onedir BlueSky.spec
# Afterwards copy the following into the bundle:
# /opt/local/libexec/qt5/lib/QtWebEngineCore.framework/Helpers/QtWebEngineProcess.app to BlueSky.app/Contents/MacOS
# /opt/local/libexec/qt5/lib/QtWebEngineCore.framework/Resources/* to BlueSky.app/Contents/Resources

block_cipher = None


a = Analysis(['BlueSky.py'],
             pathex=['/Users/joost/work/Python/bluesky'],
             binaries=[],
             datas=[('data/coefficients/BS_aircraft', 'data/coefficients/BS_aircraft'),
                    ('data/coefficients/BS_engines', 'data/coefficients/BS_engines'),
                    ('data/coefficients/BS_procedures', 'data/coefficients/BS_procedures'),
                    ('data/coefficients/BADA/README.md', 'data/coefficients/BADA'),
                    ('data/graphics', 'data/graphics'),
                    ('data/html', 'data/html'),
                    ('data/navdata', 'data/navdata'),
                    ('data/default.cfg', 'data'),
                    ('scenario', 'scenario'),
                    ('plugins/*.py', 'plugins')],
             hiddenimports=['bluesky.ui.qtgl.console'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='BlueSky',
          debug=False,
          strip=False,
          upx=False,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='BlueSky')
app = BUNDLE(coll,
             name='BlueSky.app',
             icon='data/graphics/bluesky.icns',
             info_plist={'NSHighResolutionCapable': 'True'},
             bundle_identifier='org.qt-project.Qt.QtWebEngineCore')
