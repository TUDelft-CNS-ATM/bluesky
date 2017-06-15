# -*- mode: python -*-

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
             hiddenimports=['bluesky.ui.qtgl.console', 'PyQt5.QtWebEngineWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui'],
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
             bundle_identifier=None)
