# -*- mode: python -*-

block_cipher = None


a = Analysis(['BlueSky.py'],
             pathex=['/Users/joost/work/Python/bluesky'],
             binaries=None,
             datas=[('data', 'data'), ('scenario', 'scenario'), ('settings.cfg', '.')],
             hiddenimports=[],
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
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='BlueSky')
app = BUNDLE(coll,
             name='BlueSky.app',
             icon='data/graphics/icon.gif',
             bundle_identifier=None)
