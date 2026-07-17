# indataout.spec — build with:  pyinstaller indataout.spec
#
# Produces a single dist\inDataOut.exe. Run it from a folder that also
# contains: .env, static\, templates\, icon.ico  (PyInstaller copies static/
# templates INTO the exe per the datas list below, but .env is deliberately
# NOT bundled — see build_exe.bat for why).

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['desktop_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
    ],
    hiddenimports=[
        'waitress',
        'flask',
        'werkzeug',
        'jinja2',
        'mysql.connector',
        'mysql.connector.locales.eng.client_error',
        'mysql.connector.plugins.mysql_native_password',
        'dotenv',
        'chardet',
        'sklearn',
        'sklearn.preprocessing',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes',
        'pandas',
        'numpy',
        'webview',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='inDataOut',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,          # no terminal window behind the app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)
