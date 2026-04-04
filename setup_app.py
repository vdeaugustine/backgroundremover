"""
py2app setup script for Background Remover macOS Application

To build the app:
    python setup_app.py py2app

The built app will be in the dist/ folder.
"""

from setuptools import setup
import os
import sys

# Ensure we're in the right directory
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_REPO_ROOT)

APP = ['app_gui.py']
APP_NAME = 'Background Remover'

_ICNS_PATH = os.path.join(_REPO_ROOT, 'BackgroundRemover.icns')
_WINDOW_ICON_PNG = os.path.join(_REPO_ROOT, 'app_window_icon.png')

DATA_FILES = [
    # Include the backgroundremover package
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': _ICNS_PATH if os.path.isfile(_ICNS_PATH) else None,
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleGetInfoString': 'Remove backgrounds from images with AI',
        'CFBundleIdentifier': 'com.backgroundremover.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': 'Copyright © 2024',
        'NSHighResolutionCapable': True,
        'NSRequiresAquaSystemAppearance': False,  # Support Dark Mode
        'LSMinimumSystemVersion': '12.0',
        'CFBundleDocumentTypes': [
            {
                'CFBundleTypeName': 'Image',
                'CFBundleTypeRole': 'Editor',
                'LSItemContentTypes': [
                    'public.jpeg',
                    'public.png',
                    'public.heic',
                ],
            }
        ],
    },
    'packages': [
        'backgroundremover',
        'torch',
        'torchvision', 
        'PIL',
        'numpy',
        'scipy',
        'skimage',
        'pymatting',
        'hsh',
        'moviepy',
        'tqdm',
        'requests',
        'ffmpeg',
        'filetype',
        'more_itertools',
    ],
    'includes': [
        'tkinter',
        'PIL._tkinter_finder',
    ],
    'excludes': [
        'matplotlib',
        'PyQt5',
        'PyQt6', 
        'PySide2',
        'PySide6',
        'wx',
        'test',
        'tests',
        'unittest',
    ],
    'resources': ['app_window_icon.png'] if os.path.isfile(_WINDOW_ICON_PNG) else [],
    'semi_standalone': False,
    'site_packages': True,
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
