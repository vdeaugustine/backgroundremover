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
os.chdir(os.path.dirname(os.path.abspath(__file__)))

APP = ['app_gui.py']
APP_NAME = 'Background Remover'

DATA_FILES = [
    # Include the backgroundremover package
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,  # Will create an icon
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
    'resources': [],
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
