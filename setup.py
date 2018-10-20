from __future__ import print_function

import hashlib
import os
import sys
import subprocess
import traceback
import tempfile
import zipfile

from glob import glob
from setuptools import find_packages
from distutils.core import setup
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
from setuptools import setup, Distribution
from multiprocessing import Process

try:
    import pip._internal.pep425tags as pep425tags
    pep425tags.get_supported()
    raise Exception()
except Exception as e:
    import pep425tags

try:
    from urllib.request import urlretrieve
except BaseException:
    from urllib import urlretrieve

PACKAGE_NAME = 'pymagnitude'

# Setup path constants
PROJ_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
THIRD_PARTY = PROJ_PATH + '/pymagnitude/third_party'
BUILD_THIRD_PARTY = PROJ_PATH + '/build/lib/pymagnitude/third_party'
PYSQLITE = THIRD_PARTY + '/_pysqlite'
INTERNAL = THIRD_PARTY + '/internal'
PYSQLITE2 = INTERNAL + '/pysqlite2'

# Get the package version
__version__ = None
with open(os.path.join(PROJ_PATH, 'version.py')) as f:
    exec(f.read())

# Setup remote wheel configurations
RM_WHEELHOUSE = 'https://s3.amazonaws.com/magnitude.plasticity.ai/wheelhouse/'
INSTALLED_FROM_WHEEL = os.path.join(
    tempfile.gettempdir(),
    PACKAGE_NAME +
    '-' +
    __version__ +
    '-' +
    hashlib.md5(PROJ_PATH.encode('utf-8')).hexdigest() +
    '.whlinstall'
)


def get_supported_wheels(package_name=PACKAGE_NAME, version=__version__):
    """Get supported wheel strings"""
    def tuple_invalid(t):
        return (
            t[1] == 'none' or
            'fat32' in t[2] or
            'fat64' in t[2] or
            '_universal' in t[2]
        )
    return ['-'.join((package_name, __version__) + t) + '.whl'
            for t in pep425tags.get_supported() if not(tuple_invalid(t))]


def install_wheel(whl):
    """Installs a wheel file"""
    rc = subprocess.Popen([
        sys.executable,
        '-m',
        'pip',
        'install',
        whl,
    ]).wait()
    return rc


def skip_wheel():
    """ Checks if a wheel install should be skipped """
    return "SKIP_MAGNITUDE_WHEEL" in os.environ


def installed_wheel():
    """Checks if a pre-compiled remote wheel was installed"""
    return os.path.exists(INSTALLED_FROM_WHEEL)


def download_and_install_wheel():
    """Downloads and installs pre-compiled remote wheels"""
    if skip_wheel():
        return False
    if installed_wheel():
        return True
    print("Downloading and installing wheel (if it exists)...")
    tmpwhl_dir = tempfile.gettempdir()
    for whl in get_supported_wheels():
        exitcodes = []
        whl_url = RM_WHEELHOUSE + whl
        dl_path = os.path.join(tmpwhl_dir, whl)
        try:
            urlretrieve(whl_url, dl_path)
        except BaseException:
            continue
        extract_dir = os.path.join(
            tempfile.gettempdir(), whl.replace(
                '.whl', ''))
        extract_dir = os.path.join(
            tempfile.gettempdir(), whl.replace(
                '.whl', ''))
        try:
            zip_ref = zipfile.ZipFile(dl_path, 'r')
        except BaseException:
            continue
        zip_ref.extractall(extract_dir)
        for ewhl in glob(extract_dir + "/*/req_wheels/*.whl"):
            print("Installing requirement wheel: ", ewhl)
            exitcodes.append(install_wheel(ewhl))
        print("Installing wheel: ", dl_path)
        exitcodes.append(install_wheel(dl_path))
        zip_ref.extractall(PROJ_PATH)
        zip_ref.close()
        if len(exitcodes) > 0 and max(exitcodes) == 0 and min(exitcodes) == 0:
            print("Done downloading and installing wheel")
            return True
    print("Done trying to download and install wheel (it didn't exist)")
    return False


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def custom_sqlite3_build():
    """ Checks if custom SQLite has been built already """
    so_files = glob(INTERNAL + '/pysqlite2/*.so')
    pyd_files = glob(INTERNAL + '/pysqlite2/*.pyd')
    return len(so_files + pyd_files) > 0


def install_custom_sqlite3():
    """ Begin install custom SQLite
    Can be safely ignored even if it fails, however, system SQLite
    imitations may prevent large .magnitude files with many columns
    from working."""
    print("Installing custom SQLite 3....")
    rc = subprocess.Popen([
        sys.executable,
        PYSQLITE + '/setup.py',
        'install',
        '--install-lib=' + INTERNAL,
    ], cwd=PYSQLITE).wait()
    if rc:
        print("")
        print("============================================================")
        print("=========================WARNING============================")
        print("============================================================")
        print("It seems like building a custom version of SQLite on your")
        print("machine has failed. This is fine, Magnitude will likely work")
        print("just fine with the sytem version of SQLite for most use cases.")
        print("However, if you are trying to load extremely high dimensional")
        print("models > 999 dimensions, you may run in to SQLite limitations")
        print("that can only be resolved by using the custom version of SQLite.")
        print("To troubleshoot make sure you have appropriate build tools on")
        print("your machine for building C programs like GCC and the standard")
        print("library. Also make sure you have the python-dev development")
        print("libraries and headers for building Python C extensions.")
        print("If you need more help with this, please reach out to ")
        print("opensource@plasticity.ai.")
        print("============================================================")
        print("============================================================")
        print("")
    else:
        print("")
        print("============================================================")
        print("=========================SUCCESS============================")
        print("============================================================")
        print("Building a custom version of SQLite on your machine has")
        print("succeeded.")
        print("Listing internal...")
        os.system("ls '" + INTERNAL + "'")
        os.system("dir '" + INTERNAL + "'")
        print("Listing internal/pysqlite2...")
        os.system("ls '" + PYSQLITE2 + "'")
        os.system("dir '" + PYSQLITE2 + "'")
        print("============================================================")
        print("============================================================")
        print("")


def build_req_wheels():
    """Builds requirement wheels"""
    print("Building requirements wheels...")
    rc = subprocess.Popen([
        sys.executable,
        '-m',
        'pip',
        'wheel',
        '-r',
        'requirements.txt',
        '--wheel-dir=pymagnitude/req_wheels'
    ], cwd=PROJ_PATH).wait()

    download_req_wheels = [
        ('http://download.pytorch.org/whl/cpu/', 'torch', '0.4.1'),
        ('http://download.pytorch.org/whl/cpu/', 'torch', '0.4.1.post2')
    ]

    for wheelhouse, package, version in download_req_wheels:
        for whl in get_supported_wheels(package, version):
            exitcodes = []
            whl_url = wheelhouse + whl
            print("Trying to download...", whl_url)
            dl_path = os.path.join('pymagnitude/req_wheels', whl)
            try:
                urlretrieve(whl_url, dl_path)
            except BaseException:
                continue

    if rc:
        print("Failed to build requirements wheels!")
        pass


def install_req_wheels():
    """Installs requirement wheels"""
    print("Installing requirements wheels...")
    for whl in glob('pymagnitude/req_wheels/*.whl'):
        rc = subprocess.Popen([
            sys.executable,
            '-m',
            'pip',
            'install',
            whl,
        ], cwd=PROJ_PATH).wait()
    print("Done installing requirements wheels")


def install_requirements():
    """Installs requirements.txt"""
    print("Installing requirements...")
    rc = subprocess.Popen([
        sys.executable,
        '-m',
        'pip',
        'install',
        '-r',
        'requirements.txt'
    ], cwd=PROJ_PATH).wait()
    if rc:
        print("Failed to install some requirements!")
    print("Done installing requirements")


def copy_custom_sqlite3():
    """Copy the pysqlite2 folder into site-packages under
    pymagnitude/third_party/internal/ and
    ./build/lib/pymagnitude/third_party/internal/
    for good measure"""
    from distutils.dir_util import copy_tree
    try:
        import site
        cp_from = INTERNAL + '/'
        if hasattr(site, 'getsitepackages'):
            site_packages = site.getsitepackages()
        else:
            from distutils.sysconfig import get_python_lib
            site_packages = [get_python_lib()]
        for sitepack in site_packages:
            globbed = glob(sitepack + '/pymagnitude*/')
            try:
                cp_to = globbed[0] + '/pymagnitude/third_party/internal/'
            except IndexError as e:
                print(
                    "Site Package: '" +
                    sitepack +
                    "' did not have pymagnitude")
                continue
            print("Copying from: ", cp_from, " --> to: ", cp_to)
            copy_tree(cp_from, cp_to)
    except Exception as e:
        print("Error copying internal pysqlite folder to site packages:")
        traceback.print_exc(e)
    try:
        cp_from = INTERNAL + '/'
        cp_to = BUILD_THIRD_PARTY + '/internal/'
        print("Copying from: ", cp_from, " --> to: ", cp_to)
        copy_tree(cp_from, cp_to)
    except Exception as e:
        print("Error copying internal pysqlite folder to build folder:")
        traceback.print_exc(e)


cmdclass = {}

try:
    from wheel.bdist_wheel import bdist_wheel as bdist_wheel_

    class CustomBdistWheelCommand(bdist_wheel_):
        def run(self):
            if not(download_and_install_wheel()):
                install_custom_sqlite3()
                build_req_wheels()
            print("Running wheel...")
            bdist_wheel_.run(self)
            print("Done running wheel")
            copy_custom_sqlite3()

    cmdclass['bdist_wheel'] = CustomBdistWheelCommand

except ImportError as e:
    pass


class CustomInstallCommand(install):
    def run(self):
        if not(download_and_install_wheel()):
            install_custom_sqlite3()
            install_req_wheels()
        print("Running install...")
        p = Process(target=install.run, args=(self,))
        p.start()
        p.join()
        print("Done running install")
        if not(download_and_install_wheel()):
            print("Running egg_install...")
            p = Process(target=install.do_egg_install, args=(self,))
            p.start()
            p.join()
            install_requirements()
            print("Done running egg_install")
        else:
            print("Skipping egg_install")
        copy_custom_sqlite3()

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


cmdclass['install'] = CustomInstallCommand


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


if __name__ == '__main__':

    # Attempt to install from a remote pre-compiled wheel
    if any([a in sys.argv for a in ['egg_info', 'install']]):
        if download_and_install_wheel():
            open(INSTALLED_FROM_WHEEL, 'w+').close()

    # Only create requirements if not installing from a wheel
    if not(installed_wheel()):
        reqs = parse_requirements('requirements.txt')
    else:
        reqs = []

    setup(
        name=PACKAGE_NAME,
        packages=find_packages(
            exclude=[
                'tests',
                'tests.*']),
        version=__version__,
        description='A fast, efficient universal vector embedding utility package.',
        long_description="""
    About
    -----
    A feature-packed Python package and vector storage file format for utilizing vector embeddings in machine learning models in a fast, efficient, and simple manner developed by `Plasticity <https://www.plasticity.ai/>`_. It is primarily intended to be a faster alternative to `Gensim <https://radimrehurek.com/gensim/>`_, but can be used as a generic key-vector store for domains outside NLP.

    Documentation
    -------------
    You can see the full documentation and README at the `GitLab repository <https://gitlab.com/Plasticity/magnitude>`_ or the `GitHub repository <https://github.com/plasticityai/magnitude>`_.
        """,
        author='Plasticity',
        author_email='opensource@plasticity.ai',
        url='https://gitlab.com/Plasticity/magnitude',
        keywords=[
                    'pymagnitude',
                    'magnitude',
                    'plasticity',
                    'nlp',
                    'natural',
                    'language',
                    'processing',
                    'word',
                    'vector',
                    'embeddings',
                    'embedding',
                    'word2vec',
                    'gensim',
                    'alternative',
                    'machine',
                    'learning',
                    'annoy',
                    'index',
                    'approximate',
                    'nearest',
                    'neighbors'],
        license='MIT',
        include_package_data=True,
        install_requires=reqs,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            'Intended Audience :: Developers',
            "Topic :: Software Development :: Libraries :: Python Modules",
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Text Processing :: Linguistic',
            "Operating System :: OS Independent",
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.0',
            'Programming Language :: Python :: 3.7'],
        cmdclass=cmdclass,
        distclass=BinaryDistribution,
    )
