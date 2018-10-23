.. _building:

Building
********

setup.py
========

Short story: You run :file:`setup.py` but you should ideally follow
the :ref:`recommended way <recommended_build>` which will also fetch
needed components for you.

+-------------------------------------------------------------+-------------------------------------------------------------------------+
| Command                                                     |  Result                                                                 |
+=============================================================+=========================================================================+
| | python setup.py install test                              | Compiles APSW with default Python compiler, installs it into Python     |
|                                                             | site library directory and then runs the test suite.                    |
+-------------------------------------------------------------+-------------------------------------------------------------------------+
| | python setup.py install :option:`--user`                  | (Python 2.6+, 3). Compiles APSW with default Python                     |
|                                                             | compiler and installs it into a subdirectory of your home directory.    |
|                                                             | See :pep:`370` for more details.                                        |
+-------------------------------------------------------------+-------------------------------------------------------------------------+
| | python setup.py build :option:`--compile=mingw32` install | On Windows this will use the                                            |
|                                                             | `free <http://www.gnu.org/philosophy/free-sw.html>`_                    |
|                                                             | `MinGW compiler <http://mingw.org>`_ instead of the                     |
|                                                             | Microsoft compilers.                                                    |
+-------------------------------------------------------------+-------------------------------------------------------------------------+
| | python setup.py build_ext :option:`--force`               | Compiles the extension but doesn't install it. The resulting file       |
|   :option:`--inplace` test                                  | will be in the current directory named apsw.so (Unix/Mac) or            |
|                                                             | apsw.pyd (Windows). The test suite is then run. (Note on recent versions|
|                                                             | of CPython the extension filenames may be more complicated due to       |
|                                                             | :pep:`3149`.)                                                           |
+-------------------------------------------------------------+-------------------------------------------------------------------------+
| | python setup.py build :option:`--debug` install           | Compiles APSW with debug information.  This also turns on `assertions   |
|                                                             | <http://en.wikipedia.org/wiki/Assert.h>`_                               |
|                                                             | in APSW that double check the code assumptions.  If you are using the   |
|                                                             | SQLite amalgamation then assertions are turned on in that too.  Note    |
|                                                             | that this will considerably slow down APSW and SQLite.                  |
+-------------------------------------------------------------+-------------------------------------------------------------------------+

.. _setup_py_flags:

Additional :file:`setup.py` flags
=================================

There are a number of APSW specific flags to commands you can specify.

fetch
-----

:file:`setup.py` can automatically fetch SQLite and other optional
components.  You can set the environment variable :const:`http_proxy`
to control proxy usage for the download. **Note** the files downloaded
are modified from their originals to ensure various names do not
clash, adjust them to the download platform and to graft them cleanly
into the APSW module.  You should not commit them to source code
control systems (download separately if you need clean files).

If any files are downloaded then the build step will automatically use
them.  This still applies when you do later builds without
re-fetching.

  | python setup.py fetch *options*

+----------------------------------------+--------------------------------------------------------------------------------------+
| fetch flag                             |  Result                                                                              |
+========================================+======================================================================================+
| | :option:`--version=VERSION`          | By default the `SQLite download page                                                 |
|                                        | <https://sqlite.org/download.html>`__ is                                             |
|                                        | consulted to find the current SQLite version                                         |
|                                        | which you can override using this flag.                                              |
|                                        |                                                                                      |
|                                        | .. note::                                                                            |
|                                        |                                                                                      |
|                                        |    You can also specify `fossil` as the version                                      |
|                                        |    and the current development version from `SQLite's source tracking system         |
|                                        |    <https://sqlite.org/src/timeline>`__ will be used.  (The system is named          |
|                                        |    `Fossil <http://www.fossil-scm.org>`__.) Note that checksums can't be checked     |
|                                        |    for fossil. You will also need TCL and make installed for the amalgamation to     |
|                                        |    build as well as several other common Unix tools.  (ie this is very unlikely to   |
|                                        |    work on Windows.)                                                                 |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--missing-checksum-ok`      | Allows setup to continue if the :ref:`checksum <fetch_checksums>` is missing.        |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--all`                      | Gets all components listed below.                                                    |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--sqlite`                   | Automatically downloads the `SQLite amalgamation                                     |
|                                        | <https://sqlite.org/amalgamation.html>`__. The amalgamation is the                   |
|                                        | preferred way to use SQLite as you have total control over what components are       |
|                                        | included or excluded (see below) and have no dependencies on any existing            |
|                                        | libraries on your developer or deployment machines. The amalgamation includes the    |
|                                        | fts3/4/5, rtree, json1 and icu extensions. On non-Windows platforms, any existing    |
|                                        | :file:`sqlite3/` directory will be erased and the downloaded code placed in a newly  |
|                                        | created :file:`sqlite3/` directory.                                                  |
+----------------------------------------+--------------------------------------------------------------------------------------+

.. _fetch_checksums:

.. note::

  The SQLite downloads are not `digitally signed
  <http://en.wikipedia.org/wiki/Digital_signature>`__ which means you
  have no way of verifying they were produced by the SQLite team or
  were not modified between the SQLite servers and your computer.

  Consequently APSW ships with a :source:`checksums file <checksums>`
  that includes checksums for the various SQLite downloads.  If the
  download does not match the checksum then it is rejected and an
  error occurs.

  The SQLite download page is not checksummed, so in theory a bad guy
  could modify it to point at a malicious download version instead.
  (setup only uses the page to determine the current version number -
  the SQLite download site URL is hard coded.)

  If the URL is not listed in the checksums file then setup aborts.
  You can use :option:`--missing-checksum-ok` to continue.  You are
  recommended instead to update the checksums file with the
  correct information.

.. _fetch_configure:

.. note::

  (This note only applies to non-Windows platforms.)  By default the
  amalgamation will work on your platform.  It detects
  the operating system (and compiler if relevant) and uses the
  appropriate APIs.  However it then only uses the oldest known
  working APIs.  For example it will use the *sleep* system call.
  More recent APIs may exist but the amalgamation needs to be told
  they exist.  As an example *sleep* can only sleep in increments of
  one second while the *usleep* system call can sleep in increments of
  one microsecond. The default SQLite busy handler does small sleeps
  (eg 1/50th of a second) backing off as needed.  If *sleep* is used
  then those will all be a minimum of a second.  A second example is
  that the traditional APIs for getting time information are not
  re-entrant and cannot be used concurrently from multiple threads.
  Consequently SQLite has mutexes to ensure that concurrent calls do
  not happen.  However you can tell it you have more recent re-entrant
  versions of the calls and it won't need to bother with the mutexes.

  After fetching the amalgamation, setup automatically determines what
  new APIs you have by running the :file:`configure` script that comes
  with SQLite and noting the output.  The information is placed in
  :file:`sqlite3/sqlite3config.h`.  The build stage will automatically
  take note of this as needed.

  If you get the fossil version then the configure script does not
  work.  Instead the fetch will save and re-use any pre-existing
  :file:`sqlite3/sqlite3config.h`.

.. _setup_build_flags:

build/build_ext
---------------

You can enable or omit certain functionality by specifying flags to
the build and/or build_ext commands of :file:`setup.py`.

  | python setup.py build *options*

Note that the options do not accumulate.  If you want to specify multiple enables or omits then you
need to give the flag once and giving a comma separated list.  For example:

  | python setup.py build :option:`--enable=fts3,fts3_parenthesis,rtree,icu`

+----------------------------------------+--------------------------------------------------------------------------------------+
| build/build_ext flag                   | Result                                                                               |
+========================================+======================================================================================+
| | :option:`--enable-all-extensions`    | Enables the STAT4, FTS3/4/5, RTree, JSON1, RBU, and ICU extensions if *icu-config*   |
|                                        | is on your path                                                                      |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--enable=fts3`              | Enables the :ref:`full text search extension <ext-fts3>`.                            |
| | :option:`--enable=fts4`              | This flag only helps when using the amalgamation. If not using the                   |
| | :option:`--enable=fts5`              | amalgamation then you need to separately ensure fts3/4/5 is enabled in the SQLite    |
|                                        | install. You are likely to want the `parenthesis option                              |
|                                        | <https://sqlite.org/compile.html#enable_fts3_parenthesis>`__ on unless you have      |
|                                        | legacy code (`--enable-all-extensions` turns it on).                                 |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--enable=rtree`             | Enables the :ref:`spatial table extension <ext-rtree>`.                              |
|                                        | This flag only helps when using the amalgamation. If not using the                   |
|                                        | amalgamation then you need to separately ensure rtree is enabled in the SQLite       |
|                                        | install.                                                                             |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--enable=json1`             | Enables the :ref:`JSON1 extension <ext-json1>`.                                      |
|                                        | This flag only helps when using the amalgamation. If not using the                   |
|                                        | amalgamation then you need to separately ensure json1 is enabled in the SQLite       |
|                                        | install.                                                                             |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--enable=rbu`               | Enables the :ref:`reumable bulk update extension <ext-rbu>`.                         |
|                                        | This flag only helps when using the amalgamation. If not using the                   |
|                                        | amalgamation then you need to separately ensure rbu is enabled in the SQLite         |
|                                        | install.                                                                             |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--enable=icu`               | Enables the :ref:`International Components for Unicode extension <ext-icu>`.         |
|                                        | Note that you must have the ICU libraries on your machine which setup will           |
|                                        | automatically try to find using :file:`icu-config`.                                  |
|                                        | This flag only helps when using the amalgamation. If not using the                   |
|                                        | amalgamation then you need to separately ensure ICU is enabled in the SQLite         |
|                                        | install.                                                                             |
+----------------------------------------+--------------------------------------------------------------------------------------+
| | :option:`--omit=ITEM`                | Causes various functionality to be omitted. For example                              |
|                                        | :option:`--omit=load_extension` will omit code to do with loading extensions. If     |
|                                        | using the amalgamation then this will omit the functionality from APSW and           |
|                                        | SQLite, otherwise the functionality will only be omitted from APSW (ie the code      |
|                                        | will still be in SQLite, APSW just won't call it). In almost all cases you will need |
|                                        | to regenerate the SQLite source because the omits also alter the generated SQL       |
|                                        | parser. See `the relevant SQLite documentation                                       |
|                                        | <https://sqlite.org/compile.html#omitfeatures>`_.                                    |
+----------------------------------------+--------------------------------------------------------------------------------------+

.. note::

  Extension loading is enabled by default when using the amalgamation
  and disabled when using existing libraries as this most closely
  matches current practise.  Use :option:`--omit=load_extension` or
  :option:`--enable=load_extension` to explicity disable/enable the
  extension loading code.

Finding SQLite 3
================

SQLite 3 is needed during the build process. If you specify
:option:`fetch --sqlite` to the :file:`setup.py` command line
then it will automatically fetch the current version of the SQLite
amalgamation. (The current version is determined by parsing the
`SQLite download page <https://sqlite.org/download.html>`_). You
can manually specify the version, for example
:option:`fetch --sqlite --version=3.7.4`.

These methods are tried in order:

  `Amalgamation <https://sqlite.org/amalgamation.html>`__

      The file :file:`sqlite3.c` and then :file:`sqlite3/sqlite3.c` is
      looked for. The SQLite code is then statically compiled into the
      APSW extension and is invisible to the rest of the
      process. There are no runtime library dependencies on SQLite as
      a result.  When you use :option:`fetch` this is where it places
      the downloaded amalgamation.

  Local build

    The header :file:`sqlite3/sqlite3.h` and library :file:`sqlite3/libsqlite3.{a,so,dll}` is looked for.


  User directories

    If you are using Python 2.6+ or Python 3 and specified
    :option:`--user` then your user directory is searched first. See
    :pep:`370` for more details.

  System directories

    The default compiler include path (eg :file:`/usr/include`) and library path (eg :file:`/usr/lib`) are used.


.. note::

  If you compiled SQLite with any OMIT flags (eg
  :const:`SQLITE_OMIT_LOAD_EXTENSION`) then you must include them in
  the :file:`setup.py` command or file. For this example you could use
  :option:`setup.py build --omit=load_extension` to add the same flags.

.. _recommended_build:

Recommended
===========

These instructions show how to build automatically downloading and
using the amalgamation plus other :ref:`extensions`. Any existing SQLite on
your system is ignored at build time and runtime. (Note that you can
even use APSW in the same process as a different SQLite is used by
other libraries - this happens a lot on Mac.) You should follow these
instructions with your current directory being where you extracted the
APSW source to.

  Windows::

      # Leave out --compile=mingw32 flag if using Microsoft compiler
    > python setup.py fetch --all build --enable-all-extensions --compile=mingw32 install test

  Mac/Linux etc::

    $ python setup.py fetch --all build --enable-all-extensions install test

.. note::

  There will be some warnings during the compilation step about
  sqlite3.c, `but they are harmless <https://sqlite.org/faq.html#q17>`_


The extension just turns into a single file apsw.so (Linux/Mac) or
apsw.pyd (Windows). (More complicated name on Pythons implementing
:pep:`3149`). You don't need to install it and can drop it into any
directory that is more convenient for you and that your code can
reach. To just do the build and not install, leave out *install* from
the lines above. (Use *build_ext --inplace* to have the extension put
in the main directory.)

The test suite will be run. It will print the APSW file used, APSW and
SQLite versions and then run lots of tests all of which should pass.

Source distribution (advanced)
==============================

If you want to make a source distribution or a binary distribution
that creates an intermediate source distribution such as `bdist_rpm`
then you can have the SQLite amalgamation automatically included as
part of it.  If you specify the fetch command as part of the same
command line then everything fetched is included in the source
distribution.  For example this will fetch all components, include
them in the source distribution and build a rpm using those
components::

  $ python setup.py fetch --all bdist_rpm

Testing
=======

SQLite itself is `extensively tested
<https://sqlite.org/testing.html>`__. It has considerably more code
dedicated to testing than makes up the actual database functionality.

APSW includes a :file:`tests.py` file which uses the standard Python
testing modules to verify correct operation. New code is developed
alongside the tests. Reported issues also have test cases to ensure
the issue doesn't happen or doesn't happen again.::

  $ python setup.py test
                 Python /usr/bin/python (2, 6, 6, 'final', 0)
  Testing with APSW file /space/apsw/apsw.so
            APSW version 3.7.4-r1
      SQLite lib version 3.7.4
  SQLite headers version 3007004
      Using amalgamation True
  ............................................................................
  ----------------------------------------------------------------------
  Ran 76 tests in 404.557s

  OK

The tests also ensure that as much APSW code as possible is executed
including alternate paths through the code.  95.5% of the APSW code is
executed by the tests. If you checkout the APSW source then there is a
script :source:`tools/coverage.sh` that enables extra code that
deliberately induces extra conditions such as memory allocation
failures, SQLite returning undocumented error codes etc. That brings
coverage up to 99.6% of the code.

A memory checker `Valgrind <http://valgrind.org>`_ is used while
running the test suite. The test suite is run multiple times to make
any memory leaks or similar issues stand out. A checking version of
Python is also used.  See :source:`tools/valgrind.sh` in the source.

To ensure compatibility with the various Python versions, a script
downloads and compiles all supported Python versions in both 2 byte
and 4 byte Unicode character configurations against the APSW and
SQLite supported versions running the tests. See
:source:`tools/megatest.py` in the source.

In short both SQLite and APSW have a lot of testing!
