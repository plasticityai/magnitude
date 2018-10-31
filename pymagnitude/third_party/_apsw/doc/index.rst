APSW documentation
==================

.. centered:: APSW |version| released |today|

Use with SQLite 3.25 or later, CPython 2.3 or later, and CPython
3.1 or later.

APSW provides an SQLite 3 wrapper that provides the thinnest layer
over the `SQLite <https://sqlite.org>`_ database library
possible. Everything you can do from the `SQLite C API
<https://sqlite.org/c3ref/intro.html>`_, you can do from
Python. Although APSW looks vaguely similar to the :pep:`249` (DBAPI),
it is :ref:`not compliant <dbapinotes>` with that API because instead
it works the way SQLite 3 does. (`pysqlite <https://github.com/ghaering/pysqlite>`_
is DBAPI compliant - see the :ref:`differences between apsw and
pysqlite 2 <pysqlitediffs>`).

APSW is hosted at https://github.com/rogerbinns/apsw

Contents:

.. toctree::
   :maxdepth: 2

   tips
   example
   download
   build
   extensions

   apsw
   connection
   cursor
   blob
   backup
   vtable
   vfs
   shell

   exceptions
   types
   execution
   dbapi
   pysqlite
   benchmarking
   copyright
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
