.. _pysqlitediffs:

pysqlite differences
********************

.. currentmodule:: apsw

pysqlite and APSW approached the problem of providing access to SQLite
from Python from fundamentally different directions.

APSW only wraps version 3 of SQLite and provides access in whatever
way is normal for SQLite.  It makes no effort to hide how SQLite is
different from other databases.

pysqlite tries to provide a DBAPI compliant wrapper for SQLite and in
doing so needs to make it have the same behaviour as other databases.
Consequently it does hide some of SQLite's nuances.

.. note::

   I suggest using APSW when you want to directly use SQLite and its
   functionality or are using your own code to deal with database
   independence rather than DBAPI.  Use pysqlite and DBAPI if your
   needs are simple, and you don't want to use SQLite features.


What APSW does better
=====================

APSW has the following enhancements/differences over pysqlite 2 (wrapping SQLite 3):

* APSW stays up to date with SQLite.  As features are added and
  functionality changed in SQLite, APSW tracks them.

* APSW gives all functionality of SQLite including :ref:`virtual
  tables <virtualtables>`, :ref:`VFS`, :ref:`BLOB I/O <blobio>`,
  :ref:`backups <backup>` and :meth:`file control
  <Connection.filecontrol>`.

* You can use the same :class:`Connection` across threads with APSW
  without needing any additional level of locking.  pysqlite requires
  that the :class:`Connection` and any :class:`cursors <Cursor>` are
  used in the same thread.  You can disable its checking, but unless you
  are very careful with your own mutexes you will have a crash or a
  deadlock.

* APSW is a single file for the extension, :file:`apsw.pyd` on Windows
  and :file:`apsw.so` on Unix/Mac (Note :pep:`3149`). There are no
  other files needed and the :ref:`build instructions <building>` show
  you how to include SQLite statically in this file. You can put this
  file anywhere your Python session can reach. pysqlite is one binary
  file and several .py files, all of which need to be available.

* **Nothing** happens behind your back. By default pysqlite tries to
  manage transactions by parsing your SQL for you, but you can turn it
  off. This can result in very unexpected behaviour with pysqlite.

* When using a :class:`Connection` as a :meth:`context manager
  <Connection.__enter__>` APSW uses SQLite's ability to have `nested
  transactions <https://sqlite.org/lang_savepoint.html>`__.
  pysqlite only deals with one transaction at a time and cannot nest
  them.  (Savepoints were introduced in SQLite 3.6.8 - another
  illustration of the benefits of keeping up to date with SQLite.)

* APSW **always** handles Unicode correctly (this was one of the major
  reasons for writing it in the first place). pysqlite has since fixed
  many of its issues but you are still stuck with some.

* You can use semi-colons at the end of commands and you can have
  multiple commands in the execute string in APSW. There are no
  restrictions on the type of commands used. For example this will
  work fine in APSW but is not allowed in pysqlite::

    import apsw
    con=apsw.Connection(":memory:")
    cur=con.cursor()
    for row in cur.execute("create table foo(x,y,z);insert into foo values (?,?,?);"
                           "insert into foo values(?,?,?);select * from foo;drop table foo;"
                           "create table bar(x,y);insert into bar values(?,?);"
                           "insert into bar values(?,?);select * from bar;",
                           (1,2,3,4,5,6,7,8,9,10)):
                               print row

  And the output as you would expect::

    (1, 2, 3)
    (4, 5, 6)
    (7, 8)
    (9, 10)

* :meth:`Cursor.executemany` also works with statements that return
  data such as selects, and you can have multiple statements.
  pysqlite's :meth:`executescript` method doesn't allow any form of
  data being returned (it silently ignores any returned data).

* pysqlite swallows exceptions in your callbacks making it far harder
  to debug problems. That also prevents you from raising exceptions in
  your callbacks to be handled in your code that called
  SQLite. pysqlite does let you turn on `printing of tracebacks
  <http://readthedocs.org/docs/pysqlite/en/latest/sqlite3.html#sqlite3.enable_callback_tracebacks>`_,
  but
  that is a poor substitute. apsw does the right thing as demonstrated
  by this example.

  Source::

    def badfunc(t):
        return 1/0

    # pysqlite
    from pysqlite2 import dbapi2 as sqlite

    con = sqlite.connect(":memory:")
    con.create_function("badfunc", 1, badfunc)
    cur = con.cursor()
    cur.execute("select badfunc(3)")

    # apsw
    import apsw
    con = apsw.Connection(":memory:")
    con.createscalarfunction("badfunc", badfunc, 1)
    cur = con.cursor()
    cur.execute("select badfunc(3)")

  Exceptions::

    # pysqlite

    Traceback (most recent call last):
      File "func.py", line 8, in ?
        cur.execute("select badfunc(3)")
    pysqlite2.dbapi2.OperationalError: user-defined function raised exception

    # apsw

    Traceback (most recent call last):
      File "t.py", line 8, in ?
        cur.execute("select badfunc(3)")
      File "apsw.c", line 3660, in resetcursor
      File "apsw.c", line 1871, in user-defined-scalar-badfunc
      File "t.py", line 3, in badfunc
        return 1/0

* APSW has significantly enhanced debuggability. More details are
  available than just what is printed out when exceptions happen like
  above. See :ref:`augmented stack traces <augmentedstacktraces>`

* APSW has :ref:`execution and row tracers <tracing>`.  pysqlite has no
  equivalent to :ref:`execution tracers <executiontracer>` and does have
  data adaptors which aren't the same thing as a :ref:`row tracer
  <rowtracer>` (for example you can't skip rows or add a new column to
  each row returned).  pysqlite does have a `row factory
  <http://readthedocs.org/docs/pysqlite/en/latest/sqlite3.html#accessing-columns-by-name-instead-of-by-index>`_
  but you can easily emulate that with the row tracer and
  :meth:`Cursor.getdescription`.

* APSW has an :ref:`apswtrace <apswtrace>` utility script that traces
  execution and results in your code without having to modify it in
  any way.  It also outputs summary reports making it easy to see what
  your most time consuming queries are, which are most popular etc.

* APSW has an exception corresponding to each SQLite error code and
  provides the extended error code.  pysqlite combines several SQLite
  error codes into corresponding DBAPI exceptions.  This is a good
  example of the difference in approach of the two wrappers.

* The APSW test suite is larger and tests more functionality. Code
  coverage by the test suite is 99.6%. pysqlite is good at 81% for C
  code although there are several places that coverage can be
  improved. I haven't measured code coverage for pysqlite's Python
  code.  The consequences of this are that APSW catches issues earlier
  and gives far better diagnostics.  As an example try returning an
  unsupported type from a registered scalar function.

* APSW is faster than pysqlite in my testing.  Try the
  :ref:`speedtest` benchmark.

What pysqlite does better
=========================

* pysqlite has an `adaptor system
  <http://readthedocs.org/docs/pysqlite/en/latest/sqlite3.html#using-adapters-to-store-additional-python-types-in-sqlite-databases>`_
  that lets you pretend SQLite stores and returns more types than it
  really supports.  Note that the database won't be useful in a
  non-pysqlite context (eg PHP code looking at the same database isn't
  going to recognise your Point class).  You can implement something similar in APSW by intercepting :meth:`Cursor.execute` calls
  that suitably mangles the bindings going to SQLite and does
  something similar to the rows the iterator returns.

* pysqlite lets you work with a database that contains invalid Unicode data by
  setting a `text factory
  <http://readthedocs.org/docs/pysqlite/en/latest/sqlite3.html#sqlite3.Connection.text_factory>`_
  that deals with the text data.

  APSW does not let you put non-Unicode data into the database in the first place and it will
  be considered invalid by other tools reading the data (eg Java,
  PHP).  If you somehow do manage to get non-Unicode data as a SQLite
  string, you can cast it to a blob::

     for row in cursor.execute("select CAST(column as BLOB) from table"):
        # row[0] is buffer (py2) or bytes (py3) here
        deal_with_binary_data(row[0])
