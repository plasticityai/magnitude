.. _exceptions:

Exceptions
**********
.. currentmodule:: apsw


:exc:`apsw.Error` is the base for APSW exceptions.

.. exception:: Error

.. attribute:: Error.result

         For exceptions corresponding to `SQLite error codes
         <https://sqlite.org/c3ref/c_abort.html>`_ codes this attribute
         is the numeric error code.

.. attribute:: Error.extendedresult

         APSW runs with `extended result codes
         <https://sqlite.org/c3ref/c_ioerr_blocked.html>`_ turned on.
         This attribute includes the detailed code.

As an example, if SQLite issued a read request and the system returned
less data than expected then :attr:`~Error.result` would have the value
:const:`SQLITE_IOERR` while :attr:`~Error.extendedresult` would have
the value :const:`SQLITE_IOERR_SHORT_READ`.


APSW specific exceptions
========================

The following exceptions happen when APSW detects various problems.

.. exception:: ThreadingViolationError

  You have used an object concurrently in two threads. For example you
  may try to use the same cursor in two different threads at the same
  time, or tried to close the same connection in two threads at the
  same time.

  You can also get this exception by using a cursor as an argument to
  itself (eg as the input data for :meth:`Cursor.executemany`).
  Cursors can only be used for one thing at a time.

.. exception:: ForkingViolationError

  See :meth:`apsw.fork_checker`.

.. exception:: IncompleteExecutionError

  You have tried to start a new SQL execute call before executing all
  the previous ones. See the :ref:`execution model <executionmodel>`
  for more details.

.. exception:: ConnectionNotClosedError

  This exception is no longer generated.  It was required in earlier
  releases due to constraints in threading usage with SQLite.

.. exception:: ConnectionClosedError

  You have called :meth:`Connection.close` and then continued to use
  the :class:`Connection` or associated :class:`cursors <Cursor>`.

.. exception:: CursorClosedError

  You have called :meth:`Cursor.close` and then tried to use the cursor.

.. exception:: BindingsError

  There are several causes for this exception.  When using tuples, an incorrect number of bindings where supplied::

     cursor.execute("select ?,?,?", (1,2))     # too few bindings
     cursor.execute("select ?,?,?", (1,2,3,4)) # too many bindings

  You are using named bindings, but not all bindings are named.  You should either use entirely the
  named style or entirely numeric (unnamed) style::

     cursor.execute("select * from foo where x=:name and y=?")

  .. note::

     It is not considered an error to have missing keys in a dictionary. For example this is perfectly valid::

          cursor.execute("insert into foo values($a,:b,$c)", {'a': 1})

     *b* and *c* are not in the dict.  For missing keys, None/NULL
     will be used. This is so you don't have to add lots of spurious
     values to the supplied dict. If your schema requires every column
     have a value, then SQLite will generate an error due to some
     values being None/NULL so that case will be caught.


.. exception:: ExecutionCompleteError

  A statement is complete but you try to run it more anyway!


.. exception:: ExecTraceAbort

  The :ref:`execution tracer <executiontracer>` returned False so
  execution was aborted.


.. exception:: ExtensionLoadingError

  An error happened loading an `extension
  <https://sqlite.org/cvstrac/wiki/wiki?p=LoadableExtensions>`_.

.. exception:: VFSNotImplementedError

  A call cannot be made to an inherited :ref:`VFS` method as the VFS
  does not implement the method.

.. exception:: VFSFileClosedError

  The VFS file is closed so the operation cannot be performed.

SQLite Exceptions
=================

The following lists which Exception classes correspond to which `SQLite
error codes <https://sqlite.org/c3ref/c_abort.html>`_.


General Errors
^^^^^^^^^^^^^^

.. exception:: SQLError

  :const:`SQLITE_ERROR`.  This error is documented as a bad SQL query
  or missing database, but is also returned for a lot of other
  situations.  It is the default error code unless there is a more
  specific one.

.. exception:: MismatchError

  :const:`SQLITE_MISMATCH`. Data type mismatch.  For example a rowid
  or integer primary key must be an integer.

.. exception:: NotFoundError

  :const:`SQLITE_NOTFOUND`. Returned when various internal items were
  not found such as requests for non-existent system calls or file
  controls.

Internal Errors
^^^^^^^^^^^^^^^

.. exception:: InternalError

  :const:`SQLITE_INTERNAL`. (No longer used) Internal logic error in SQLite.

.. exception:: ProtocolError

  :const:`SQLITE_PROTOCOL`. (No longer used) Database lock protocol error.

.. exception:: MisuseError

  :const:`SQLITE_MISUSE`.  SQLite library used incorrectly.

.. exception:: RangeError

  :const:`SQLITE_RANGE`.  (Cannot be generated using APSW).  2nd parameter to `sqlite3_bind <https://sqlite.org/c3ref/bind_blob.html>`_ out of range

Permissions Etc
^^^^^^^^^^^^^^^

.. exception:: PermissionsError

  :const:`SQLITE_PERM`. Access permission denied by the operating system, or parts of the database are readonly such as a cursor.

.. exception:: ReadOnlyError

  :const:`SQLITE_READONLY`. Attempt to write to a readonly database.

.. exception:: CantOpenError

  :const:`SQLITE_CANTOPEN`.  Unable to open the database file.

.. exception:: AuthError

  :const:`SQLITE_AUTH`.  :meth:`Authorization <Connection.setauthorizer>` denied.

Abort/Busy Etc
^^^^^^^^^^^^^^

.. exception:: AbortError

  :const:`SQLITE_ABORT`. Callback routine requested an abort.

.. exception:: BusyError

  :const:`SQLITE_BUSY`.  The database file is locked.  Use
  :meth:`Connection.setbusytimeout` to change how long SQLite waits
  for the database to be unlocked or :meth:`Connection.setbusyhandler`
  to use your own handler.

.. exception:: LockedError

  :const:`SQLITE_LOCKED`.  A table in the database is locked.

.. exception:: InterruptError

  :const:`SQLITE_INTERRUPT`.  Operation terminated by
  `sqlite3_interrupt <https://sqlite.org/c3ref/interrupt.html>`_ -
  use :meth:`Connection.interrupt`.

.. exception:: SchemaChangeError

  :const:`SQLITE_SCHEMA`.  The database schema changed.  A
  :meth:`prepared statement <Cursor.execute>` becomes invalid
  if the database schema was changed.  Behind the scenes SQLite
  reprepares the statement.  Another or the same :class:`Connection`
  may change the schema again before the statement runs.  SQLite will
  attempt up to 5 times before giving up and returning this error.

.. exception:: ConstraintError

  :const:`SQLITE_CONSTRAINT`. Abort due to `constraint
  <https://sqlite.org/lang_createtable.html>`_ violation.  This
  would happen if the schema required a column to be within a specific
  range.  If you have multiple constraints, you `can't tell
  <https://sqlite.org/src/tktview/23b212820161c6599cbf414aa99bf8a5bfa5e7a3>`__
  which one was the cause.

Memory/Disk
^^^^^^^^^^^

.. exception:: NoMemError

  :const:`SQLITE_NOMEM`.  A memory allocation failed.

.. exception:: IOError

  :const:`SQLITE_IOERR`.  Some kind of disk I/O error occurred.  The
  :ref:`extended error code <exceptions>` will give more detail.

.. exception:: CorruptError

  :const:`SQLITE_CORRUPT`.  The database disk image appears to be a
  SQLite database but the values inside are inconsistent.

.. exception:: FullError

  :const:`SQLITE_FULL`.  The disk appears to be full.

.. exception:: TooBigError

  :const:`SQLITE_TOOBIG`.  String or BLOB exceeds size limit.  You can
  change the limits using :meth:`Connection.limit`.

.. exception:: NoLFSError

  :const:`SQLITE_NOLFS`.  SQLite has attempted to use a feature not
  supported by the operating system such as `large file support
  <http://en.wikipedia.org/wiki/Large_file_support>`_.

.. exception:: EmptyError

  :const:`SQLITE_EMPTY`. Database is completely empty.

.. exception:: FormatError

  :const:`SQLITE_FORMAT`. (No longer used) `Auxiliary database <https://sqlite.org/lang_attach.html>`_ format error.

.. exception:: NotADBError

  :const:`SQLITE_NOTADB`.  File opened that is not a database file.
  SQLite has a header on database files to verify they are indeed
  SQLite databases.


.. _augmentedstacktraces:

Augmented stack traces
======================

When an exception occurs, Python does not include frames from
non-Python code (ie the C code called from Python).  This can make it
more difficult to work out what was going on when an exception
occurred for example when there are callbacks to collations, functions
or virtual tables, triggers firing etc.

This is an example showing the difference between the tracebacks you
would have got with earlier versions of apsw and the augmented
traceback::

  import apsw

  def myfunc(x):
    1/0

  con=apsw.Connection(":memory:")
  con.createscalarfunction("foo", myfunc)
  con.createscalarfunction("fam", myfunc)
  cursor=con.cursor()
  cursor.execute("create table bar(x,y,z);insert into bar values(1,2,3)")
  cursor.execute("select foo(1) from bar")

+-----------------------------------------------------------+----------------------------------------------------------+
| Original Traceback                                        |      Augmented Traceback                                 |
+===========================================================+==========================================================+
| ::                                                        | ::                                                       |
|                                                           |                                                          |
|   Traceback (most recent call last):                      |   Traceback (most recent call last):                     |
|     File "t.py", line 11, in <module>                     |     File "t.py", line 11, in <module>                    |
|       cursor.execute("select foo(1) from bar")            |       cursor.execute("select foo(1) from bar")           |
|     File "t.py", line 4, in myfunc                        |     File "apsw.c", line 3412, in resetcursor             |
|       1/0                                                 |     File "apsw.c", line 1597, in user-defined-scalar-foo |
|   ZeroDivisionError: integer division or modulo by zero   |     File "t.py", line 4, in myfunc                       |
|                                                           |       1/0                                                |
|                                                           |   ZeroDivisionError: integer division or modulo by zero  |
+-----------------------------------------------------------+----------------------------------------------------------+

In the original traceback you can't even see that code in apsw was
involved. The augmented traceback shows that there were indeed two
function calls within apsw and gives you line numbers should you need
to examine the code. Also note how you are told that the call was in
`user-defined-scalar-foo` (ie you can tell which function was called.)

*But wait, there is more!!!* In order to further aid troubleshooting,
the augmented stack traces make additional information available. Each
frame in the traceback has local variables defined with more
information. You can print out the variables using `ASPN recipe 52215 <http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52215>`_

  In the recipe, the initial code in :func:`print_exc_plus` is far
  more complicated than need be, and also won't work correctly with
  all tracebacks (it depends on :attr:`f_prev` being set which isn't always
  the case). Change the function to start like this::

    tb = sys.exc_info()[2]
    stack = []

    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next

    traceback.print_exc()
    print "Locals by frame, innermost last"


Here is a far more complex example from some :ref:`virtual tables
<Virtualtables>` code I was writing. The BestIndex method in my code
had returned an incorrect value. The augmented traceback includes
local variables using recipe 52215. I can see what was passed in to my
method, what I returned and which item was erroneous. The original
traceback is almost completely useless.

Original traceback::

  Traceback (most recent call last):
    File "tests.py", line 1387, in testVtables
      cursor.execute(allconstraints)
  TypeError: Bad constraint (#2) - it should be one of None, an integer or a tuple of an integer and a boolean

Augmented traceback with local variables::

  Traceback (most recent call last):
    File "tests.py", line 1387, in testVtables
      cursor.execute(allconstraints)
                  VTable =  __main__.VTable
                     cur =  <apsw.Cursor object at 0x988f30>
                       i =  10
                    self =  testVtables (__main__.APSW)
          allconstraints =  select rowid,* from foo where rowid>-1000 ....

    File "apsw.c", line 4050, in Cursor_execute.sqlite3_prepare
              Connection =  <apsw.Connection object at 0x978800>
               statement =  select rowid,* from foo where rowid>-1000 ....

    File "apsw.c", line 2681, in VirtualTable.xBestIndex
                    self =  <__main__.VTable instance at 0x98d8c0>
                    args =  (((-1, 4), (0, 32), (1, 8), (2, 4), (3, 64)), ((2, False),))
                  result =  ([4, (3,), [2, False], [1], [0]], 997, u'\xea', False)

    File "apsw.c", line 2559, in VirtualTable.xBestIndex.result_constraint
                 indices =  [4, (3,), [2, False], [1], [0]]
                    self =  <__main__.VTable instance at 0x98d8c0>
                  result =  ([4, (3,), [2, False], [1], [0]], 997, u'\xea', False)
              constraint =  (3,)

  TypeError: Bad constraint (#2) - it should be one of None, an integer or a tuple of an integer and a boolean


