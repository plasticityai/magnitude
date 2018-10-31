.. _dbapinotes:

DBAPI notes
***********

.. currentmodule:: apsw

DBAPI is defined in :pep:`249`. This section describes how APSW complies or differs from it.

Module Interface
================

There is no connect method. Use the :class:`Connection` constructor instead.

The Connection object and any cursors can be used in any thread.  As
an extreme example, you could call :meth:`Cursor.next` in separate
threads each thread getting the next row.  You cannot use the cursor
concurrently in multiple threads for example calling
:meth:`Cursor.execute` at the same time.  If you attempt to do so then
an :exc:`exception <ThreadingViolationError>` will be raised. The
Python Global Interpreter Lock (GIL) is released during all SQLite API
calls allowing for maximum concurrency.

Three different paramstyles are supported. Note that SQLite starts
parameter numbers from one not zero when using *qmark/numeric* style.

+-----------------+---------------------------------+
| qmark           | ``... WHERE name=?``            |
+-----------------+---------------------------------+
| numeric         | ``... WHERE name=?4``           |
+-----------------+---------------------------------+
| named           | | ``... WHERE name=:name``  or  |
|                 | | ``... WHERE name=$name``      |
+-----------------+---------------------------------+

The DBAPI exceptions are not used.  The :ref:`exceptions <exceptions>`
used correspond to specific SQLite error codes.

Connection Objects
==================

There are no commit or rollback methods. You should use
:meth:`Cursor.execute` with `BEGIN` and `COMMIT` or `ROLLBACK` as
appropriate. The `SQLite documentation
<https://sqlite.org/lockingv3.html>`_ has more details.  In
particular note that SQLite does not support nested transactions.  You
can only start one transaction and will get an error if you try to
start another one.

Several methods that are defined in DBAPI to be on the cursor are
instead on the Connection object, since this is where SQLite actually
stores the information. Doing operations in any other cursor attached
to the same Connection object does update their values, and this makes
you aware of that.

Cursor Objects
==============

Use :meth:`Cursor.getdescription` instead of description. This
information is only obtained on request.

.. _rowcount:

There is no rowcount.  Row counts don't make sense in SQLite any way.
SQLite returns results one row at a time, not calculating the next
result row until you ask for it.  Consequently getting a rowcount
would have to calculate all the result rows and would not reduce the
amount of effort needed.

callproc is not implemented as SQLite doesn't support stored procedures.

:meth:`~Cursor.execute` returns the Cursor object and you can use it
as an iterator to get the results (if any).

:meth:`~Cursor.executemany` returns the Cursor object and you can use
it as an iterator to get the results (if any).

fetchone is not available. Use the cursor as an iterator, or call
:meth:`~Cursor.next` to get the next row, or raises StopIteration when
there are no more results.

fetchmany is not available. Simply use the cursor as an iterator or
call :meth:`~Cursor.next` for however many results you want.

fetchall is available, but not too useful. Simply use the cursor as an
iterator, call :meth:`~Cursor.next`, or use list which is less typing::

  all=list(cursor.execute("...."))

nextset is not applicable or implemented.

arraysize is not available as fetchmany isn't.

Neither setinputsizes or setoutputsize are applicable or implemented.

Type objects
============

None of the date or time methods are available since SQLite 3 does not
have a native date or time type.  There are `functions
<https://sqlite.org/lang_datefunc.html>`_ for
manipulating dates and time which are represented as strings or
`Julian days <http://en.wikipedia.org/wiki/Julian_day>`_ (floating
point number).

Use the standard Python buffer class for BLOBs in Python 2 and the
bytes type in Python 3.

Optional DB API Extensions
==========================

rownumber is not available.

Exception classes are not available as attributes of Connection but
instead are on the :mod:`apsw` module.  See :ref:`exceptions` for
more details.

Use :meth:`Cursor.getconnection` to get the associated Connection
object from a cursor.

scroll and messages are not available.

The Cursor object supports the iterator protocol and this is the only
way of getting information back.

To get the last inserted row id, call
:meth:`Connection.last_insert_rowid`. That stores the id from the last
insert on any Cursor associated with the the Connection. You can also
add `select last_insert_rowid() <https://sqlite.org/lang_corefunc.html>`_ to the end of your execute
statements::

  for row in cursor.execute("BEGIN; INSERT ... ; INSERT ... ; SELECT last_insert_rowid(); COMMIT"):
     lastrowid=row[0]

There is no errorhandler attribute.
