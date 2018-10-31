.. _types:

Types
*****

.. currentmodule:: apsw

Read about `SQLite 3 types
<https://sqlite.org/datatype3.html>`_. ASPW always maintains the
correct type for values, and never converts them to something
else. Note however that SQLite may convert types based on column
affinity as `described <https://sqlite.org/datatype3.html>`_. ASPW
requires that all values supplied are one of the corresponding
Python/SQLite types (or a subclass).

Mapping
=======

* None in Python is NULL in SQLite

* Python int or long is INTEGER in SQLite. The value represented must
  fit within a 64 bit signed quantity (long long at the C level) or an
  overflow exception is generated.

* Python's float type is used for REAL in SQLite. (At the C level they
  are both 8 byte quantities and there is no loss of precision).

* In Python 2, Python's string or unicode is used for TEXT supplied to
  SQLite and all text returned from SQLite is unicode.  For Python 3
  only unicode is used.

* For Python 2 the buffer class is used for BLOB in SQLite. In Python
  3 the bytes type is used, although you can still supply buffers.

.. _unicode:

Unicode
=======

All SQLite strings are Unicode. The actual binary representations can
be UTF8, or UTF16 in either byte order. ASPW uses the UTF8 interface
to SQLite which results in the binary string representation in your
database defaulting to UTF8 as well. All this is totally transparent
to your Python code.

Everywhere strings are used (eg as database values, SQL statements,
bindings names, user defined functions) you can use Unicode strings,
and in Python 3 must use Unicode.  In Python 2, you can also use the
bare Python string class, and ASPW will automatically call the unicode
converter if any non-ascii characters are present.

When returning text values from SQLite, ASPW always uses the Python
unicode class.

If you don't know much about Unicode then read `Joel's article
<http://www.joelonsoftware.com/articles/Unicode.html>`_.  SQLite does
not include conversion from random non-Unicode encodings to or from
Unicode.  (It does include conversion between 8 bit and 16 bit Unicode
encodings).  Python includes `codecs
<http://www.python.org/doc/2.5.2/lib/module-codecs.html>`_ for
conversion to or from many different character sets.

If you don't want to use Unicode and instead want a simple bytes in
are the same bytes out then you should only use blobs.

If you want to do manipulation of unicode text such as upper/lower
casing or sorting then you need to know about locales.  This is
because the exact same sequence of characters sort, upper case, lower
case etc differently depending on where you are.  As an example Turkic
languages have multiple letter i, German has ß which behaves like ss,
various accents sort differently in different European countries.
Fortunately there is a library you can ask to do the right locale
specific thing `ICU
<http://en.wikipedia.org/wiki/International_Components_for_Unicode>`_.
A default SQLite compilation only deals with the 26 letter Roman
alphabet.  If you enable ICU with SQLite then you get `good stuff
<https://sqlite.org/src/finfo?name=ext/icu/README.txt>`_.
See the :ref:`building` section on how to enable ICU for SQLite with
APSW.  Note that Python does not currently include ICU support and
hence sorting, upper/lower casing etc are limited and do not take
locales into account.

In summary, never confuse bytes with strings (which C sadly treats as
the same thing).  Either always use bytes (and SQLite blobs) for
everything or use strings (and SQLite strings) for everything.  If you
take the latter approach and have to deal with external input/output
then you must know what encodings are being used and it is best to
convert to Unicode as early as possible on input and late as possible on
output.
