.. currentmodule:: apsw

.. _shell:

Shell
*****

The shell provides a convenient way for you to interact with SQLite,
perform administration and supply SQL for execution.  It is modelled
after the `shell that comes with SQLite
<https://sqlite.org/sqlite.html>`__ which requires separate
compilation and installation.

A number of the quirks and bugs in the SQLite shell are also
addressed.  It provides command line editing and completion.  You can
easily include it into your own program to provide SQLite interaction
and add your own commands.  The autoimport and find commands are also
useful.

Commands
========

In addition to executing SQL, these are the commands available with
their short help description.  Use `.help *command*` eg (`.help
autoimport`) to get more detailed information.

.. help-begin:

.. code-block:: text

  
  .autoimport FILENAME ?TABLE?  Imports filename creating a table and
                                automatically working out separators and data
                                types (alternative to .import command)
  .backup ?DB? FILE             Backup DB (default "main") to FILE
  .bail ON|OFF                  Stop after hitting an error (default OFF)
  .colour SCHEME                Selects a colour scheme from default, off
  .databases                    Lists names and files of attached databases
  .dump ?TABLE? [TABLE...]      Dumps all or specified tables in SQL text format
  .echo ON|OFF                  If ON then each SQL statement or command is
                                printed before execution (default OFF)
  .encoding ENCODING            Set the encoding used for new files opened via
                                .output and imports
  .exceptions ON|OFF            If ON then detailed tracebacks are shown on
                                exceptions (default OFF)
  .exit                         Exit this program
  .explain ON|OFF               Set output mode suitable for explain (default OFF)
  .find what ?TABLE?            Searches all columns of all tables for a value
  .header(s) ON|OFF             Display the column names in output (default OFF)
  .help ?COMMAND?               Shows list of commands and their usage.  If
                                COMMAND is specified then shows detail about that
                                COMMAND.  ('.help all' will show detailed help
                                about all commands.)
  .import FILE TABLE            Imports separated data from FILE into TABLE
  .indices TABLE                Lists all indices on table TABLE
  .load FILE ?ENTRY?            Loads a SQLite extension library
  .mode MODE ?TABLE?            Sets output mode to one of column csv html insert
                                json line list python tabs tcl
  .nullvalue STRING             Print STRING in place of null values
  .open ?OPTIONS? ?FILE?        Closes existing database and opens a different one
  .output FILENAME              Send output to FILENAME (or stdout)
  .print STRING                 print the literal STRING
  .prompt MAIN ?CONTINUE?       Changes the prompts for first line and
                                continuation lines
  .quit                         Exit this program
  .read FILENAME                Processes SQL and commands in FILENAME (or Python
                                if FILENAME ends with .py)
  .restore ?DB? FILE            Restore database from FILE into DB (default
                                "main")
  .schema ?TABLE? [TABLE...]    Shows SQL for table
  .separator STRING             Change separator for output mode and .import
  .show                         Show the current values for various settings.
  .tables ?PATTERN?             Lists names of tables matching LIKE pattern
  .timeout MS                   Try opening locked tables for MS milliseconds
  .timer ON|OFF                 Control printing of time and resource usage after
                                each query
  .width NUM NUM ...            Set the column widths for "column" mode
  
  

.. help-end:

Command Line Usage
==================

You can use the shell directly from the command line.  Invoke it like
this::

  $ python -c "import apsw;apsw.main()"  [options and arguments]

The following command line options are accepted:

.. usage-begin:

.. code-block:: text

  Usage: program [OPTIONS] FILENAME [SQL|CMD] [SQL|CMD]...
  FILENAME is the name of a SQLite database. A new database is
  created if the file does not exist.
  OPTIONS include:
     -init filename       read/process named file
     -echo                print commands before execution
     -[no]header          turn headers on or off
     -bail                stop after hitting an error
     -interactive         force interactive I/O
     -batch               force batch I/O
     -column              set output mode to 'column'
     -csv                 set output mode to 'csv'
     -html                set output mode to 'html'
     -line                set output mode to 'line'
     -list                set output mode to 'list'
     -python              set output mode to 'python'
     -separator 'x'       set output field separator (|)
     -nullvalue 'text'    set text string for NULL values
     -version             show SQLite version
     -encoding 'name'     the encoding to use for files
                          opened via .import, .read & .output
     -nocolour            disables colour output to screen
  

.. usage-end:

Notes
=====

To interrupt the shell press Control-C. (On Windows if you press
Control-Break then the program will be instantly aborted.)

For Windows users you won't have command line editing and completion
unless you install a `readline module
<http://docs.python.org/library/readline.html>`__.  Fortunately there
is one at https://ipython.org/pyreadline.html which works.
However if this :class:`Shell` offers no completions it will start
matching filenames even if they make no sense in the context.

For Windows users you won't get colour output unless you install
`colorama <http://pypi.python.org/pypi/colorama>`__

Example
=======

All examples of using the SQLite shell should work as is, plus you get
extra features and functionality like colour, command line completion
and better dumps.  (The standard SQLite shell does have several more Commands
that help with debugging and introspecting SQLite itself.)

You can also use the shell programmatically (or even interactively and
programmatically at the same time).  See the :ref:`example
<example-shell>` for using the API.

Unicode
=======

SQLite only works with `Unicode
<http://en.wikipedia.org/wiki/Unicode>`__ strings.  All data supplied
to it should be Unicode and all data retrieved is Unicode.  (APSW
functions the same way because of this.)

At the technical level there is a difference between bytes and
characters.  Bytes are how data is stored in files and transmitted
over the network.  In order to turn bytes into characters and
characters into bytes an encoding has to be used.  Some example
encodings are ASCII, UTF-8, ISO8859-1, SJIS etc.  (With the exception
of UTF-8/16/32, other encodings can only map a very small subset of
Unicode.)

If the shell reads data that is not valid for the input encoding or
cannot convert Unicode to the output encoding then you will get an
error.

When the shell starts, Python automatically detects the encodings to
use for console input and output.  (For example on Unix like systems
the LC_CTYPE environment variable is sometimes used.  On Windows it
can find out the `code page
<http://en.wikipedia.org/wiki/Code_page>`__.)  You can override this
autodetection by setting the PYTHONIOENCODING environment variable.

There is also a .encoding command.  This sets what encoding is used
for any subsequent .read, .import and .output commands but does not
affect existing open files and console.  When other programs offer you
a choice for encoding the best value to pick is UTF8 as it allows full
representation of Unicode.

In addition to specifying the encoding, you can also specify the error
handling when a character needs to be output but is not present in the
encoding.  The default is 'strict' which results in an error.
'replace' will replace the character with '?' or something similar
while 'xmlcharrefreplace' uses xml entities.  To specify the error
handling add a colon and error after the encoding - eg::

   .encoding iso-8859-1:replace

The same method is used when setting PYTHONIOENCODING.

This `Joel on Software article
<http://www.joelonsoftware.com/articles/Unicode.html>`__ contains an
excellent overview of character sets, code pages and Unicode.

Shell class
===========

This is the API should you want to integrate the code into your shell.
Not shown here are the functions that implement various commands.
They are named after the command.  For example .exit is implemented by
command_exit.  You can add new commands by having your subclass have
the relevant functions.  The doc string of the function is used by the
help command.  Output modes work in a similar way.  For example there
is an output_html method and again doc strings are used by the help
function and you add more by just implementing an appropriately named
method.

Note that in addition to extending the shell, you can also use the
**.read** command supplying a filename with a **.py** extension.  You
can then `monkey patch <http://en.wikipedia.org/wiki/Monkey_patch>`__
the shell as needed.

.. autoclass:: apsw.Shell
     :members:
     :undoc-members: