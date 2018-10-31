u"""
A logger that maintains logs of both stdout and stderr when models are run.
"""


from __future__ import absolute_import
#typing
import os
from io import open

def replace_cr_with_newline(message     ):
    u"""
    TQDM and requests use carriage returns to get the training line to update for each batch
    without adding more lines to the terminal output.  Displaying those in a file won't work
    correctly, so we'll just make sure that each batch shows up on its one line.
    :param message: the message to permute
    :return: the message with carriage returns replaced with newlines
    """
    if u'\r' in message:
        message = message.replace(u'\r', u'')
        if not message or message[-1] != u'\n':
            message += u'\n'
    return message

class TeeLogger(object):
    u"""
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::

        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stdout.log", sys.stderr)
    """
    def __init__(self, filename     , terminal        , file_friendly_terminal_output      )        :
        self.terminal = terminal
        self.file_friendly_terminal_output = file_friendly_terminal_output
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, u'a')

    def write(self, message):
        cleaned = replace_cr_with_newline(message)

        if self.file_friendly_terminal_output:
            self.terminal.write(cleaned)
        else:
            self.terminal.write(message)

        self.log.write(cleaned)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
