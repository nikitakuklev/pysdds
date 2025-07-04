"""
A lexical analyzer class for simple shell-like syntaxes, modified to work with SDDS.
Based on CPython 3.11.
"""

import os
import re
import sys
from collections import deque
import warnings
from io import StringIO

__all__ = ["shlex_sdds", "split_sdds", "quote", "join"]


class shlex_sdds:
    "A lexical analyzer class for simple shell-like syntaxes."

    def __init__(self, instream=None, infile=None, posix=False, punctuation_chars=False):
        if isinstance(instream, str):
            instream = StringIO(instream)
        if instream is not None:
            self.instream = instream
            self.infile = infile
        else:
            self.instream = sys.stdin
            self.infile = None
        self.posix = posix
        if posix:
            self.eof = None
        else:
            self.eof = ""
        self.commenters = "#"
        self.octal_numbers = "01234567"
        self.wordchars = "abcdfeghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        # if self.posix:
        #     self.wordchars += ('ßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'
        #                        'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ')
        self.whitespace = " \t\r\n"
        self.whitespace_split = False
        self.quotes = '"'  # Do not consider single quotes
        self.escape = "\\"
        self.escapedquotes = '"'
        self.state = " "
        self.pushback = deque()
        self.lineno = 1
        self.debug = 0
        self.filestack = deque()
        self.source = None
        if not punctuation_chars:
            punctuation_chars = ""
        elif punctuation_chars is True:
            punctuation_chars = "();<>|&"
        self._punctuation_chars = punctuation_chars
        self.wordchars += "~+$@.:;/[]{}<>|%^-?!"
        if punctuation_chars:
            # _pushback_chars is a push back queue used by lookahead logic
            self._pushback_chars = deque()
            # these chars added because allowed in file names, args, wildcards
            self.wordchars += "~-./*?="
            # remove any punctuation chars from wordchars
            t = self.wordchars.maketrans(dict.fromkeys(punctuation_chars))
            self.wordchars = self.wordchars.translate(t)

    @property
    def punctuation_chars(self):
        return self._punctuation_chars

    def push_token(self, tok):
        "Push a token onto the stack popped by the get_token method"
        if self.debug >= 1:
            print("shlex: pushing token " + repr(tok))
        self.pushback.appendleft(tok)

    def push_source(self, newstream, newfile=None):
        "Push an input source onto the lexer's input source stack."
        if isinstance(newstream, str):
            newstream = StringIO(newstream)
        self.filestack.appendleft((self.infile, self.instream, self.lineno))
        self.infile = newfile
        self.instream = newstream
        self.lineno = 1
        if self.debug:
            if newfile is not None:
                print("shlex: pushing to file %s" % (self.infile,))
            else:
                print("shlex: pushing to stream %s" % (self.instream,))

    def pop_source(self):
        "Pop the input source stack."
        self.instream.close()
        (self.infile, self.instream, self.lineno) = self.filestack.popleft()
        if self.debug:
            print("shlex: popping to %s, line %d" % (self.instream, self.lineno))
        self.state = " "

    def get_token(self):
        "Get a token from the input stream (or from stack if it's nonempty)"
        if self.pushback:
            tok = self.pushback.popleft()
            if self.debug >= 1:
                print("shlex: popping token " + repr(tok))
            return tok
        # No pushback.  Get a token.
        raw = self.read_token()
        # Handle inclusions
        if self.source is not None:
            while raw == self.source:
                spec = self.sourcehook(self.read_token())
                if spec:
                    (newfile, newstream) = spec
                    self.push_source(newstream, newfile)
                raw = self.get_token()
        # Maybe we got EOF instead?
        while raw == self.eof:
            if not self.filestack:
                return self.eof
            else:
                self.pop_source()
                raw = self.get_token()
        # Neither inclusion nor EOF
        if self.debug >= 1:
            if raw != self.eof:
                print("shlex: token=" + repr(raw))
            else:
                print("shlex: token=EOF")
        return raw

    def read_token(self):
        quoted = False
        escaped_octal_mode = False
        octal_buffer = ""
        escapedstate = " "
        token = ""
        state = " "
        octal_pushback = deque()
        while True:
            if self.punctuation_chars and self._pushback_chars:
                nextchar = self._pushback_chars.pop()
            else:
                if octal_pushback:
                    nextchar = octal_pushback.pop()
                else:
                    nextchar = self.instream.read(1)
            if nextchar == "\n":
                self.lineno += 1
            # if self.debug >= 3:
            #     print("shlex: in state %r I see character: %r" % (state,
            #                                                       nextchar))
            if state is None:
                token = ""  # past end of file
                break
            elif state == " ":
                if not nextchar:
                    state = None  # end of file
                    break
                elif nextchar in self.whitespace:
                    if self.debug >= 2:
                        print("shlex: I see whitespace in whitespace state")
                    if token or (self.posix and quoted):
                        break  # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                elif self.posix and nextchar in self.escape:
                    escapedstate = "a"
                    state = nextchar
                # elif nextchar in self.wordchars:
                #     token = nextchar
                #     state = 'a'
                # elif nextchar in self.punctuation_chars:
                #     token = nextchar
                #     state = 'c'
                elif nextchar in self.quotes:
                    if not self.posix:
                        token = nextchar
                    state = nextchar
                elif self.whitespace_split:
                    token = nextchar
                    state = "a"
                else:
                    # Regular character
                    # token = nextchar
                    token = nextchar
                    state = "a"
                    continue
                    # if token or (self.posix and quoted):
                    #     break  # emit current token
                    # else:
                    #     continue
            elif state in self.quotes:
                quoted = True
                if not nextchar:  # end of file
                    if self.debug >= 2:
                        print("shlex: I see EOF in quotes state")
                    # XXX what error should be raised here?
                    raise ValueError("No closing quotation")
                if nextchar == state:
                    if not self.posix:
                        token += nextchar
                        state = " "
                        break
                    else:
                        state = "a"
                elif self.posix and nextchar in self.escape and state in self.escapedquotes:
                    escapedstate = state
                    state = nextchar
                else:
                    token += nextchar
            elif state in self.escape:
                if not nextchar:  # end of file
                    if self.debug >= 2:
                        print("shlex: I see EOF in escape state")
                    # XXX what error should be raised here?
                    raise ValueError("No escaped character")

                if escaped_octal_mode:
                    if nextchar not in self.octal_numbers:
                        escaped_octal_mode = False
                        token += chr(int(octal_buffer, 8))
                        state = escapedstate
                        octal_buffer = ""
                        octal_pushback.append(nextchar)
                    else:
                        octal_buffer += nextchar
                elif nextchar in self.octal_numbers:
                    escaped_octal_mode = True
                    octal_buffer += nextchar
                elif nextchar == "!":
                    # sdds escape
                    token += nextchar
                    state = escapedstate
                elif escapedstate in self.quotes and nextchar != state and nextchar != escapedstate:
                    # In posix shells, only the quote itself or the escape
                    # character may be escaped within quotes.
                    token += state
                    token += nextchar
                    state = escapedstate
                else:
                    token += nextchar
                    state = escapedstate
            elif state in ("a", "c"):
                if not nextchar:
                    state = None  # end of file
                    break
                elif nextchar in self.whitespace:
                    # if self.debug >= 2:
                    #     print("shlex: I see whitespace in word state")
                    state = " "
                    if token or (self.posix and quoted):
                        break  # emit current token
                    else:
                        continue
                elif nextchar in self.commenters:
                    self.instream.readline()
                    self.lineno += 1
                    if self.posix:
                        state = " "
                        if token or (self.posix and quoted):
                            break  # emit current token
                        else:
                            continue
                elif state == "c":
                    if nextchar in self.punctuation_chars:
                        token += nextchar
                    else:
                        if nextchar not in self.whitespace:
                            self._pushback_chars.append(nextchar)
                        state = " "
                        break
                elif self.posix and nextchar in self.quotes:
                    state = nextchar
                elif self.posix and nextchar in self.escape:
                    escapedstate = "a"
                    state = nextchar
                elif nextchar in self.quotes or (self.whitespace_split and nextchar not in self.punctuation_chars):
                    token += nextchar
                else:
                    token += nextchar
                # elif nextchar in self.wordchars:
                #     token += nextchar
                # else:
                #     self.pushback.appendleft(nextchar)
                #     # if self.debug >= 2:
                #     #     print("shlex: I see punctuation in word state")
                #     state = ' '
                #     if token or (self.posix and quoted):
                #         break  # emit current token
                #     else:
                #         continue
        result = token
        self.state = state
        if self.posix and not quoted and result == "":
            result = None
        if self.debug > 1:
            if result:
                print("shlex: raw token=" + repr(result))
            else:
                print("shlex: raw token=EOF")
        return result

    def sourcehook(self, newfile):
        "Hook called on a filename to be sourced."
        if newfile[0] == '"':
            newfile = newfile[1:-1]
        # This implements cpp-like semantics for relative-path inclusion.
        if isinstance(self.infile, str) and not os.path.isabs(newfile):
            newfile = os.path.join(os.path.dirname(self.infile), newfile)
        return (newfile, open(newfile, "r"))

    def error_leader(self, infile=None, lineno=None):
        "Emit a C-compiler-like, Emacs-friendly error-message leader."
        if infile is None:
            infile = self.infile
        if lineno is None:
            lineno = self.lineno
        return '"%s", line %d: ' % (infile, lineno)

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token == self.eof:
            raise StopIteration
        return token


def split_sdds(s, comments=False, posix=True):
    """Split the string *s* using shell-like syntax."""
    if s is None:
        warnings.warn(
            "Passing None for 's' to shlex.split() is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
    lex = shlex_sdds(s, posix=posix)
    lex.whitespace_split = True
    if not comments:
        lex.commenters = ""
    return list(lex)


def join(split_command):
    """Return a shell-escaped string from *split_command*."""
    return " ".join(quote(arg) for arg in split_command)


_find_unsafe = re.compile(r"[^\w@%+=:,./-]", re.ASCII).search


def quote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"
    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _print_tokens(lexer):
    while 1:
        tt = lexer.get_token()
        if not tt:
            break
        print("Token: " + repr(tt))
