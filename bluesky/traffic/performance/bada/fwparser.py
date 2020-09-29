''' Text parsing utilities for fixed-width column data'''
# Use parser = FixedWidthParser(specformat)
#
# List of formats strings, one per string per line, each string contains field formats:
#
#   First argument per line is a discriminator, it searches first line with this key
#
#      3x = skip 3 places
#      6S = string of 6 chars
#     10F = float of 10 chars
#      5I = integer of 5 chars
#
#     'CD, 3X, 6S, 9X, 1I, 12X, 9S, 17X, 1S',  # means find first line starting with CD, then skip 3 columns,
#     'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
#     'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
#     'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
#
# First create a parser for a file format, then you can read different files with this parser,
# by calling the parse(file) method

import re

# Define regular expressions for finding:
# The types in a dataformat string
re_types = re.compile('[.*[fis]]*', re.IGNORECASE)
# The ranges that need to be skipped (indicated with [0-9]X)
re_skip  = re.compile('[\s,]*(\d+)x[\s,]*', re.IGNORECASE)
# The ranges that contain data (indicated with [0-9][fis])
re_data  = re.compile('[\s,]*(\d+)[fis][\s,]*', re.IGNORECASE)


class FixedWidthParser:
    ''' Use a FixedWidthParser to parse fixed-width column-based text files.
        In BlueSky, this is currently used to parse BADA data files.

        The column format should be specified in a table passed to the constructor
        of FixedWidthParser. '''

    # Data elements can be either floats, ints, or strings.
    types = {'f': float, 'i': int, 's': str}

    def __init__(self, specformat):
        self.dformat  = []
        # If the provided dataformat only has one line assume that all lines
        # have the same column format
        self.repeat   = len(specformat) == 1
        # Split the specified dataformat in a table of types to convert the
        # parsed text with, and a list of regular expression objects that will
        # be used to extract the fixed-with data columns.
        for line in specformat:
            line_re   = re.compile(re_data.sub(r'(.{\1})', re_skip.sub(r'.{\1}', line)))
            linetypes = [self.types[t.lower()] for t in re_types.findall(line)]
            self.dformat.append((line_re, linetypes))

    def parse(self, fname):
        line_re, dtypes = self.dformat[0]
        data            = []
        with open(fname) as f:
            for lineno, line in enumerate(f):
                match = line_re.match(line)
                if match:
                    try:
                        dline = [t(s.strip()) for t, s in zip(dtypes, match.groups())]
                    except:
                        raise ParseError(fname, lineno + 1)
                    data.append(dline)
                    if not self.repeat:
                        # If we've extracted all datalines we can close the file
                        if len(data) == len(self.dformat):
                            break
                        # otherwise select the next dataline parser
                        line_re, dtypes = self.dformat[len(data)]
        return data

class ParseError(Exception):
    def __init__(self, fname, lineno):
        super().__init__()
        self.fname = fname
        self.lineno = lineno
