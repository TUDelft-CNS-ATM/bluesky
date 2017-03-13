''' Text parsing utilities for fixed-width column data'''

# TODO: Add user manual with examples of how to use dformat

import re

# Define regular expressions for finding:
# The types in a dataformat string
re_types = re.compile('[.*[fis]]*', re.IGNORECASE)
# The ranges that need to be skipped (indicated with [0-9]X)
re_skip  = re.compile('[\s,]*(\d+)x[\s,]*', re.IGNORECASE)
# The ranges that contain data (indicated with [0-9][fis])
re_data  = re.compile('[\s,]*(\d+)[fis][\s,]*', re.IGNORECASE)


class FixedWidthParser(object):
    ''' Use a FixedWidthParser to parse fixed-width column-based text files.
        In BlueSky, this is currently used to parse BADA data files.

        The column format should be specified in a table passed to the constructor
        of FixedWidthParser. '''

    # Data elements can be either floats, ints, or strings.
    types = {'f': float, 'i': int, 's': str}

    def __init__(self, dformat):
        self.dformat  = []
        # If the provided dataformat only has one line assume that all lines
        # have the same column format
        self.repeat   = len(dformat) == 1
        # Split the specified dataformat in a table of types to convert the
        # parsed text with, and a list of regular expression objects that will
        # be used to extract the fixed-with data columns.
        for line in dformat:
            line_re   = re.compile(re_data.sub(r'(.{\1})', re_skip.sub(r'.{\1}', line)))
            linetypes = [self.types[t.lower()] for t in re_types.findall(line)]
            self.dformat.append((line_re, linetypes))

    def parse(self, file):
        line_re, dtypes = self.dformat[0]
        data            = []
        with open(file) as f:
            for line in f:
                match = line_re.match(line)
                if match:
                    dline = [t(s.strip()) for t, s in zip(dtypes, match.groups())]
                    data.append(dline)
                    if not self.repeat:
                        # If we've extracted all datalines we can close the file
                        if len(data) == len(self.dformat):
                            break
                        # otherwise select the next dataline parser
                        line_re, dtypes = self.dformat[len(data)]
        return data
