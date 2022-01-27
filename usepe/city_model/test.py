#!/usr/bin/python

"""

"""

__author__ = 'jbueno'
__copyright__ = '(c) Nommon 2021'


if __name__ == '__main__':
    output_path = r'C:\workspace3\bluesky\scenario\mytest.scn'
    input_path = r'C:\workspace3\bluesky\scenario\tester0.scn'

    output_file = open( output_path, 'a' )

    with open( output_path, 'a' ) as output_file:
        with open( input_path, 'r' ) as input_file:
            for line in input_file:
                new_line = line.strip()
                a = [1, 2, 2, 34, 5, 5, 6]
                output_file.write( ','.join( [str( x ) for x in a] ) + '\n' )

    output_file.close()

    # Para ordenar
    lines = []
    with open( input_path, 'r' ) as input_file:
        for line in input_file:
            lines.append( line.strip() )

    with open( output_path, 'w' ) as outp:
        for item in sorted( lines ):
            pass
