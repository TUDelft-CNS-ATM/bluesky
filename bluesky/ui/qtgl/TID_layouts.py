import bluesky as bs
from bluesky.ui.qtgl import console

import bluesky.ui.qtgl.mainwindow
from bluesky.ui.qtgl.TID import showTID
from bluesky.ui.qtgl.TIDS.base_tid import tidclose

base = [  ['a1',    'UCO',     'lambda: console.process_cmdline("UCO ")'],
          ['a2',    'HDG',     'lambda: console.process_cmdline("HDG ")'],
          ['a3',    'EFL',     'lambda: console.process_cmdline("SPD ")'],
          ['a4',    'SPD',     'lambda: console.process_cmdline("ALT FL")'],

          ['b1',    'WPTS',    "lambda: show_basetid('waypoints','wpts')"],
          ['b2',    'MAPS',    "lambda: show_basetid('maps','maps')"],
          ['b3',    '',        None],
          ['b4',    '',        None],

          ['c1',    '7',       'lambda: console.process_cmdline("7")'],
          ['c2',    '4',       'lambda: console.process_cmdline("4")'],
          ['c3',    '1',       'lambda: console.process_cmdline("1")'],
          ['c4',    '0',       'lambda: console.process_cmdline("0")'],

          ['d1',    '8',       'lambda: console.process_cmdline("8")'],
          ['d2',    '5',       'lambda: console.process_cmdline("5")'],
          ['d3',    '2',       'lambda: console.process_cmdline("2")'],
          ['d4',    'SPACE',   'lambda: console.process_cmdline(" ")'],

          ['e1',    '9',       'lambda: console.process_cmdline("9")'],
          ['e2',    '6',       'lambda: console.process_cmdline("6")'],
          ['e3',    '3',       'lambda: console.process_cmdline("3")'],
          ['e4',    'FL',      'lambda: console.process_cmdline("FL")'],

          ['f1',    'COR',     'lambda: console.Console._instance.set_cmdline(console.Console._instance.command_line[:-1])'],
          ['f2',    'CLR',     'lambda: console.Console._instance.set_cmdline("")'],
          ['f3',    '',   None],
          ['f4',    'EXQ',     'lambda: console.Console._instance.stack(console.Console._instance.command_line)']

          ]

wpts =  [ ['a1',    'RIVER',        'lambda:  tidclose(console.process_cmdline("RIVER"), "waypoints")'],
          ['a2',    '',        None],
          ['a3',    '',        None],
          ['a4',    '',        None],

          ['b1',    'SUGOL',        'lambda:  tidclose(console.process_cmdline("SUGOL"), "waypoints")'],
          ['b2',    '',        None],
          ['b3',    '',        None],
          ['b4',    '',        None],

          ['c1',    'ARTIP',        'lambda:  tidclose(console.process_cmdline("SUGOL"), "waypoints")'],
          ['c2',    '',        None],
          ['c3',    '',        None],
          ['c4',    '',        None],

          ['d1',    '',        None],
          ['d2',    '',        None],
          ['d3',    '',        None],
          ['d4',    '',        None],

          ['e1',    '',        None],
          ['e2',    '',        None],
          ['e3',    '',        None],
          ['e4',    '',        None],

          ['f1',    'BACK',        'lambda:  waypoints.close()'],
          ['f2',    '',        None],
          ['f3',    '',        None],
          ['f4',    '',        None],
          ]

maps =  [ ['a1',    '751',        'lambda:  tidclose(console.process_cmdline("map 751"), "maps")'],
          ['a2',    '752',        'lambda:  tidclose(console.process_cmdline("map 752"), "maps")'],
          ['a3',    '',        None],
          ['a4',    '',        None],

          ['b1',    '',        None],
          ['b2',    '',        None],
          ['b3',    '',        None],
          ['b4',    '',        None],

          ['c1',    '',        None],
          ['c2',    '',        None],
          ['c3',    '',        None],
          ['c4',    '',        None],

          ['d1',    '',        None],
          ['d2',    '',        None],
          ['d3',    '',        None],
          ['d4',    '',        None],

          ['e1',    '',        None],
          ['e2',    '',        None],
          ['e3',    '',        None],
          ['e4',    '',        None],

          ['f1',    'BACK',        'lambda:  maps.close()'],
          ['f2',    '',        None],
          ['f3',    '',        None],
          ['f4',    '',        None],
          ]

blank =  [['a1',    '',        None],
          ['a2',    '',        None],
          ['a3',    '',        None],
          ['a4',    '',        None],

          ['b1',    '',        None],
          ['b2',    '',        None],
          ['b3',    '',        None],
          ['b4',    '',        None],

          ['c1',    '',        None],
          ['c2',    '',        None],
          ['c3',    '',        None],
          ['c4',    '',        None],

          ['d1',    '',        None],
          ['d2',    '',        None],
          ['d3',    '',        None],
          ['d4',    '',        None],

          ['e1',    '',        None],
          ['e2',    '',        None],
          ['e3',    '',        None],
          ['e4',    '',        None],

          ['f1',    '',        None],
          ['f2',    '',        None],
          ['f3',    '',        None],
          ['f4',    '',        None],
          ]

