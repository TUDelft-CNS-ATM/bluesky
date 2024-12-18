# TODO items
## Network branch
- Move ECHO to client
- What to do with sharedstates that make changes both ways (like pan/zoom)? Is there more than pan/zoom? Otherwise a different approach can also be taken
    Where are pan/zoom used in sim?
    = implemented in screen(io).getviewctr() and getviewbounds()
    - MCRE
    - waypoint lookup
  Possible approach: add explicit (but optional?) reflat/reflon args to these stack functions, with stack implementations on both sim and client side. The stack function on client side adds pan/zoom as ref if no ref specified, and then forwards command to sim
  This also makes these commands (MCRE / WPT) more suitable to script in a
  scenario file
    - PAN/ZOOM as scenario init commands in SCN file.
    => Store client-stack commands for late client joiners?
    => OR: copy entire scenario stack on join / SCN load. Can also be used to show stack in UI
## Common/plugin
- auto separation margin by sector
- time-based separation / RECAT WTC
- Add Metric class to extend logger with more complex metrics?
- Create batch gui tool?

## Stack
STACK signal is used for inter-process STACK commands, but not locally called stack.stack(). Make it so that calling this function also 
emits the STACK signal?