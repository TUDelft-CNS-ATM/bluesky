# TODO items
## Core
core.base.reset: call Base.reset for each derived class?
client-side reset is not called for everything?

## Network branch
- Move ECHO to client
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
Fix: SAVEIC adds np.float64() since numpy2