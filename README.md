# BlueSky - The Open Air Traffic Simulator

BlueSky is distributed under the GNU General Public License v3.

BlueSky is meant as a tool to perform research on Air Traffic Management and Air Traffic Flows.

The goal of BlueSky is to provide everybody who wants to visualize, analyze or simulate air 
traffic with a tool to do so without any restrictions, licenses or limitations. It can be copied, 
modified, cited, etc. without any limitations.

Please send suggestions, proposed changes or contributions through GitHub, preferably after 
debugging it and optimising it for run-time performance.

## BlueSky Releases
If you are not (yet) interested in reading and editing the source of BlueSky, you can also download a release version of BlueSky, that you can install directly, without having to worry about python and library dependencies. You can find the latest release here:
https://github.com/ProfHoekstra/bluesky/releases

## BlueSky Wiki
Installation and user guides are accessible at:
https://github.com/ProfHoekstra/bluesky/wiki

##Some features of BlueSky:
- Written in the freely available, ultra-simple-hence-easy-to-learn, multi-platform language 
Python 2.x (using numpy and either pygame or Qt+OpenGL for visualisation) with source
- Contains open source data on navaids, performance data of aircraft and geography
- Global coverage navaid and airport data
- Contains simulations of aircraft performance, flight management system (LNAV, VNAV under construction), 
autopilot, conflict detection and resolution and airborne separation assurance systems
- Compatible with BADA 3.x data
- Compatible wth NLR Traffic Manager TMX as used by NLR and NASA LaRC
- Traffic is controlled via user inputs in a console window or by playing scenario files (.SCN) 
containing the same commands with a time stamp before the command ("HH:MM:SS.hh>")
- Mouse clicks in traffic window are use in console for lat/lon/heading and position inputs

## Contributions
BlueSky is still under heavy development. We would like to encourage anyone with a strong interest in 
ATM and/or Python to join us. Please feel free to comment, criticise, and contribute to this project.
