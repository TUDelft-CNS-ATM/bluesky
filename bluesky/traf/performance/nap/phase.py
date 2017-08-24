import numpy as np

NA = 0  # Unknown phase
TO = 1  # Take-off
IC = 2  # Initial climb
CL = 3  # Climb
CR = 4  # Cruise
DE = 5  # Descent
AP = 6  # Approach
LD = 7  # Landing
GD = 8  # Ground, Taxiing


def get(spd, roc, alt, unit="SI"):
    """Get the phase of flight base on aircraft state data

    Args:
    spd (float or 1D array): aircraft speed(s)
    roc (float or 1D array): aircraft vertical rate(s)
    alt (float or 1D array): aricraft altitude(s)
    unit (String):  unit, default 'SI', option 'EP'

    Returns:
    int: phase indentifier

    """

    if (unit not in ['SI', 'EP']):
        raise RuntimeError('wrong unit type')


    if unit == 'SI':
        spd = spd / 0.514444
        roc = roc / 0.00508
        alt = alt / 0.3048

    PH = np.zeros(len(spd), dtype=int)

    PH[(alt<10) & (roc<100) & (roc>-100)] = GD
    PH[(alt>0) & (alt<1000) & (roc>0)] = IC
    PH[(alt>0) & (alt<1000) & (roc<0)] = AP
    PH[(alt>=1000) & (roc>100)] = CL
    PH[(alt>=1000) & (roc<-100)] = DE
    PH[(alt>=5000) & (roc<100) & (roc>-100)] = CR

    return PH
