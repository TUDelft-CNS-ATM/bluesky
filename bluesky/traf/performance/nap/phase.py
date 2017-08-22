import numpy as np

NA = 0  # Unknown phase
TO = 1  # Take-off
IC = 2  # Initial climb
ER = 3  # Cruise
AP = 5  # Approach
LD = 6  # Landing
GD = 7  # Ground, Taxiing


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


    if unit == 'EP':
        spd = spd * 0.514444
        roc = roc * 0.00508
        alt = alt * 0.3048

    PH = np.zeros(len(spd), dtype=int)

    PH[np.where(alt<10) & (roc<100) & (roc>-100)] = GD
    PH[np.where((alt>=10) & (roc>100) & (alt<1000))] = IC
    PH[np.where((alt>=10) & (roc>100) & (alt>1000))] = CL
    PH[np.where((alt>=10) & (roc<-100) & (alt>1000))] = DE
    PH[np.where((alt>=10) & (roc<-100) & (alt<1000))] = FA
    PH[np.where((alt>=10) & (roc<-100) & (alt>1000))] = DE

    return PH
