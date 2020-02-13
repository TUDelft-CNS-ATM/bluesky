import numpy as np
from bluesky.traffic.performance.openap import coeff

NA = 0  # Unknown phase
GD = 1  # Ground
IC = 2  # Initial climb
CL = 3  # Climb
CR = 4  # Cruise
DE = 5  # Descent
AP = 6  # Approach


def readable_phase(ph):
    phases = {
        0: "Unknown phase",
        1: "Ground",
        2: "Initial climb",
        3: "Climb",
        4: "Cruise",
        5: "Descent",
        6: "Approach",
    }
    return phases[ph]


def get(lifttype, spd, roc, alt, unit="SI"):
    ph = np.zeros(len(spd))

    # phase for fixwings
    ph = np.where(lifttype == coeff.LIFT_FIXWING, get_fixwing(spd, roc, alt, unit), ph)

    # phase for rotors
    ph = np.where(lifttype == coeff.LIFT_ROTOR, get_rotor(spd, roc, alt, unit), ph)
    return ph


def get_fixwing(spd, roc, alt, unit="SI"):
    """Get the phase of flight base on aircraft state data

    Args:
    spd (float or 1D array): aircraft speed(s)
    roc (float or 1D array): aircraft vertical rate(s)
    alt (float or 1D array): aricraft altitude(s)
    unit (String):  unit, default 'SI', option 'EP'

    Returns:
    int: phase indentifier

    """

    if unit not in ["SI", "EP"]:
        raise RuntimeError("wrong unit type")

    if unit == "SI":
        spd = spd / 0.514444
        roc = roc / 0.00508
        alt = alt / 0.3048

    ph = np.zeros(len(spd), dtype=int)

    ph[(alt <= 75)] = GD
    ph[(alt >= 75) & (alt <= 1000) & (roc >= 150)] = IC
    ph[(alt >= 75) & (alt <= 1000) & (roc <= -150)] = AP
    ph[(alt >= 1000) & (roc >= 150)] = CL
    ph[(alt >= 1000) & (roc <= -150)] = DE
    ph[(alt >= 10000) & (roc <= 150) & (roc >= -150)] = CR

    return ph


def get_rotor(spd, roc, alt, unit="SI"):
    ph = np.ones(len(spd)) * NA
    return ph
