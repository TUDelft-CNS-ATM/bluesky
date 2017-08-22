import os
import json
import numpy as np
import pandas as pd
import warnings
from bluesky.tools import aero
from bluesky import settings

settings.set_variable_defaults(perf_path_nap="data/coefficients/NAP")

ENG_TF = 1
ENG_TP = 2
ENG_PS = 3

nap_path = settings.perf_path_nap
db_aircraft = nap_path + "/aircraft.json"
db_engine = nap_path + "/engines.csv"
envelope_dir = nap_path + "/envelop/"


def get_aircraft(mdl):
    mdl = mdl.upper()
    with open(db_aircraft) as f:
        acs = json.load(f)

    if mdl in acs:
        return acs[mdl]
    else:
        raise RuntimeError('Aircraft data not found')


def get_engine(eng):
    eng = eng.strip().upper()
    allengines = pd.read_csv(db_engine, encoding='utf-8')
    selengine = allengines[allengines['name'].str.startswith(eng)]

    if selengine.shape[0] == 0:
        raise RuntimeError('Engine data not found')

    if selengine.shape[0] > 1:
        warnings.warn('Multiple engines data found, last one returned. \n\
                      matching engines are: %s' % selengine.name.tolist())

    return json.loads(selengine.iloc[-1, :].to_json())


def get_ac_default_engine(mdl):
    ac = get_aircraft(mdl)
    eng = ac['engines'][0]
    return get_engine(eng)


def load_all_aircraft_flavor():
    allengines = pd.read_csv(db_engine, encoding='utf-8')

    with open(db_aircraft) as f:
        warnings.simplefilter("ignore")
        acs = json.load(f)
        del(acs['__comment'])
        for mdl, ac in acs.items():
            engines = ac['engines']
            acs[mdl]['engines'] = {}
            for e in engines:
                e = e.strip().upper()
                selengine = allengines[allengines['name'].str.startswith(e)]
                if selengine.shape[0] >= 1:
                    engine = json.loads(selengine.iloc[-1, :].to_json())
                    acs[mdl]['engines'][engine['name']] = engine
    return acs

def load_all_aircraft_envelop():
    """ load aircraft envelop from the model database,
        All unit in SI"""
    fdb = open(db_aircraft, 'r')
    acs = json.load(fdb)
    del(acs['__comment'])

    limits = {}
    for mdl, ac in acs.items():
        fenv = envelope_dir + mdl.lower() + '.csv'

        if os.path.exists(fenv):
            print(fenv)
            df = pd.read_csv(fenv, index_col='param')
            limits[mdl] = {}
            limits[mdl]['vminto'] = df.loc['to_v_lof']['min']
            limits[mdl]['vmaxto'] = df.loc['to_v_lof']['max']
            limits[mdl]['vminic'] = df.loc['ic_va_avg']['min']
            limits[mdl]['vmaxic'] = df.loc['ic_va_avg']['max']
            limits[mdl]['vminer'] = min(df.loc['cl_v_cas_const']['min'],
                                       df.loc['cr_v_cas_mean']['min'],
                                       df.loc['de_v_cas_const']['min'])
            limits[mdl]['vmaxer'] = min(df.loc['cl_v_cas_const']['max'],
                                       df.loc['cr_v_cas_mean']['max'],
                                       df.loc['de_v_cas_const']['max'])
            limits[mdl]['vminap'] = df.loc['fa_va_avg']['min']
            limits[mdl]['vmaxap'] = df.loc['fa_va_avg']['max']
            limits[mdl]['vminld'] = df.loc['ld_v_app']['min']
            limits[mdl]['vmaxld'] = df.loc['ld_v_app']['max']

            limits[mdl]['vmo'] = limits[mdl]['vmaxer']
            limits[mdl]['mmo'] = df.loc['cr_v_mach_max']['opt']

            limits[mdl]['hmaxalt'] = df.loc['cr_h_max']['opt'] * 1000
            limits[mdl]['crosscl'] = df.loc['cl_h_mach_const']['opt']
            limits[mdl]['crossde'] = df.loc['de_h_cas_const']['opt']

            limits[mdl]['amaxhoriz'] = df.loc['to_acc_tof']['max']

            limits[mdl]['vsmax'] = max(df.loc['ic_vh_avg']['max'],
                                       df.loc['cl_vh_avg_pre_cas']['max'],
                                       df.loc['cl_vh_avg_cas_const']['max'],
                                       df.loc['cl_vh_avg_mach_const']['max'])

            limits[mdl]['vsmin'] = min(df.loc['ic_vh_avg']['min'],
                                       df.loc['de_vh_avg_after_cas']['min'],
                                       df.loc['de_vh_avg_cas_const']['min'],
                                       df.loc['de_vh_avg_mach_const']['min'])

            # limits['amaxverti'] = None # max vertical acceleration (m/s2)
    return limits


def compute_thrust_ratio(phase, bpr, spd, alt, unit='SI'):
    """Computer the dynamic thrust based on engine bypass-ratio, static maximum
    thrust, aircraft true airspeed, and aircraft altitude

    Args:
        phase (int or 1D-array): phase of flight, option: phase.[NA, TO, IC, CL,
            CR, DE, FA, LD, GD]
        bpr (int or 1D-array): engine bypass ratio
        tas (int or 1D-array): aircraft true airspeed (kt)
        alt (int or 1D-array): aircraft altitude (ft)

    Returns:
        int or 1D-array: thust in N
    """

    n = len(phase)

    if unit == 'EP':
        spd = spd * aero.kts
        roc = roc * aero.pfm
        alt = alt * aero.ft

    G0 = 0.0606 * bpr + 0.6337
    Mach = aero.tas2mach(v, h)
    P0 = aero.p0
    P = aero.pressure(H)
    PP = P / P0

    # thrust ratio at take off
    ratio_takeoff = 1 - 0.377 * (1+bpr) / np.sqrt((1+0.82*bpr)*G0) * Mach \
               + (0.23 + 0.19 * np.sqrt(bpr)) * Mach**2

    # thrust ratio for climb and cruise
    A = -0.4327 * PP**2 + 1.3855 * PP + 0.0472
    Z = 0.9106 * PP**3 - 1.7736 * PP**2 + 1.8697 * PP
    X = 0.1377 * PP**3 - 0.4374 * PP**2 + 1.3003 * PP

    ratio_inflight = A - 0.377 * (1+bpr) / np.sqrt((1+0.82*bpr)*G0) * Z * Mach \
          + (0.23 + 0.19 * np.sqrt(bpr)) * X * Mach**2

    # thrust ratio for descent, considering 15% of inflight model thrust
    ratio_idle = 0.15 * ratio_inflight

    # thrust ratio array
    #   LD and GN assume ZERO thrust
    tr = np.zeros(n)
    tr = np.where(phase==ph.TO, ratio_takeoff, 0)
    tr = np.where(phase==ph.IC or phase==ph.CL or phase==ph.CR,
                  ratio_inflight, 0)
    tr = np.where(phase==ph.DE or phase==ph.FA,
                  ratio_idle, 0)

    return tr


def compute_fuel_flow(thrust_ratio, n_engines, fficao):
    """Compute fuel flow based on engine icao fuel flow model

    Args:
        thrust_ratio (1D-array): thrust ratio between 0 and 1
        n_engines (1D-array): number of engines on the aircraft
        fficao (2D-array): rows are
            ff_idl : fuel flow - idle thrust
            ff_ap : fuel flow - approach
            ff_co : fuel flow - climb out
            ff_to : fuel flow - takeoff

    Returns:
        float or 1D-array: Fuel flow in kg
    """

    ff_idl = fficao[:, 0]
    ff_ap = fficao[:, 1]
    ff_co = fficao[:, 2]
    ff_to = fficao[:, 3]

    # standard fuel flow at test thrust ratios
    y = [np.zeros(ff_idl.shape), ff_idl, ff_ap, ff_co, ff_to]
    x = [0, 0.07, 0.3, 0.85, 1.0]  # test thrust ratios

    ff_model = np.poly1d(np.polyfit(x, y, 2))      # fuel flow model f(T/T0)
    ff = ff_model(thrust_ratio) * n_engines
    return ff
