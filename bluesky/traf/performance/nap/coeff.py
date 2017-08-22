import os
import json
import numpy as np
import pandas as pd
import warnings
# from bluesky import settings
# settings.set_variable_defaults(perf_path_nap="data/coefficients/NAP")

ENG_TF = 1
ENG_TP = 2
ENG_PS = 3

nap_path = os.path.dirname(os.path.realpath(__file__)) \
            + '/../../../../data/coefficients/NAP/'

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
    warnings.simplefilter("ignore")
    allengines = pd.read_csv(db_engine, encoding='utf-8')

    acs = json.load(open(db_aircraft, 'r'))
    acs.pop('__comment')

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
    acs.pop('__comment')

    limits = {}
    for mdl, ac in acs.items():
        fenv = envelope_dir + mdl.lower() + '.csv'

        if os.path.exists(fenv):
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


def get_initial_values(actypes):
    """construct a matrix of initial parameters"""
    acs = load_all_aircraft_flavor()

    engtypes = {
        'TF': ENG_TF,
        'TP': ENG_TP,
        'PS': ENG_PS,
    }

    n = len(actypes)
    params = np.zeros((n, 8))

    unique_ac_mdls = np.unique(actypes)
    for mdl in unique_ac_mdls:
        mdl = mdl.upper()
        if mdl not in list(acs.keys()):
            mdl = 'B744'

        allengs = list(acs[mdl]['engines'].keys())
        params[:, 0] = np.where(actypes==mdl, acs[mdl]['wa'], params[:, 0])
        params[:, 1] = np.where(actypes==mdl, acs[mdl]['oew'], params[:, 1])
        params[:, 2] = np.where(actypes==mdl, acs[mdl]['mtow'], params[:, 2])
        params[:, 3] = np.where(actypes==mdl, acs[mdl]['n_engines'], params[:, 3])
        params[:, 4] = np.where(actypes==mdl, engtypes[acs[mdl]['engine_type']], params[:, 4])
        params[:, 5] = np.where(actypes==mdl, acs[mdl]['engines'][allengs[0]]['thr'], params[:, 5])
        params[:, 6] = np.where(actypes==mdl, acs[mdl]['engines'][allengs[0]]['bpr'], params[:, 6])

    return params
