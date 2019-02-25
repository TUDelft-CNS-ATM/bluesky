''' OpenAP performance library. '''
import os
import json
import numpy as np
import pandas as pd
from bluesky import settings
settings.set_variable_defaults(perf_path_openap="data/performance/OpenAP")

LIFT_FIXWING = 1     # fixwing aircraft
LIFT_ROTOR = 2       # rotor aircraft

ENG_TYPE_TF = 1         # turbofan, fixwing
ENG_TYPE_TP = 2         # turboprop, fixwing
ENG_TYPE_TS = 3         # turboshlft, rotor

synonyms_db = settings.perf_path_openap + "/synonym.dat"

fixwing_aircraft_db = settings.perf_path_openap + "/fixwing/aircraft.json"
fixwing_engine_db = settings.perf_path_openap + "/fixwing/engines.csv"
fixwing_envelops_dir = settings.perf_path_openap + "/fixwing/wrap/"
fixwing_dragpolar_db = settings.perf_path_openap + "/fixwing/dragpolar.csv"

rotor_aircraft_db = settings.perf_path_openap + "/rotor/aircraft.json"

class Coefficient():
    def __init__(self):
        self.acs_fixwing = self.__load_all_fixwing_flavor()
        self.engines_fixwing = pd.read_csv(fixwing_engine_db, encoding='utf-8')
        self.limits_fixwing = self.__load_all_fixwing_envelop()

        self.acs_rotor = self.__load_all_rotor_flavor()
        self.limits_rotor = self.__load_all_rotor_envelop()

        self.actypes_fixwing = list(self.acs_fixwing.keys())
        self.actypes_rotor = list(self.acs_rotor.keys())

        df = pd.read_csv(fixwing_dragpolar_db, index_col='mdl')
        self.dragpolar_fixwing = df.to_dict(orient='index')
        self.dragpolar_fixwing['NA'] = df.mean().to_dict()

        # Load synonyms.dat text file into dictionary
        self.synodict = {}
        f_syno = open(synonyms_db,"r")
        for line in f_syno.readlines():
            if line.count("#")>0:
                dataline,comment = line.split("#")
            else:
                dataline = line.strip("\n")
            acmod,synomod = dataline.split("=")
            self.synodict[acmod.strip().upper()] = synomod.strip().upper()
        f_syno.close()

    def __load_all_fixwing_flavor(self):
        import warnings
        warnings.simplefilter("ignore")

        # read fixwing aircraft and engine files
        allengines = pd.read_csv(fixwing_engine_db, encoding='utf-8')
        allengines['name'] = allengines['name'].str.upper()
        acs = json.load(open(fixwing_aircraft_db, 'r'))
        acs.pop('__comment')
        acs_ = {}

        for mdl, ac in acs.items():
            acengines = ac['engines']
            acs_[mdl.upper()] = ac.copy()
            acs_[mdl.upper()]['lifttype'] = LIFT_FIXWING
            acs_[mdl.upper()]['engines'] = {}

            for e in acengines:
                e = e.strip().upper()
                selengine = allengines[allengines['name'].str.startswith(e)]
                if selengine.shape[0] >= 1:
                    engine = json.loads(selengine.iloc[-1, :].to_json())
                    acs_[mdl.upper()]['engines'][engine['name']] = engine
        return acs_

    def __load_all_rotor_flavor(self):
        # read rotor aircraft
        acs = json.load(open(rotor_aircraft_db, 'r'))
        acs.pop('__comment')
        acs_ = {}
        for mdl, ac in acs.items():
            acs_[mdl.upper()] = ac.copy()
            acs_[mdl.upper()]['lifttype'] = LIFT_ROTOR
        return acs_

    def __load_all_fixwing_envelop(self):
        """ load aircraft envelop from the model database,
            All unit in SI"""
        limits_fixwing = {}
        for mdl, ac in self.acs_fixwing.items():
            fenv = fixwing_envelops_dir + mdl.lower() + '.csv'

            if os.path.exists(fenv):
                df = pd.read_csv(fenv, index_col='param')
                limits_fixwing[mdl] = {}
                limits_fixwing[mdl]['vminto'] = df.loc['to_v_lof']['min']
                limits_fixwing[mdl]['vmaxto'] = df.loc['to_v_lof']['max']
                limits_fixwing[mdl]['vminic'] = df.loc['ic_va_avg']['min']
                limits_fixwing[mdl]['vmaxic'] = df.loc['ic_va_avg']['max']
                limits_fixwing[mdl]['vminer'] = min(df.loc['cl_v_cas_const']['min'],
                                           df.loc['cr_v_cas_mean']['min'],
                                           df.loc['de_v_cas_const']['min'])
                limits_fixwing[mdl]['vmaxer'] = min(df.loc['cl_v_cas_const']['max'],
                                           df.loc['cr_v_cas_mean']['max'],
                                           df.loc['de_v_cas_const']['max'])
                limits_fixwing[mdl]['vminap'] = df.loc['fa_va_avg']['min']
                limits_fixwing[mdl]['vmaxap'] = df.loc['fa_va_avg']['max']
                limits_fixwing[mdl]['vminld'] = df.loc['ld_v_app']['min']
                limits_fixwing[mdl]['vmaxld'] = df.loc['ld_v_app']['max']

                limits_fixwing[mdl]['vmo'] = limits_fixwing[mdl]['vmaxer']
                limits_fixwing[mdl]['mmo'] = df.loc['cr_v_mach_max']['opt']

                limits_fixwing[mdl]['hmax'] = df.loc['cr_h_max']['opt'] * 1000
                limits_fixwing[mdl]['crosscl'] = df.loc['cl_h_mach_const']['opt']
                limits_fixwing[mdl]['crossde'] = df.loc['de_h_cas_const']['opt']

                limits_fixwing[mdl]['axmax'] = df.loc['to_acc_tof']['max']

                limits_fixwing[mdl]['vsmax'] = max(df.loc['ic_vz_avg']['max'],
                                           df.loc['cl_vz_avg_pre_cas']['max'],
                                           df.loc['cl_vz_avg_cas_const']['max'],
                                           df.loc['cl_vz_avg_mach_const']['max'])

                limits_fixwing[mdl]['vsmin'] = min(df.loc['ic_vz_avg']['min'],
                                           df.loc['de_vz_avg_after_cas']['min'],
                                           df.loc['de_vz_avg_cas_const']['min'],
                                           df.loc['de_vz_avg_mach_const']['min'])

                # limits_fixwing['amaxverti'] = None # max vertical acceleration (m/s2)
        return limits_fixwing


    def __load_all_rotor_envelop(self):
        """ load rotor aircraft envelop, all unit in SI"""
        limits_rotor = {}
        for mdl, ac in self.acs_rotor.items():
            limits_rotor[mdl] = {}
            limits_rotor[mdl]['vmin'] = ac['envelop']['v_min']
            limits_rotor[mdl]['vmax'] = ac['envelop']['v_max']
            limits_rotor[mdl]['vsmin'] = ac['envelop']['vs_min']
            limits_rotor[mdl]['vsmax'] = ac['envelop']['vs_max']
            limits_rotor[mdl]['hmax'] = ac['envelop']['h_max']
        return limits_rotor
