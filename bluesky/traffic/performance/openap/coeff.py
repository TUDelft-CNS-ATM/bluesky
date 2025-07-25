""" OpenAP performance library. """
import json
import pandas as pd
import bluesky as bs
from bluesky import stack

from openap import prop, WRAP, drag

bs.settings.set_variable_defaults(perf_path_openap="performance/OpenAP")

LIFT_FIXWING = 1  # fixwing aircraft
LIFT_ROTOR = 2  # rotor aircraft

ENG_TYPE_TF = 1  # turbofan, fixwing
ENG_TYPE_TP = 2  # turboprop, fixwing
ENG_TYPE_TS = 3  # turboshlft, rotor


class Coefficient:
    def __init__(self):
        # Load synonyms.dat text file into dictionary
        self.synodict = {}
        with open(bs.resource(bs.settings.perf_path_openap) / 'synonym.dat', "r") as f_syno:
            for line in f_syno.readlines():
                if line.count("#") > 0:
                    dataline, comment = line.split("#")
                else:
                    dataline = line.strip("\n")
                acmod, synomod = dataline.split("=")
                acmod = acmod.strip().upper()
                synomod = synomod.strip().upper()

                if acmod == synomod:
                    continue
                self.synodict[acmod] = synomod

        self.actypes_fixwing = prop.available_aircraft() # fixed wing types from openap
        self.acs_fixwing = self._load_all_fixwing_flavor()
        self.engines_fixwing = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/engines.csv", encoding="utf-8")
        self.limits_fixwing = self._load_all_fixwing_envelop()

        self.acs_rotor = self._load_all_rotor_flavor()
        self.limits_rotor = self._load_all_rotor_envelop()

        self.actypes_rotor = list(self.acs_rotor.keys())

        df = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/dragpolar.csv", index_col="mdl")
        self.dragpolar_fixwing = df.to_dict(orient="index")
        self.dragpolar_fixwing["NA"] = df.mean().to_dict()
        # self.dragpolar_fixwing = self._load_fixedwing_dragpolar()

    def _load_all_fixwing_flavor(self):
        import warnings
        warnings.simplefilter("ignore")

        # load fixwing aircraft and engine from openap
        acs = {}
        # match acs_ with openap native data
        for mdl in self.actypes_fixwing:
            ac = prop.aircraft(mdl)
            acs[mdl.upper()] = ac.copy()
            engines = []
            engines.append(prop.engine(ac['engine']['default']))
            # options can have repeated strings as default or dicts (with model variant as key), do we handle this?
            # engines.append([prop.engine(e) for e in ac_['engine']['options']])
            acs[mdl.upper()]['engines'] = {}
            for e in engines:
                acs[mdl.upper()]['engines'][e['name']] = e.copy()
        return acs

    def _load_all_rotor_flavor(self):
        # read rotor aircraft
        acs = json.load(open(bs.resource(bs.settings.perf_path_openap) / "rotor/aircraft.json", "r"))
        acs.pop("__comment")
        acs_ = {}
        for mdl, ac in acs.items():
            acs_[mdl.upper()] = ac.copy()
            acs_[mdl.upper()]["lifttype"] = LIFT_ROTOR
        return acs_

    def _load_all_fixwing_envelop(self):
        """load aircraft envelop from the openap database,
        All unit in SI"""
        _MAX = 'maximum'
        _MIN = 'minimum'
        _OPT = 'default'
        limits_fixwing = {}
        for mdl in self.actypes_fixwing:
            wrap = WRAP(ac=mdl)
            mdl = mdl.upper()
            limits_fixwing[mdl] = {}
            limits_fixwing[mdl]["vminto"] = wrap.takeoff_speed()[_MIN]
            limits_fixwing[mdl]["vmaxto"] = wrap.takeoff_speed()[_MAX]
            limits_fixwing[mdl]["vminic"] = wrap.initclimb_vcas()[_MIN]
            limits_fixwing[mdl]["vmaxic"] = wrap.initclimb_vcas()[_MAX]
            limits_fixwing[mdl]["vminer"] = min(
                wrap.initclimb_vcas()[_MIN],
                wrap.climb_const_vcas()[_MIN],
                wrap.cruise_mean_vcas()[_MIN],
                wrap.descent_const_vcas()[_MIN],
                wrap.finalapp_vcas()[_MIN],
            )
            limits_fixwing[mdl]["vmaxer"] = max(
                wrap.initclimb_vcas()[_MAX],
                wrap.climb_const_vcas()[_MAX],
                wrap.cruise_mean_vcas()[_MAX],
                wrap.descent_const_vcas()[_MAX],
                wrap.finalapp_vcas()[_MAX],
            )
            limits_fixwing[mdl]["vminap"] = wrap.finalapp_vcas()[_MIN]
            limits_fixwing[mdl]["vmaxap"] = wrap.finalapp_vcas()[_MAX]
            limits_fixwing[mdl]["vminld"] = wrap.landing_speed()[_MIN]
            limits_fixwing[mdl]["vmaxld"] = wrap.landing_speed()[_MAX]

            limits_fixwing[mdl]["vmo"] = limits_fixwing[mdl]["vmaxer"]
            limits_fixwing[mdl]["mmo"] = wrap.cruise_max_mach()[_OPT]

            limits_fixwing[mdl]["hmax"] = wrap.cruise_max_alt()[_OPT] * 1000.0
            limits_fixwing[mdl]["crosscl"] = wrap.climb_cross_alt_conmach()[_OPT]
            limits_fixwing[mdl]["crossde"] = wrap.descent_const_vcas()[_OPT]

            limits_fixwing[mdl]["axmax"] = wrap.takeoff_acceleration()[_MAX]

            limits_fixwing[mdl]["vsmax"] = max(
                wrap.initclimb_vs()[_MAX],
                wrap.climb_vs_pre_concas()[_MAX],
                wrap.climb_vs_concas()[_MAX],
                wrap.climb_vs_conmach()[_MAX],
            )

            limits_fixwing[mdl]["vsmin"] = min(
                wrap.initclimb_vs()[_MIN],
                wrap.climb_vs_pre_concas()[_MIN],
                wrap.climb_vs_concas()[_MIN],
                wrap.climb_vs_conmach()[_MIN],
            )
        # create envolop based on synonym
        for mdl in self.synodict.keys():
            if mdl not in limits_fixwing:
                limits_fixwing[mdl] = limits_fixwing[self.synodict[mdl]]

        return limits_fixwing

    def _load_all_rotor_envelop(self):
        """load rotor aircraft envelop, all unit in SI"""
        limits_rotor = {}
        for mdl, ac in self.acs_rotor.items():
            limits_rotor[mdl] = {}

            limits_rotor[mdl]["vmin"] = ac["envelop"].get("v_min", -20)
            limits_rotor[mdl]["vmax"] = ac["envelop"].get("v_max", 20)
            limits_rotor[mdl]["vsmin"] = ac["envelop"].get("vs_min", -5)
            limits_rotor[mdl]["vsmax"] = ac["envelop"].get("vs_max", 5)
            limits_rotor[mdl]["hmax"] = ac["envelop"].get("h_max", 2500)

            params = ["v_min", "v_max", "vs_min", "vs_max", "h_max"]
            if set(params) <= set(ac["envelop"].keys()):
                pass
            else:
                warn = f"Warning: Some performance parameters for {mdl} are not found, default values used."
                stack.echo(warn)

        return limits_rotor
    
    def _load_fixedwing_dragpolar(self):
        dragpolar = {}
        for mdl in self.actypes_fixwing:
            mdl = mdl.upper()
            _polar = drag.Drag(mdl).polar
            dragpolar[mdl]['cd0_clean'] = _polar['clean']['cd0']
            dragpolar[mdl]['k_clean'] = _polar['clean']['k']
            dragpolar[mdl]['e_clean'] = _polar['clean']['e']
            # openap relies on flap angle to caculate drag, no TO and LD cd available
            # dragpolar[mdl]['cd0_to'] = 