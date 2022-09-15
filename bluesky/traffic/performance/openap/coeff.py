""" OpenAP performance library. """
import json
import pandas as pd
import bluesky as bs


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

        self.acs_fixwing = self._load_all_fixwing_flavor()
        self.engines_fixwing = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/engines.csv", encoding="utf-8")
        self.limits_fixwing = self._load_all_fixwing_envelop()

        self.acs_rotor = self._load_all_rotor_flavor()
        self.limits_rotor = self._load_all_rotor_envelop()

        self.actypes_fixwing = list(self.acs_fixwing.keys())
        self.actypes_rotor = list(self.acs_rotor.keys())

        df = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/dragpolar.csv", index_col="mdl")
        self.dragpolar_fixwing = df.to_dict(orient="index")
        self.dragpolar_fixwing["NA"] = df.mean().to_dict()

    def _load_all_fixwing_flavor(self):
        import warnings

        warnings.simplefilter("ignore")

        # read fixwing aircraft and engine files
        allengines = pd.read_csv(bs.resource(bs.settings.perf_path_openap) / "fixwing/engines.csv", encoding="utf-8")
        allengines["name"] = allengines["name"].str.upper()
        acs = json.load(open(bs.resource(bs.settings.perf_path_openap) / "fixwing/aircraft.json", "r"))
        acs.pop("__comment")
        acs_ = {}

        for mdl, ac in acs.items():
            acengines = ac["engines"]
            acs_[mdl.upper()] = ac.copy()
            acs_[mdl.upper()]["lifttype"] = LIFT_FIXWING
            acs_[mdl.upper()]["engines"] = {}

            for e in acengines:
                e = e.strip().upper()
                selengine = allengines[allengines["name"].str.startswith(e)]
                if selengine.shape[0] >= 1:
                    engine = json.loads(selengine.iloc[-1, :].to_json())
                    acs_[mdl.upper()]["engines"][engine["name"]] = engine

        return acs_

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
        """load aircraft envelop from the model database,
        All unit in SI"""
        limits_fixwing = {}
        for mdl, ac in self.acs_fixwing.items():
            fenv = bs.resource(bs.settings.perf_path_openap) / "fixwing/wrap" / (mdl.lower() + ".txt")

            if fenv.is_file():
                df = pd.read_fwf(fenv).set_index("variable")
                limits_fixwing[mdl] = {}
                limits_fixwing[mdl]["vminto"] = df.loc["to_v_lof"]["min"]
                limits_fixwing[mdl]["vmaxto"] = df.loc["to_v_lof"]["max"]
                limits_fixwing[mdl]["vminic"] = df.loc["ic_va_avg"]["min"]
                limits_fixwing[mdl]["vmaxic"] = df.loc["ic_va_avg"]["max"]
                limits_fixwing[mdl]["vminer"] = min(
                    df.loc["ic_va_avg"]["min"],
                    df.loc["cl_v_cas_const"]["min"],
                    df.loc["cr_v_cas_mean"]["min"],
                    df.loc["de_v_cas_const"]["min"],
                    df.loc["fa_va_avg"]["min"],
                )
                limits_fixwing[mdl]["vmaxer"] = max(
                    df.loc["ic_va_avg"]["max"],
                    df.loc["cl_v_cas_const"]["max"],
                    df.loc["cr_v_cas_mean"]["max"],
                    df.loc["de_v_cas_const"]["max"],
                    df.loc["fa_va_avg"]["max"],
                )
                limits_fixwing[mdl]["vminap"] = df.loc["fa_va_avg"]["min"]
                limits_fixwing[mdl]["vmaxap"] = df.loc["fa_va_avg"]["max"]
                limits_fixwing[mdl]["vminld"] = df.loc["ld_v_app"]["min"]
                limits_fixwing[mdl]["vmaxld"] = df.loc["ld_v_app"]["max"]

                limits_fixwing[mdl]["vmo"] = limits_fixwing[mdl]["vmaxer"]
                limits_fixwing[mdl]["mmo"] = df.loc["cr_v_mach_max"]["opt"]

                limits_fixwing[mdl]["hmax"] = df.loc["cr_h_max"]["opt"] * 1000
                limits_fixwing[mdl]["crosscl"] = df.loc["cl_h_mach_const"]["opt"]
                limits_fixwing[mdl]["crossde"] = df.loc["de_h_cas_const"]["opt"]

                limits_fixwing[mdl]["axmax"] = df.loc["to_acc_tof"]["max"]

                limits_fixwing[mdl]["vsmax"] = max(
                    df.loc["ic_vs_avg"]["max"],
                    df.loc["cl_vs_avg_pre_cas"]["max"],
                    df.loc["cl_vs_avg_cas_const"]["max"],
                    df.loc["cl_vs_avg_mach_const"]["max"],
                )

                limits_fixwing[mdl]["vsmin"] = min(
                    df.loc["ic_vs_avg"]["min"],
                    df.loc["de_vs_avg_after_cas"]["min"],
                    df.loc["de_vs_avg_cas_const"]["min"],
                    df.loc["de_vs_avg_mach_const"]["min"],
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
                print(warn)
                bs.scr.echo(warn)

        return limits_rotor
