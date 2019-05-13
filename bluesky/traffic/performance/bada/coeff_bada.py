'''BADA Coefficient file loader

   This module provides access to the performance data contained in the various
   BADA data files.

   The current implementation is based on the official documentation described in
   report: EEC Technical/Scientific Report No. 14/04/24-44. This report can be obtained here:
   https://www.eurocontrol.int/sites/default/files/field_tabs/content/documents/sesar/user-manual-bada-3-12.pdf
'''
from glob import glob
from os import path
import re
from bluesky.tools.fwparser import FixedWidthParser, ParseError

# File formats of BADA data files. Uses fortran-like notation
# Adapted from the BADA manual format lines. (page 61-81 in the BADA manual)
# Skip characters are indicated with nnX
# Variables are indicated with nnF, nnI, and nnS (float, int, string)
syn_format = ['CD, 1X, 1S, 1X, 4S, 3X, 18S, 1X, 25S, 1X, 6S, 2X, 1S']
syn_parser = FixedWidthParser(syn_format)

opf_format = [
              # aircraft type block (1 data line)
              'CD, 3X, 6S, 9X, 1I, 12X, 9S, 17X, 1S',
              # mass block (1 data line)
              'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
              # flight envelope block (1 data line)
              'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
              # aerodynamics block (12 data lines)
              'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 15X, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 15X, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 15X, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 15X, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 15X, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD 50X',
              'CD 50X',
              'CD 50X',
              'CD, 31X, 10F',
              'CD 50X',
              'CD 50X',
              # engine thrust block (3 data lines)
              'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F',
              'CD, 2X, 3X, 10F, 3X, 10F',
              # fuel consumption block (3 data lines)
              'CD, 2X, 3X, 10F, 3X, 10F',
              'CD, 2X, 3X, 10F, 3X, 10F',
              'CD, 5X, 10F',
              # ground movement block (1 data line)
              'CD, 2X, 3X, 10F, 3X, 10F, 3X, 10F, 3X, 10F']
opf_parser = FixedWidthParser(opf_format)

apt_format = [
              # company name (1 line)
              'CD, 2X, 3S, 1X, 2S, 4X, 15S',
              # profiles for climb, cruise, and descent (3 lines)
              'CD, 25X, 3I, 1X, 3I, 1X, 2I, 10X, 3I, 1X, 3I, 1X, 2I, 2X, 2I, 1X, 3I, 1X, 3I',
              'CD, 25X, 3I, 1X, 3I, 1X, 2I, 10X, 3I, 1X, 3I, 1X, 2I, 2X, 2I, 1X, 3I, 1X, 3I',
              'CD, 25X, 3I, 1X, 3I, 1X, 2I, 10X, 3I, 1X, 3I, 1X, 2I, 2X, 2I, 1X, 3I, 1X, 3I']
apf_parser = FixedWidthParser(apt_format)

# The available aircraft are stored by type id in synonyms. The actual coefficient data are stored in accoeffs
synonyms     = dict()
accoeffs     = dict()
release_date = 'Unknown'
bada_version = 'Unknown'


def getCoefficients(actype):
    ''' Get a set of BADA coefficients for the given aircraft type.

        This function looks for the given aircraft type in the synonym list, and
        when successful, retreives the corresponding coefficient set.
        This function returns the synonym object (which contains more detailed
        information about the aircraft type) and the coefficient object'''
    syn = synonyms.get(actype, None)
    if syn is None:
        return False, actype + ' is not found in BADA aircraft database. \
            (Check the file SYNONYM.NEW in your BADA path if you spelled the id correctly)'
    coeff = accoeffs.get(syn.file, None)
    if coeff is None:
        return False, actype + ' exists in BADA synonym database, but corresponding \
            coefficient file (%s) could not be found.' % syn.file

    return syn, coeff


def init(bada_path=''):
    ''' init() loads the available BADA datafiles in the provided directory.'''
    releasefile = path.join(path.normpath(bada_path), 'ReleaseSummary')
    if path.isfile(releasefile):
        global release_date, bada_version
        re_reldate = re.compile('Summary Date:\s+(.+(?<!\s))\s*', re.IGNORECASE)
        re_badaver = re.compile('\s*BADA Release:\s+([\d.]+)\s*', re.IGNORECASE)
        with open(releasefile) as f:
            for line in f:
                if re_reldate.match(line):
                    release_date = re_reldate.findall(line)[0]
                elif re_badaver.match(line):
                    bada_version = re_badaver.findall(line)[0]

                if 'Unknown' not in (release_date, bada_version):
                    break
        print('Found BADA version %s (release date %s)' % (bada_version, release_date))
    else:
        print('No BADA release summary found: can not determine version.')

    synonymfile = path.join(path.normpath(bada_path), 'SYNONYM.NEW')
    if not path.isfile(synonymfile):
        print('SYNONYM.NEW not found in BADA path, could not load BADA.')
        return False

    try:
        data = syn_parser.parse(synonymfile)
    except ParseError as e:
        print('Error reading synonym file {} on line {}'.format(e.fname, e.lineno))
        return False

    for line in data:
        syn = Synonym(line)
        synonyms[syn.accode] = syn
    print('%d aircraft entries loaded' % len(synonyms))

    # Load aircraft coefficient data
    for fname in glob(path.join(path.normpath(bada_path), '*.OPF')):
        ac = ACData()
        try:
            ac.setOPFData(opf_parser.parse(fname))

            if path.isfile(fname[:-4] + '.APF'):
                ac.setAPFData(apf_parser.parse(fname[:-4] + '.APF'))

        except ParseError as e:
            print('Error reading {} on line {}'.format(e.fname, e.lineno))
            ac = None

        if ac:
            accoeffs[ac.actype] = ac
    print('%d unique aircraft coefficient sets loaded' % len(accoeffs))
    return (len(synonyms) > 0 and len(accoeffs) > 0)


class Synonym(object):
    def __init__(self, data):
        self.is_equiv = (data[0] == '*')           # False if model is directly supported in bada, true if supported through equivalent model
        self.accode   = data[1]          # Aircraft code
        self.manufact = data[2]         # Aircraft manufacturer
        self.model    = data[3]        # Aircraft model
        self.file     = data[4]        # Corresponding coefficient filename
        self.icao     = (data[5].upper() == 'Y')  # designator for this aircraft type is in use according to ICAO Doc 8643 [RD2]


class ACData(object):
    # minimum speed coefficients
    CVmin          = 1.3
    CVmin_to       = 1.2

    # reduced power coefficients
    Cred_turboprop = 0.25
    Cred_jet       = 0.15
    Cred_piston    = 0.0

    # value from BADA.gpf file
    gr_acc         = 2.0

    def setOPFData(self, data):
        # aircraft type block:    1 line
        self.actype, self.neng, \
            self.engtype, self.weightcat          = data[0]
        # mass block:             1 line
        self.m_ref, self.m_min, self.m_max, \
            self.m_paymax, self.mass_grad         = data[1]
        # flight envelope block:  1 line
        self.VMO, self.MMO, self.h_MO, \
            self.h_max, self.temp_grad            = data[2]
        # aerodynamics block:     12 lines
        self.S, self.Clbo, self.k, self.CM16      = data[3]
        self.Vstall_cr, self.CD0_cr, self.CD2_cr  = data[4]
        self.Vstall_ic, self.CD0_ic, self.CD2_ic  = data[5]
        self.Vstall_to, self.CD0_to, self.CD2_to  = data[6]
        self.Vstall_ap, self.CD0_ap, self.CD2_ap  = data[7]
        self.Vstall_ld, self.CD0_ld, self.CD2_ld  = data[8]
        self.CD0_gear                             = data[12][0]
        # engine thrust block:    3 lines
        self.CTC                                  = data[15]
        self.CTdes_low, self.CTdes_high, \
            self.Hp_des, self.CTdes_app, \
            self.CTdes_land                       = data[16]
        self.Vdes_ref, self.Mdes_ref              = data[17]
        # fuel consumption block: 3 lines
        self.Cf1, self.Cf2                        = data[18]
        self.Cf3, self.Cf4                        = data[19]
        self.Cf_cruise                            = data[20][0]
        # ground movements block: 1 line
        self.TOL, self.LDL, \
            self.wingspan, self.length            = data[21]

        # Set minimum operating speeds based on stall speeds and minspeed coefficients
        self.vmto = self.Vstall_to * self.CVmin_to
        self.vmic = self.Vstall_ic * self.CVmin
        self.vmcr = self.Vstall_cr * self.CVmin
        self.vmap = self.Vstall_ap * self.CVmin
        self.vmld = self.Vstall_ld * self.CVmin

    def setAPFData(self, data):
        # Minimum, average, and high reference speeds for climb, cruise,
        # and descent. xx1=low mass, xx2=high mass
        self.CAScl1, self.CAScl2, self.Mcl, \
            self.CAScr1, self.CAScr2, self.Mcr, \
            self.Mdes, self.CASdes2, self.CASdes1 = list(zip(*data[1:]))  # swap rows/columns

        # Mach numbers are multiplied by 100 in the BADA files
        self.Mcl  = [m / 100.0 for m in self.Mcl]
        self.Mcr  = [m / 100.0 for m in self.Mcr]
        self.Mdes = [m / 100.0 for m in self.Mdes]
