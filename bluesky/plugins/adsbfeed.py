"""BlueSky ADS-B datafeed plugin. Reads the feed from a Mode-S Beast server,
and visualizes traffic in BlueSky."""

import time

from bluesky import settings, stack, traf
from bluesky.tools import aero
from bluesky.tools.network import TcpSocket

## Default settings
# Mode-S / ADS-B server hostname/ip, and server port
settings.set_variable_defaults(modeS_host="", modeS_port=0)

# Global data
reader = None


### Initialization function of the adsbfeed plugin.
def init_plugin():
    # Initialize Modesbeast reader
    global reader
    reader = Modesbeast()

    # Configuration parameters
    config = {
        "plugin_name": "DATAFEED",
        "plugin_type": "sim",
        "update_interval": 0.0,
        "preupdate": reader.update,
    }

    stackfunctions = {
        "DATAFEED": [
            "DATAFEED [ON/OFF]",
            "[onoff]",
            reader.toggle,
            "Select an ADS-B data source for traffic",
        ]
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


class Modesbeast(TcpSocket):
    def __init__(self):
        super().__init__()
        self.acpool = {}
        self.buffer = b""
        self.default_ac_mdl = "B738"

    def processData(self, data):
        self.buffer += data

        if len(self.buffer) > 2048:
            # process the buffer until the last divider <esc> 0x1a
            # then, reset the buffer with the remainder

            bfdata = self.buffer
            n = (len(bfdata) - 1) - bfdata[::-1].index(0x1A)
            data = bfdata[: n - 1]
            self.buffer = self.buffer[n:]

            messages = self.read_mode_s(data)

            if not messages:
                return

            for msg, ts in messages:
                self.read_message(msg, ts)
        return

    def read_mode_s(self, data):
        """
        <esc> "1" : 6 byte MLAT timestamp, 1 byte signal level,
            2 byte Mode-AC
        <esc> "2" : 6 byte MLAT timestamp, 1 byte signal level,
            7 byte Mode-S short frame
        <esc> "3" : 6 byte MLAT timestamp, 1 byte signal level,
            14 byte Mode-S long frame
        <esc> "4" : 6 byte MLAT timestamp, status data, DIP switch
            configuration settings (not on Mode-S Beast classic)
        <esc><esc>: true 0x1a
        <esc> is 0x1a, and "1", "2" and "3" are 0x31, 0x32 and 0x33

        timestamp:
        wiki.modesbeast.com/Radarcape:Firmware_Versions#The_GPS_timestamp
        """

        # split raw data into chunks
        chunks = []
        separator = 0x1A
        piece = []
        for d in data:
            if d == separator:
                # shortest msgs are 11 chars
                if len(piece) > 10:
                    chunks.append(piece)
                piece = []
            piece.append(d)

        # extract messages
        messages = []
        for cnk in chunks:
            msgtype = cnk[1]

            # Mode-S Short Message, 7 byte
            if msgtype == 0x32:
                msg = "".join("%02X" % i for i in cnk[9:16])

            # Mode-S Short Message, 14 byte
            elif msgtype == 0x33:
                msg = "".join("%02X" % i for i in cnk[9:23])

            # Other message tupe
            else:
                continue

            ts = time.time()

            messages.append([msg, ts])
        return messages

    def read_message(self, msg, ts):
        """
        Process ADSB messages
        """

        if len(msg) < 28:
            return

        df = Decoder.get_df(msg)

        if df == 17:
            addr = Decoder.get_icao_addr(msg)
            tc = Decoder.get_tc(msg)

            if tc >= 1 and tc <= 4:
                # aircraft identification
                callsign = Decoder.get_callsign(msg)
                self.update_callsign(addr, callsign)
            if tc >= 9 and tc <= 18:
                # airbone postion frame
                alt = Decoder.get_alt(msg)
                oe = Decoder.get_oe_flag(msg)  # odd or even frame
                cprlat = Decoder.get_cprlat(msg)
                cprlon = Decoder.get_cprlon(msg)
                self.update_cprpos(addr, oe, ts, alt, cprlat, cprlon)
            elif tc == 19:  # airbone velocity frame
                sh = Decoder.get_speed_heading(msg)
                if len(sh) == 2:
                    spd = sh[0]
                    hdg = sh[1]
                    self.update_spd_hdg(addr, spd, hdg)
        return

    def update_cprpos(self, addr, oe, ts, alt, cprlat, cprlon):
        if addr in self.acpool:
            ac = self.acpool[addr]
        else:
            ac = {}

        ac["alt"] = alt
        if oe == "1":  # odd frame cpr position
            ac["cprlat1"] = cprlat
            ac["cprlon1"] = cprlon
            ac["t1"] = ts

        if oe == "0":  # even frame cpr position
            ac["cprlat0"] = cprlat
            ac["cprlon0"] = cprlon
            ac["t0"] = ts

        ac["ts"] = time.time()

        self.acpool[addr] = ac
        return

    def update_spd_hdg(self, addr, spd, hdg):
        if addr in self.acpool:
            ac = self.acpool[addr]
        else:
            ac = {}

        ac["speed"] = spd
        ac["heading"] = hdg
        ac["ts"] = time.time()

        self.acpool[addr] = ac
        return

    def update_callsign(self, addr, callsign):
        if addr not in self.acpool:
            self.acpool[addr] = {}

        self.acpool[addr]["callsign"] = callsign
        return

    def update_all_ac_postition(self):
        keys = ("cprlat0", "cprlat1", "cprlon0", "cprlon1")
        for addr, ac in list(self.acpool.items()):
            # check if all needed keys are in dict
            if set(keys).issubset(ac):
                pos = Decoder.cpr2position(
                    ac["cprlat0"],
                    ac["cprlat1"],
                    ac["cprlon0"],
                    ac["cprlon1"],
                    ac["t0"],
                    ac["t1"],
                )

                # update positions of all aircrafts in the list
                if pos:
                    self.acpool[addr]["lat"] = pos[0]
                    self.acpool[addr]["lon"] = pos[1]
        return

    def stack_all_commands(self):
        """create and stack command"""
        params = ("lat", "lon", "alt", "speed", "heading", "callsign")
        for i, d in list(self.acpool.items()):
            # check if all needed keys are in dict
            if set(params).issubset(d):
                acid = d["callsign"]
                # check is aircraft is already beening displayed
                if traf.id2idx(acid) < 0:
                    mdl = self.default_ac_mdl
                    v = aero.tas2cas(d["speed"], d["alt"] * aero.ft)
                    cmdstr = "CRE %s, %s, %f, %f, %f, %d, %d" % (
                        acid,
                        mdl,
                        d["lat"],
                        d["lon"],
                        d["heading"],
                        d["alt"],
                        v,
                    )
                    stack.stack(cmdstr)
                else:
                    cmdstr = "MOVE %s, %f, %f, %d" % (
                        acid,
                        d["lat"],
                        d["lon"],
                        d["alt"],
                    )
                    stack.stack(cmdstr)

                    cmdstr = "HDG %s, %f" % (acid, d["heading"])
                    stack.stack(cmdstr)

                    v_cas = aero.tas2cas(d["speed"], d["alt"] * aero.ft)
                    cmdstr = "SPD %s, %f" % (acid, v_cas)
                    stack.stack(cmdstr)
        return

    def remove_outdated_ac(self):
        """House keeping, remove old entries (offline > 100s)"""
        for addr, ac in list(self.acpool.items()):
            if "ts" in ac:
                # threshold, remove ac after 90 seconds of no-seen
                if (int(time.time()) - ac["ts"]) > 100:
                    del self.acpool[addr]
                    # remove from sim traffic
                    if "callsign" in ac:
                        stack.stack("DEL %s" % ac["callsign"])
        return

    def debug(self):
        addlist = str.join(", ", self.acpool.keys())
        print(addlist)
        print("")
        print("total count: %d" % len(self.acpool.keys()))
        return

    def update(self):
        if self.isConnected():
            # self.debug()
            self.remove_outdated_ac()
            self.update_all_ac_postition()
            self.stack_all_commands()

    def toggle(self, flag=None):
        if flag is None:
            if self.isConnected():
                return True, "Connected to %s on port %s" % (
                    settings.modeS_host,
                    settings.modeS_port,
                )
            else:
                return True, "Not connected"
        elif flag:
            self.connectToHost(settings.modeS_host, settings.modeS_port)
            stack.stack("OP")
            return True, "Connecting to %s on port %s" % (
                settings.modeS_host,
                settings.modeS_port,
            )
        else:
            self.disconnectFromHost()

        return True


class Decoder:
    import numpy as np

    # fmt: off
    MODES_CHECKSUM_TABLE = [
        0x3935ea, 0x1c9af5, 0xf1b77e, 0x78dbbf,
        0xc397db, 0x9e31e9, 0xb0e2f0, 0x587178,
        0x2c38bc, 0x161c5e, 0x0b0e2f, 0xfa7d13,
        0x82c48d, 0xbe9842, 0x5f4c21, 0xd05c14,
        0x682e0a, 0x341705, 0xe5f186, 0x72f8c3,
        0xc68665, 0x9cb936, 0x4e5c9b, 0xd8d449,
        0x939020, 0x49c810, 0x24e408, 0x127204,
        0x093902, 0x049c81, 0xfdb444, 0x7eda22,
        0x3f6d11, 0xe04c8c, 0x702646, 0x381323,
        0xe3f395, 0x8e03ce, 0x4701e7, 0xdc7af7,
        0x91c77f, 0xb719bb, 0xa476d9, 0xadc168,
        0x56e0b4, 0x2b705a, 0x15b82d, 0xf52612,
        0x7a9309, 0xc2b380, 0x6159c0, 0x30ace0,
        0x185670, 0x0c2b38, 0x06159c, 0x030ace,
        0x018567, 0xff38b7, 0x80665f, 0xbfc92b,
        0xa01e91, 0xaff54c, 0x57faa6, 0x2bfd53,
        0xea04ad, 0x8af852, 0x457c29, 0xdd4410,
        0x6ea208, 0x375104, 0x1ba882, 0x0dd441,
        0xf91024, 0x7c8812, 0x3e4409, 0xe0d800,
        0x706c00, 0x383600, 0x1c1b00, 0x0e0d80,
        0x0706c0, 0x038360, 0x01c1b0, 0x00e0d8,
        0x00706c, 0x003836, 0x001c1b, 0xfff409,
        0x000000, 0x000000, 0x000000, 0x000000,
        0x000000, 0x000000, 0x000000, 0x000000,
        0x000000, 0x000000, 0x000000, 0x000000,
        0x000000, 0x000000, 0x000000, 0x000000,
        0x000000, 0x000000, 0x000000, 0x000000,
        0x000000, 0x000000, 0x000000, 0x000000
    ]
    # fmt: on

    @staticmethod
    def hex2bin(hexstr):
        """Convert a hexadecimal string to binary string, with zero fillings."""
        length = len(hexstr) * 4
        msgbin = bin(int(hexstr, 16))[2:]
        return msgbin.zfill(length)

    @staticmethod
    def bin2int(binstr):
        return int(binstr, 2)

    @staticmethod
    def hex2int(hexstr):
        return int(hexstr, 16)

    @staticmethod
    def checksum(msg):
        if len(msg) == 28:
            offset = 0
        elif len(msg) == 14:
            offset = 112 - 56
        else:
            return False
        msgbin = Decoder.hex2bin(msg)
        chk = int(msg[22:28], 16)
        crc = 0
        for i in range(len(msgbin)):
            if int(msgbin[i]):
                crc ^= Decoder.MODES_CHECKSUM_TABLE[i + offset]
        return crc == chk

    @staticmethod
    def get_df(msg):
        msgbin = Decoder.hex2bin(msg)
        return Decoder.bin2int(msgbin[0:5])

    @staticmethod
    def get_ca(msg):
        msgbin = Decoder.hex2bin(msg)
        return Decoder.bin2int(msgbin[5:8])

    @staticmethod
    def get_icao_addr(msg):
        return msg[2:8]

    @staticmethod
    def get_tc(msg):
        msgbin = Decoder.hex2bin(msg)
        return Decoder.bin2int(msgbin[32:37])

    @staticmethod
    def get_oe_flag(msg):
        msgbin = Decoder.hex2bin(msg)
        return msgbin[53]

    @staticmethod
    def get_alt(msg):
        msgbin = Decoder.hex2bin(msg)
        q = msgbin[47]
        if q == "1":
            n = Decoder.bin2int(msgbin[40:47] + msgbin[48:52])
            return n * 25 - 1000
        else:
            return None

    @staticmethod
    def get_cprlat(msg):
        msgbin = Decoder.hex2bin(msg)
        return Decoder.bin2int(msgbin[54:71])

    @staticmethod
    def get_cprlon(msg):
        msgbin = Decoder.hex2bin(msg)
        return Decoder.bin2int(msgbin[71:88])

    @staticmethod
    def get_speed_heading(msg):
        msgbin = Decoder.hex2bin(msg)
        v_ew_dir = Decoder.bin2int(msgbin[45])
        v_ew = Decoder.bin2int(msgbin[46:56])
        v_ns_dir = Decoder.bin2int(msgbin[56])
        v_ns = Decoder.bin2int(msgbin[57:67])
        v_ew = -v_ew if v_ew_dir else v_ew
        v_ns = -v_ns if v_ns_dir else v_ns
        speed = Decoder.np.sqrt(v_ns * v_ns + v_ew * v_ew)
        heading = Decoder.np.arctan2(v_ew, v_ns)
        heading = heading * 360.0 / (2 * Decoder.np.pi)
        heading = heading if heading >= 0 else heading + 360
        return [speed, heading]

    @staticmethod
    def get_callsign(msg):
        chars = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######"
        msgbin = Decoder.hex2bin(msg)
        csbin = msgbin[40:96]
        cs = ""
        for i in range(0, 48, 6):
            cs += chars[Decoder.bin2int(csbin[i : i + 6])]
        return cs.replace("_", "").replace("#", "")

    @staticmethod
    def get_position(msg0, msg1, t0, t1):
        cprlat0 = Decoder.get_cprlat(msg0)
        cprlat1 = Decoder.get_cprlat(msg1)
        cprlon0 = Decoder.get_cprlon(msg0)
        cprlon1 = Decoder.get_cprlon(msg1)
        return Decoder.cpr2position(cprlat0, cprlat1, cprlon0, cprlon1, t0, t1)

    @staticmethod
    def cpr2position(cprlat0, cprlat1, cprlon0, cprlon1, t0, t1):
        cprlat_even = cprlat0 / 131072.0
        cprlat_odd = cprlat1 / 131072.0
        cprlon_even = cprlon0 / 131072.0
        cprlon_odd = cprlon1 / 131072.0
        air_d_lat_even = 360.0 / 60
        air_d_lat_odd = 360.0 / 59
        j = int(59 * cprlat_even - 60 * cprlat_odd + 0.5)
        lat_even = air_d_lat_even * ((j % 60) + cprlat_even)
        lat_odd = air_d_lat_odd * ((j % 59) + cprlat_odd)
        if lat_even >= 270:
            lat_even -= 360
        if lat_odd >= 270:
            lat_odd -= 360
        if Decoder.cprNL(lat_even) != Decoder.cprNL(lat_odd):
            return None
        if t0 > t1:
            ni = Decoder.cprN(lat_even, 0)
            m = Decoder.np.floor(
                cprlon_even * (Decoder.cprNL(lat_even) - 1)
                - cprlon_odd * Decoder.cprNL(lat_even)
                + 0.5
            )
            lon = (360.0 / ni) * ((m % ni) + cprlon_even)
            lat = lat_even
        else:
            ni = Decoder.cprN(lat_odd, 1)
            m = Decoder.np.floor(
                cprlon_even * (Decoder.cprNL(lat_odd) - 1)
                - cprlon_odd * Decoder.cprNL(lat_odd)
                + 0.5
            )
            lon = (360.0 / ni) * ((m % ni) + cprlon_odd)
            lat = lat_odd
        if lon > 180:
            lon -= 360
        return [lat, lon]

    @staticmethod
    def cprN(lat, is_odd):
        nl_val = Decoder.cprNL(lat) - is_odd
        return nl_val if nl_val > 1 else 1

    @staticmethod
    def cprNL(lat):
        try:
            nz = 60
            a = 1 - Decoder.np.cos(Decoder.np.pi * 2 / nz)
            b = Decoder.np.cos(Decoder.np.pi / 180.0 * abs(lat)) ** 2
            nl_val = 2 * Decoder.np.pi / (Decoder.np.acos(1 - a / b))
            return int(nl_val)
        except Exception:
            return 1
