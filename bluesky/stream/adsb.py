""" 
Stream data from a TCP server providing datafeed of ADS-B messages

"""

import os
import socket
import time
import threading
import math


class Decoder:
    """ 
    Decoder for ADS-S messages

    """

    def hex2bin(self, hexstr):
        """Convert a hexdecimal string to binary string, with zero fillings"""
        length = len(hexstr) * 4
        binstr = bin(int(hexstr, 16))[2:]
        while ((len(binstr)) < length):
            binstr = '0' + binstr
        return binstr

    def bin2int(self, msgbin):
        """Convert a binary string to decimal integer"""
        return int(msgbin, 2)

    def get_df(self, msg):
        """Decode Downlink Format vaule, message bits 1 to 5."""
        msgbin = hex2bin(msg)
        return bin2int( msgbin[0:5] )

    def get_ca(self, msg):        
        """Decode CA vaule, message bits: 6 to 8."""
        msgbin = hex2bin(msg)
        return bin2int( msgbin[5:8] )

    def get_icao_addr(self, msg):
        """Get the ICAO 24 message bits address, bytes 3 to 8. """
        return msg[2:8]

    def get_tc(self, msg):
        """Get Type Code, message bits 33 to 37 """
        msgbin = hex2bin(msg)
        return bin2int(msgbin[32:37])

    def get_oe_flag(self, msg):
        """Check the odd/even flag; message bit 54. 0 for even, 1 for odd."""
        msgbin = hex2bin(msg)
        return msgbin[53]

    def get_alt(self, msg):
        """Calculate the altitude from the message. Bit 41 to 52, Q-bit at 48"""
        msgbin = hex2bin(msg)
        q = msgbin[47]
        if q:
            n = bin2int(msgbin[40:47]+msgbin[48:52])
            alt = n * 25 - 1000
            return alt
        else:
            return None

    def get_cprlat(self, msg):
        msgbin = hex2bin(msg)
        return bin2int(msgbin[54:71])

    def get_cprlon(self, msg):
        msgbin = hex2bin(msg)
        return bin2int(msgbin[71:88])

    def get_position(self, msg0, msg1, t0, t1):
        cprlat0 = self.get_cprlat(msg0)
        cprlat1 = self.get_cprlat(msg1)
        cprlon0 = self.get_cprlon(msg0)
        cprlon1 = self.get_cprlon(msg1)
        return self.cpr2position(cprlat0, cprlat1, cprlon0, cprlon1, t0, t1)

    def cpr2position(self, cprlat0, cprlat1, cprlon0, cprlon1, t0, t1):
        """
        Calculation of position from CPR format
        Inspired by: 
        http://www.lll.lu/~edward/edward/adsb/DecodingADSBposition.html.
         
        131072 is 2^17 since CPR latitude and longitude are encoded 
        in 17 message bits.
        """

        cprlat_even = cprlat0 / 131072.0
        cprlat_odd  = cprlat1 / 131072.0
        cprlon_even = cprlon0 / 131072.0
        cprlon_odd  = cprlon0 / 131072.0

        air_d_lat_even = 360.0 / 60 
        air_d_lat_odd = 360.0 / 59 

        # compute latitude index 'j'
        j = int(59 * cprlat_even - 60 * cprlat_odd + 0.5)

        lat_even = float(air_d_lat_even * (j % 60 + cprlat_even))
        lat_odd  = float(air_d_lat_odd  * (j % 59 + cprlat_odd))

        if lat_even >= 270:
            lat_even = lat_even - 360

        if lat_odd >= 270:
            lat_odd = lat_odd - 360

        # check if both are in the same latidude zone, exit if not
        if self._cprNL(lat_even) != self._cprNL(lat_odd):
          return None

        # compute ni, longitude index m, and longitude
        if (t0 > t1):
          ni = self._cprN(lat_even, 0)
          m = math.floor( cprlon_even * (self._cprNL(lat_even)-1) - cprlon_odd * self._cprNL(lat_even) + 0.5 ) 
          lon = (360.0 / ni) * (m % ni + cprlon_even)
          lat = lat_even
        else:
          ni = self._cprN(lat_odd, 1)
          m = math.floor( cprlon_even * (self._cprNL(lat_odd)-1) - cprlon_odd * self._cprNL(lat_odd) + 0.5 ) 
          lon = (360.0 / ni) * (m % ni + cprlon_odd)
          lat = lat_odd

        if lon > 180:
            lon = lon - 360 

        return [lat, lon]


    def get_speed_heading(self, msg):
        """Calculate the speed and heading."""
        msgbin = hex2bin(msg)

        v_ew_dir = bin2int(msgbin[45])
        v_ew     = bin2int(msgbin[46:56])       # east-west velocity

        v_ns_dir = bin2int(msgbin[56])
        v_ns     = bin2int(msgbin[57:67])       # north-south velocity

        v_ew = -1*v_ew if v_ew_dir else v_ew
        v_ns = -1*v_ns if v_ns_dir else v_ns

        # vr       = bin2int(msgbin[68:77])       # vertical rate
        # vr_dir   = bin2int(msgbin[77])

        speed = math.sqrt(v_ns*v_ns + v_ew*v_ew)    # unit in kts

        heading = math.atan2(v_ew, v_ns)
        heading = heading * 360.0 / (2 * math.pi)   #convert to degrees
        heading = heading if heading >= 0 else heading + 360     # no negative val
        return [speed, heading]

    def get_callsign(self, msg):
        """Decode aircraft identification, aka. Callsign"""
        
        charset = '#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######'
        msgbin = hex2bin(msg)
        csbin = msgbin[40:96]

        cs = ''
        cs += charset[ bin2int(csbin[0:6]) ]
        cs += charset[ bin2int(csbin[6:12]) ]
        cs += charset[ bin2int(csbin[12:18]) ]
        cs += charset[ bin2int(csbin[18:24]) ]
        cs += charset[ bin2int(csbin[24:30]) ]
        cs += charset[ bin2int(csbin[30:36]) ]
        cs += charset[ bin2int(csbin[36:42]) ]
        cs += charset[ bin2int(csbin[42:48]) ]

        # clean string, remove spaces and marks, if any.
        cs = cs.replace('_', '')
        cs = cs.replace('#', '')
        return cs


    def _cprN (self, lat, isodd):
        nl = self._cprNL(lat) - isodd
        return nl if nl > 1 else 1


    def _cprNL(self, lat):
        """Lookup table to convert the latitude to index. """
        if lat < 0 : lat = -lat             # Table is simmetric about the equator.
        if lat < 10.47047130 : return 59
        if lat < 14.82817437 : return 58
        if lat < 18.18626357 : return 57
        if lat < 21.02939493 : return 56
        if lat < 23.54504487 : return 55
        if lat < 25.82924707 : return 54
        if lat < 27.93898710 : return 53
        if lat < 29.91135686 : return 52
        if lat < 31.77209708 : return 51
        if lat < 33.53993436 : return 50
        if lat < 35.22899598 : return 49
        if lat < 36.85025108 : return 48
        if lat < 38.41241892 : return 47
        if lat < 39.92256684 : return 46
        if lat < 41.38651832 : return 45
        if lat < 42.80914012 : return 44
        if lat < 44.19454951 : return 43
        if lat < 45.54626723 : return 42
        if lat < 46.86733252 : return 41
        if lat < 48.16039128 : return 40
        if lat < 49.42776439 : return 39
        if lat < 50.67150166 : return 38
        if lat < 51.89342469 : return 37
        if lat < 53.09516153 : return 36
        if lat < 54.27817472 : return 35
        if lat < 55.44378444 : return 34
        if lat < 56.59318756 : return 33
        if lat < 57.72747354 : return 32
        if lat < 58.84763776 : return 31
        if lat < 59.95459277 : return 30
        if lat < 61.04917774 : return 29
        if lat < 62.13216659 : return 28
        if lat < 63.20427479 : return 27
        if lat < 64.26616523 : return 26
        if lat < 65.31845310 : return 25
        if lat < 66.36171008 : return 24
        if lat < 67.39646774 : return 23
        if lat < 68.42322022 : return 22
        if lat < 69.44242631 : return 21
        if lat < 70.45451075 : return 20
        if lat < 71.45986473 : return 19
        if lat < 72.45884545 : return 18
        if lat < 73.45177442 : return 17
        if lat < 74.43893416 : return 16
        if lat < 75.42056257 : return 15
        if lat < 76.39684391 : return 14
        if lat < 77.36789461 : return 13
        if lat < 78.33374083 : return 12
        if lat < 79.29428225 : return 11
        if lat < 80.24923213 : return 10
        if lat < 81.19801349 : return 9
        if lat < 82.13956981 : return 8
        if lat < 83.07199445 : return 7
        if lat < 83.99173563 : return 6
        if lat < 84.89166191 : return 5
        if lat < 85.75541621 : return 4
        if lat < 86.53536998 : return 3
        if lat < 87.00000000 : return 2
        else : return 1


class TcpClient:
    """A TCP Client receving message from server, analysing the data, and """
    def __init__(self, datafeed, serverip, port):
        self.datafeed = datafeed
        self.serverip = serverip
        self.port = port
        self.buffer_size = 1024
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receiver_stop_flag = True

        print "connecting to tcp server..."
        try:
            self.sock.settimeout(10)    # 10 second timeout
            self.sock.connect((self.serverip, self.port))       # connecting
        except socket.error, exc:
            print "Socket connection error: %s" % exc
            raise RuntimeError("Cannot connect to server.")
            return

        self.receiver_thread = threading.Thread(target=self.receiver)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def start(self):
        print "starting receiving data..."
        self.receiver_stop_flag = False
        return

    def stop(self):
        print "stopping tcp client..."
        self.receiver_stop_flag = True
        return


    def read_message(self, msgtype, data, datalen):
        ''' Process the message that received from remote TCP server '''  
        if msgtype == 3:
            # get time from the Aircraft
            t_sec_ac = 0
            t_sec_ac |= data[0] << 24
            t_sec_ac |= data[1] << 16
            t_sec_ac |= data[2] << 8
            t_sec_ac |= data[3]

            # get receiver timestamp
            t_sec_receiver = 0
            t_sec_receiver |= data[4] << 24
            t_sec_receiver |= data[5] << 16
            t_sec_receiver |= data[6] << 8
            t_sec_receiver |= data[7]

            # ignor receiver id, data[8]
            # ignor data[9], for now

            mlat = 0
            mlat |= data[10] << 24
            mlat |= data[11] << 16
            mlat |= data[12] << 8
            mlat |= data[13]

            power = 0
            power |= data[14] << 8
            power |= data[15]
            power &= 0x3FF  
            power = power >> 6

            # process msg in the data frame
            msg = ''
            msglen = 14     # type 3 data length is 14
            msgstart = 16
            msgend = msgstart + msglen
            for i in data[msgstart : msgend] :
                msg += "%02X" % i

            # print "Type:%d | Len:%d | AcTime:%d | ReceiverTime:%d | Power:%d | MSG: %s"  % \
            #     (msgtype, datalen, t_sec_ac, t_sec_receiver, power, msg)

            datafeed.process_message(msg)
            return


    def receiver(self):
        ''' Connet to TCP server and receving data streams '''  
        while True:
            if not self.receiver_stop_flag:
                try:
                    raw_data = self.sock.recv(self.buffer_size)
                    if raw_data == b'':
                        raise RuntimeError("socket connection broken")

                    # print ''.join(x.encode('hex') for x in raw_data)

                    data = [ord(i) for i in raw_data]    # covert the char to int

                    if data[0] == 27:                    # looking for ADS-B data, start with "0x1B"
                        datatype = int(raw_data[1])
                        datalen = data[2]<<8 | data[3]
                        self.read_message(datatype, data[4:], datalen)
                except socket.error, exc:
                    print "Socket reading error: %s" % exc
            else:
                time.sleep(0.5)
        return


class DataFeed:
    """
    Reading data from ADS-B TCP server, decoding, update online aircrafts status

    Parameters
    ----------

    tcpclient : object of TcpClient a tcp client to receive data from remote server

    Attributes
    ----------
    aircrafts : dict
    a list of online aircrafts with their parameters
    each item of the dict can be read
    ac : { acid | lat | lon | alt | speed | heading | time }
    """

    # def __init__(self, tcpclient):
    #     self.aircrafts = {}    # a dict store all the aircraft absb data
    #     self.tcpclient = tcpclient
    #     # self.tcpclient = CTcpClient.TcpClient(self, '145.94.54.54', 10001)
    #     self.tcpclient.start()
    #     return

    def __init__(self, serverip, port):
        self.decoder = Decoder()
        self.tcpclient =  False
        self.serverip = serverip
        self.port = port
        self.aircrafts = {}    # a dict store all the aircraft status data
        self.routine_stop_flag = True
        self.routine_thread = threading.Thread(target=self.routine)
        self.routine_thread.daemon = True
        self.console_print = False
        return

    def connect(self):
        """Initialize TCP connection"""
        if not self.tcpclient:
            try:
                self.tcpclient = TcpClient(self, self.serverip, self.port)
            except RuntimeError, e:
                print "Error, can not connect to server.."
                self.tcpclient = False
                return

        if not self.routine_thread.isAlive():
            self.routine_thread.start()

        self.start()
        return
    
    def start(self):
        """Starting ADS-B datafeed"""
        self.routine_stop_flag = False
        
        if self.tcpclient:      # check if tcpclient exists
            self.tcpclient.start()
        else:
            print "Client not ready, connect to the server first.."
        return

    def stop(self):
        """Terminate the routine and tcpclient"""
        self.routine_stop_flag = True

        if self.tcpclient:      # check if tcpclient exists
            self.tcpclient.stop()
        else:
            print "Client not ready, connect to the server first.."
        return

    def set_console_print(self, flag):
        """Set the console printing flag"""
        self.console_print = flag
    

    def process_message(msg):
        # do some checking to see if the msg is useful for position decoding
        df = self.decoder.get_df(msg)
        tc = self.decoder.get_tc(msg)
        ca = self.decoder.get_ca(msg)

        if df==17:
            addr = self.decoder.get_ac_icao_addr(msg)

            if tc>=1 and tc<=4:                     # aircraft identification
                callsign = self.decoder.get_callsign(msg)
                self.datafeed.update_callsign_data(addr, callsign)
            if tc>=9 and tc<=18:                    # airbone postion frame
                alt = self.decoder.get_alt(msg)
                oe = self.decoder.get_oe_flag(msg)  # odd or even frame
                cprlat = self.decoder.get_cprlat(msg)
                cprlon = self.decoder.get_cprlon(msg)
                dataset = {'addr':addr, 'oe':oe, 'time':t_sec_ac, 'alt':alt,
                        'cprlat':cprlat, 'cprlon':cprlon}
                # print dataset
                self.push_cprpos_data(dataset)
            elif tc==19 and ca>=1 and ca<=4:        # airbone velocity frame
                if ca==1 or ca==2:
                    sh = self.decoder.get_speed_heading(msg)
                    if sh:
                        dataset = {'addr':addr, 'speed':sh[0], 'heading':sh[1]}
                        self.datafeed.push_speed_heading_data(dataset)

    def push_cprpos_data(self, data):
        """Add an adsb cpr position data to the pool"""

        if data['addr'] in self.aircrafts:
            acdata = self.aircrafts[data['addr']]
        else:
            acdata = {}

        acdata['alt'] = data['alt']
        if data['oe'] == '1':       # odd frame cpr position
            acdata['cprlat_o'] = data['cprlat']
            acdata['cprlon_o'] = data['cprlon']
        if data['oe'] == '0':       # even frame cpr position
            acdata['cprlat_e'] = data['cprlat']
            acdata['cprlon_e'] = data['cprlon']
        acdata['time'] = data['time']

        self.aircrafts[data['addr']] = acdata
        # self.routine()
        return

    def push_speed_heading_data(self, data):
        """Add speed and heading data to the pool"""

        if data['addr'] in self.aircrafts:
            acdata = self.aircrafts[data['addr']]
        else:
            acdata = {}

        acdata['speed'] = data['speed']
        acdata['heading'] = data['heading']

        self.aircrafts[data['addr']] = acdata
        # self.routine()
        return

    def update_callsign_data(self, addr, callsign):
        """Update callsign when it is decoded"""
        if addr in self.aircrafts:
            self.aircrafts[addr]['acid'] = callsign
        return

    def update_all_ac_postition(self):
        """Loop through the aircraft pool, calculate and update all positions"""
        keys = ('cprlat_e', 'cprlon_e', 'cprlat_o', 'cprlon_o')
        for ac, acdata in self.aircrafts.items():
            # check if all needed keys are in dict
            if set(keys).issubset(acdata):
                pos = self.decoder.decodeCPR(acdata['cprlat_e'], acdata['cprlon_e'], 
                    acdata['cprlat_o'], acdata['cprlon_o'])
                
                # update positions of all aircrafts in the list
                if pos:
                    self.aircrafts[ac]['lat'] = pos[0]
                    self.aircrafts[ac]['lon'] = pos[1]
        return

    # def stack_all_commands(self):
    #     """Generate and stack command to command controller"""
    #     keys = ('acid', 'lat', 'lon', 'alt', 'speed', 'heading')
    #     for ac, acdata in self.aircrafts.items():
    #         # check if all needed keys are in dict
    #         if set(keys).issubset(acdata):
    #             # acid = ac
    #             acid = acdata['acid']
    #             # check is aircraft is already beening displayed
    #             if(self.tmx.traf.id2idx(acid) < 0):
    #                 cmdstr = 'CRE %s, %s, %f, %f, %f, %d, %d' % \
    #                     (acid, 'B737', acdata['lat'], acdata['lon'], \
    #                         acdata['heading'], acdata['alt'], acdata['speed'])
    #                 self.tmx.cmd.stack(cmdstr)
    #             else:
    #                 cmdstr = 'MOVE %s, %f, %f, %d' % \
    #                     (acid, acdata['lat'], acdata['lon'], acdata['alt'])
    #                 self.tmx.cmd.stack(cmdstr)

    #                 cmdstr = 'HDG %s, %f' % (acid,  acdata['heading'])
    #                 self.tmx.cmd.stack(cmdstr)

    #                 cmdstr = 'SPD %s, %f' % (acid,  acdata['speed'])
    #                 self.tmx.cmd.stack(cmdstr)
    #     return

    def print_acdata(self):
        """Print aircraft status data to console"""
        os.system('cls' if os.name == 'nt' else 'clear')    # clear the console to print up-to-date table info

        print '---- Aircraft in sight (%s) ----' % str(int(time.time()))
        for ac, acdata in self.aircrafts.items():
            acid =  acdata['acid'] if 'acid' in acdata else 'n/a'
            lat =  acdata['lat'] if 'lat' in acdata else 0
            lon =  acdata['lon'] if 'lon' in acdata else 0
            alt =  acdata['alt'] if 'alt' in acdata else 0
            spd =  acdata['speed'] if 'speed' in acdata else 0
            hdg =  acdata['heading'] if 'heading' in acdata else 0
            ts =  acdata['time'] if 'time' in acdata else 0

            print "AC: %s | ID: %s | LAT: %f | LON: %f | ALT: %d | SPD: %d | HDG: %d | TIME: %d" % \
                (ac, acid, lat, lon, alt, spd, hdg, ts)
        return

    def remove_outdated_ac(self):
        """House keeping, remove old entries that are not updated for more than 100 second"""
        for ac, acdata in self.aircrafts.items():
            if 'time' in acdata:
                if (int(time.time()) - acdata['time']) > 90:    
                    # if 'acid' in acdata:    # remove from tmx screen if been presented
                    #     cmdstr = 'DEL %s' % acdata['acid']
                    #     self.tmx.cmd.stack(cmdstr)
                    del self.aircrafts[ac]
        return

    def routine(self):
        while True:
            if not self.routine_stop_flag:
                self.update_all_ac_postition()
                self.stack_all_commands()
                self.remove_outdated_ac()
                if self.console_print:
                    self.print_acdata()
                time.sleep(3)
            else:
                time.sleep(0.5)
        return
