""" 
Stream data from a TCP server providing datafeed of ADS-B messages

"""

import os
import socket
import time
import threading
import math
import adsb_decoder
import aero

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
            power &= 0x3FFF  
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

            df = adsb_decoder.get_df(msg)
            tc = adsb_decoder.get_tc(msg)
            ca = adsb_decoder.get_ca(msg)

            if df == 17:
                addr = adsb_decoder.get_icao_addr(msg)

                if tc>=1 and tc<=4:                     # aircraft identification
                    callsign = adsb_decoder.get_callsign(msg)
                    self.datafeed.update_callsign_data(addr, callsign)
                if tc>=9 and tc<=18:                    # airbone postion frame
                    alt = adsb_decoder.get_alt(msg)
                    oe = adsb_decoder.get_oe_flag(msg)  # odd or even frame
                    cprlat = adsb_decoder.get_cprlat(msg)
                    cprlon = adsb_decoder.get_cprlon(msg)
                    dataset = {'addr':addr, 'oe':oe, 'time':t_sec_ac, 'alt':alt,
                            'cprlat':cprlat, 'cprlon':cprlon}
                    # print dataset
                    self.datafeed.push_cprpos_data(dataset)
                elif tc==19:        # airbone velocity frame
                    sh = adsb_decoder.get_speed_heading(msg)
                    if sh:
                        dataset = {'addr':addr, 'speed':sh[0], 'heading':sh[1]}
                        self.datafeed.push_speed_heading_data(dataset)

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
                        try:
                            datatype = int(raw_data[1])
                            datalen = data[2]<<8 | data[3]
                            self.read_message(datatype, data[4:], datalen)
                        except:
                            pass
                except socket.error, exc:
                    print "Socket reading error: %s" % exc
            else:
                time.sleep(0.5)
        return


class DataFeed:
    def __init__(self, tmx):
        self.tmx  = tmx
        self.acdata_pool = {}    # a dict store all the aircraft status data
        self.routine_stop_flag = True
        self.routine_thread = threading.Thread(target=self.routine)
        self.routine_thread.daemon = True
        self.tcpclient =  False
        self.console_print = False
        return

    def connect(self, serverip, port):
        print "init TCP connection"
        if not self.tcpclient:
            try:
                self.tcpclient = TcpClient(self, serverip, port)
            except RuntimeError, e:
                self.tmx.scr.echo("Can not connect to server.")
                self.tcpclient = False
                return

        if not self.routine_thread.isAlive():
            self.routine_thread.start()
        self.start()
        return
    
    def start(self):
        print "starting ADS-B datafeed"
        self.routine_stop_flag = False
        
        if self.tcpclient:      # check if tcpclient exists
            self.tcpclient.start()
        else:
            self.tmx.scr.echo("Client not ready, connect to the server first")
        return

    ''' Function to terminate the routine and tcpclient threads '''  
    def stop(self):
        print "stopping ADS-B datafeed"
        self.routine_stop_flag = True

        if self.tcpclient:      # check if tcpclient exists
            self.tcpclient.stop()
        else:
            self.tmx.scr.echo("Client not ready, connect to the server first")
        return

    def set_console_print(self, flag):
            self.console_print = flag
    

    ''' Add an adsb cpr position data to the pool '''  
    def push_cprpos_data(self, data):

        if data['addr'] in self.acdata_pool:
            acdata = self.acdata_pool[data['addr']]
        else:
            acdata = {}

        acdata['alt'] = data['alt']
        if data['oe'] == '1':       # odd frame cpr position
            acdata['cprlat1'] = data['cprlat']
            acdata['cprlon1'] = data['cprlon']
            acdata['t1'] = data['time']
        if data['oe'] == '0':       # even frame cpr position
            acdata['cprlat0'] = data['cprlat']
            acdata['cprlon0'] = data['cprlon']
            acdata['t0'] = data['time']

        self.acdata_pool[data['addr']] = acdata
        # self.routine()
        return

    ''' Add speed and heading data to the pool '''  
    def push_speed_heading_data(self, data):

        if data['addr'] in self.acdata_pool:
            acdata = self.acdata_pool[data['addr']]
        else:
            acdata = {}

        acdata['speed'] = data['speed']
        acdata['heading'] = data['heading']

        self.acdata_pool[data['addr']] = acdata
        # self.routine()
        return

    ''' Update callsign when it is decoded '''  
    def update_callsign_data(self, addr, callsign):
        if addr in self.acdata_pool:
            self.acdata_pool[addr]['acid'] = callsign
        return

    ''' Loop through the aircraft pool, calculate and update all positions  '''
    def update_all_ac_postition(self):
        keys = ('cprlat0', 'cprlat1', 'cprlon0', 'cprlon1')
        for ac, acdata in self.acdata_pool.items():
            # check if all needed keys are in dict
            if set(keys).issubset(acdata):
                pos = adsb_decoder.cpr2position(acdata['cprlat0'], acdata['cprlat1'], 
                    acdata['cprlon0'], acdata['cprlon1'], acdata['t0'], acdata['t1'])

                # update positions of all aircrafts in the list
                if pos:
                    self.acdata_pool[ac]['lat'] = pos[0]
                    self.acdata_pool[ac]['lon'] = pos[1]
        return

    ''' Generate and stack command to command controller '''  
    def stack_all_commands(self):
        keys = ('lat', 'lon', 'alt', 'speed', 'heading')
        for ac, acdata in self.acdata_pool.items():
            # check if all needed keys are in dict
            if set(keys).issubset(acdata):
                acid = ac
                # acid = acdata['acid']
                # check is aircraft is already beening displayed
                if(self.tmx.traf.id2idx(acid) < 0):
                    v_cas = aero.tas2cas(acdata['speed'], acdata['alt']/3.2808)
                    cmdstr = 'CRE %s, %s, %f, %f, %f, %d, %d' % \
                        (acid, 'B738', acdata['lat'], acdata['lon'], \
                            acdata['heading'], acdata['alt'], v_cas)
                    self.tmx.cmd.stack(cmdstr)
                else:
                    cmdstr = 'MOVE %s, %f, %f, %d' % \
                        (acid, acdata['lat'], acdata['lon'], acdata['alt'])
                    self.tmx.cmd.stack(cmdstr)

                    cmdstr = 'HDG %s, %f' % (acid,  acdata['heading'])
                    self.tmx.cmd.stack(cmdstr)

                    v_cas = aero.tas2cas(acdata['speed'], acdata['alt']/3.2808)
                    cmdstr = 'SPD %s, %f' % (acid,  v_cas)
                    self.tmx.cmd.stack(cmdstr)
        return

    ''' Print aircraft status data to console '''  
    def print_acdata(self):
        os.system('cls' if os.name == 'nt' else 'clear')    # clear the console to print up-to-date table info

        count = 0
        keys = ('lat', 'lon', 'alt', 'speed', 'heading')
        for ac, acdata in self.acdata_pool.items():
            if set(keys).issubset(acdata):
                count += 1

        print 'time:', str(int(time.time())), " | Active aircraft counts: ", count
        return

    ''' House keeping, remove old entries that are not updated for more than 100 second  '''
    def remove_outdated_ac(self):
        for ac, acdata in self.acdata_pool.items():
            if 'time' in acdata:
                if (int(time.time()) - acdata['time']) > 90:    # threshold, remove ac after 90 seconds of no-seen
                    if 'acid' in acdata:    # remove from tmx screen if been presented
                        cmdstr = 'DEL %s' % acdata['acid']
                        self.tmx.cmd.stack(cmdstr)
                    del self.acdata_pool[ac]
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


if __name__ == '__main__':
    df = DataFeed(serverip='145.94.54.54', port=10001)
    df.connect()
    df.start()
    df.routine()
