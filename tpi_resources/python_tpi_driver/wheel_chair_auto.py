#! /usr/bin/python3
"""
main for interacting with the TPI via the serial interface

Copyright 2018 Dynamic Controls
"""

import time
from threading import Thread
from threading import Lock
import argparse
from tpi_serial_reader import TPIInterface


serial_mutex = Lock()
x = 10
y = 0
verbose = 0
stop = False
pkt = 0

def send_command_loop():
    global pkt
    ticks = 0
    while not stop:
        with serial_mutex:
            tpi_serial.send_modified_demand(x, y, verbose)
            time.sleep(0.04)
            ticks += 1
            if (ticks > 20):
                pkt = tpi_serial.check_for_rx_packet(verbose)
                ticks = 0
		


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', action='store', default='/dev/ttyUSB0',
                        help='Serial port to connect to')
    parser.add_argument('-d', '--drive', action='store_true', default=False,
                        help='Send some drive data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print helpful stuff')

    args = parser.parse_args()

    verbose = args.verbose  # this shows more stuff
    print("Starting TPI serial reader, hit ctrl-c to finish")
    tpi_serial = TPIInterface(args.port, 115200, timeout=0.01)
    tpi_serial.send_status(ok=True, verbose=verbose)  # to check if TPI is awake

    try:
        awake = False
        attempts = 0
        while not awake and attempts < 3:
            pkt = tpi_serial.check_for_rx_packet(verbose)
            attempts += 1
            if pkt is not None:
                if pkt.data_string.find("OK") > -1:
                    awake = True
                    
        # Start command sending loop, needs to run to keep chair out of manual
        thread = Thread(target=send_command_loop)
        thread.start()

        if args.drive:
            for i in range(10):
                for j in range(10):        
                    with serial_mutex:
                        #pkt = tpi_serial.check_for_rx_packet(verbose)
                        x = i * 10
                        y = 100 - i * 10
                        time.sleep(0.01)
                    #finally:
                      #  serial_mutex.release()
                    

    except KeyboardInterrupt:
        pass

    finally:
        print("Finishing")
        stop = True
        thread.join()
        pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)
        tpi_serial.close()