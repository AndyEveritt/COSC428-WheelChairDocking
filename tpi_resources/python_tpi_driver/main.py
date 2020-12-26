#! /usr/bin/python3
"""
main for interacting with the TPI via the serial interface

Copyright 2018 Dynamic Controls
"""

import time
import argparse
from tpi_serial_reader import TPIInterface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', action='store', default='/dev/ttyUSB0',
                        help='Serial port to connect to')
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='Run test mode')
    parser.add_argument('-c', '--check', action='store_true', default=False,
                        help='Check that two way comms are working, and which modules are present')
    parser.add_argument('-s', '--stream', action='store_true', default=False,
                        help='Enable Stream data')
    parser.add_argument('-d', '--drive', action='store_true', default=False,
                        help='Send some drive data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Print helpful stuff')
    parser.add_argument('-i', '--interactive', action='store_true', default=False,
                        help='Enable interactive issuing of messages')

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

        if args.stream or args.test:
            # Enable data streams we're interested in
            for ds in TPIInterface.data_streams:
                pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2) # in case there is already a packet waiting
                tpi_serial.enable_data_stream(ds, True, verbose)
                pkt = tpi_serial.check_for_rx_packet(verbose)
                time.sleep(0.010)
                if args.test:
                    time.sleep(0.010)
                    tpi_serial.enable_data_stream(ds, False, verbose)
                    pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)
                    pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)

            while args.stream:
                tpi_serial.check_for_rx_packet(verbose)

        if args.drive or args.test:
            for i in range(10):
                for j in range(10):
                    x = i * 10
                    y = 100 - i * 10
                    tpi_serial.send_modified_demand(x, y, verbose)
                    pkt = tpi_serial.check_for_rx_packet(verbose)
                    time.sleep(0.005)
                    if args.test:
                        break  # only do once per value in test mode

        if args.check or args.test:
            pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)
            pkt = tpi_serial.request_connected_modules(verbose)
            pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)

            for i in range(2):
                tpi_serial.send_status()
                pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)

            if tpi_serial.n_rx == 0:
                print("\n*** FAIL *** ")
            else:
                print("\n*** PASS *** ")
            print("tx: {}, rx: {}".format(tpi_serial.n_tx, tpi_serial.n_rx))

    except KeyboardInterrupt:
        pass

    finally:
        print("Finishing")
        pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)
        pkt = tpi_serial.check_for_rx_packet(verbose, timeout=2)

        # Ensure each stream is disabled (and receive any cached packets)
        if args.stream:
            for ds in TPIInterface.data_streams:
                tpi_serial.enable_data_stream(ds, False, verbose)
                pkt = tpi_serial.check_for_rx_packet(verbose)

        tpi_serial.close()
