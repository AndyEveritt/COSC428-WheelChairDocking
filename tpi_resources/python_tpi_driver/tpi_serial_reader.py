"""
TPI Serial Reader

Copyright 2018 Dynamic Controls
"""

import serial
import time
from tpi_packet_decoder import TPIPacket, TPIPacketDecoder, TPIPacketEncoder


tpi_sr_start_time = time.time()

def twos_complement(input_value, num_bits):
    '''Calculates a two's complement integer from the given input value's bits'''
    mask = 2 ** (num_bits - 1)
    return -(input_value & mask) + (input_value & ~mask)

def print_response(response_bytes):
    if len(response_bytes) > 0:
        print("[{:.3f}] RX: {}".format(time.time() - tpi_sr_start_time, " ".join([b.hex() for b in response_bytes])))

class TPIInterface(serial.Serial):
    data_streams = [v for v in TPIPacket.type_ids.values() if v.find("REQUEST_ENABLE") == 0]

    @property
    def n_rx(self):
        try:
            return self.__n_rx
        except AttributeError:
            self.__n_rx = 0
            return self.__n_rx

    @n_rx.setter
    def n_rx(self, x):
        self.__n_rx = x

    @property
    def n_tx(self):
        try:
            return self.__n_tx
        except AttributeError:
            self.__n_tx = 0
            return self.__n_tx

    @n_tx.setter
    def n_tx(self, x):
        self.__n_tx = x

    def check_for_rx_packet(self, verbose=False, timeout=50, print_packet=True):
        start_byte_found = False
        rx_packet = None
        response_bytes = []
        for attempts in range(timeout):  # allow some time for the response
            resp = self.read() # read 1 byte
            if len(resp) > 0:
                response_bytes.append(resp)
                if rx_packet is None:
                    if start_byte_found and not TPIPacket.is_start_byte(resp):
                        rx_packet = TPIPacketDecoder(resp)
                        break
                    else:
                        start_byte_found = TPIPacket.is_start_byte(resp)

        bytes_remaining = 1 if rx_packet is not None else 0
        while bytes_remaining > 0:
            time.sleep(0.0001)
            resp = self.read()
            if len(resp) > 0:
                response_bytes.append(resp)
                bytes_remaining = rx_packet.read_byte(resp)
        if rx_packet is not None and print_packet:
            print("RX:", str(rx_packet))
        if verbose:
            print_response(response_bytes)
        if rx_packet:
            self.n_rx += 1

        return rx_packet

    def send_packet(self, packet, verbose=False, print_packet=True):
        for b in packet.get_bytes():
            self.write(b)
        if verbose:
            print("[{:.3f}] TX: {}".format(time.time() - tpi_sr_start_time, " ".join([b.hex() for b in packet.get_bytes()])))
        if print_packet:
            print("TX:", str(packet))
        self.n_tx += 1

    def enable_data_stream(self, data_type, enable=True, verbose=False, print_packet=True):
        '''
        data_type must be one of TPI_Interface.data_streams
        '''
        tx_packet = TPIPacketEncoder(data_type, [enable])
        self.send_packet(tx_packet, verbose, print_packet)

    def request_connected_modules(self, verbose=False, print_packet=True):
        tx_packet = TPIPacketEncoder("REQUEST_CONNECTED_MODULES", None)
        self.send_packet(tx_packet, verbose, print_packet)

    def send_modified_demand(self, x=0, y=0, verbose=False, print_packet=True):
        tx_packet = TPIPacketEncoder("REQUEST_MODIFY_DEMAND", [x, y])
        self.send_packet(tx_packet, verbose, print_packet)

    def send_status(self, ok=True, verbose=False, print_packet=True):
        tx_packet = TPIPacketEncoder("RESPONSE_STATUS", [ok])
        self.send_packet(tx_packet, verbose, print_packet)
