"""
TPI Packet Decoder

Copyright 2018 Dynamic Controls
"""
try:
    from sf_crc8 import crc8
except ImportError:
    print("To build the crc8 python module: \n\tcd sf_crc8\n\tpython3 setup.py build_ext --inplace")
    exit(1)


# reverse dictionary look up, find a key that matches a value
def key_for_value(dictionary, value):
    return next(key for key, v in dictionary.items() if v == value)


class TPIPacket:
    # Serial Packet Status Response Codes
    status_codes = {
        b'\x00': "STATUS_OK",
        b'\x01': "UNKOWN_TYPE_IDENTIFIER",
        b'\x02': "INVALID_DATA",
        b'\x03': "INVALID_CRC",
        b'\x04': "OTHER_ERROR"
    }

    module_types = {
        b'\x00': "PMDO",
        b'\x01': "REMDO",
        b'\x02': "LAK",
        b'\x03': "PMLE",
        b'\x04': "REMLE",
        b'\x05': "PMAL",
        b'\x06': "REMAL",
        b'\x07': "GYRO",
        b'\x08': "ACT",
        b'\x09': "TPI",
        b'\x0A': "REMRE",
        b'\x0B': "TILT",
        b'\x0C': "DISP",
        b'\x0D': "ACU",
        b'\x0E': "INPUT",
        b'\x0F': "OUTPUT",
        b'\x10': "CR",
        b'\x11': "TPI_ACU",
    }

    type_ids = {
        b'\x00': "NONE",
        b'\x01': "RESPONSE_STATUS",

        b'\x70': "REQUEST_CONNECTED_MODULES",
        b'\x71': "RESPONSE_CONNECTED_MODULES",

        b'\x88': "REQUEST_MODIFY_DEMAND",

        b'\x90': "REQUEST_ENABLE_USER_INPUT",
        b'\x91': "RESPONSE_USER_INPUT",
        b'\x92': "REQUEST_ENABLE_MOTOR_SPEED",
        b'\x93': "RESPONSE_MOTOR_SPEED",
        b'\x94': "REQUEST_ENABLE_BUTTON_PRESSES",
        b'\x95': "RESPONSE_BUTTON_PRESSES",
        b'\x96': "REQUEST_ENABLE_GYRO_TURN_SPEED",
        b'\x97': "RESPONSE_GYRO_TURN_SPEED",
        b'\x98': "REQUEST_ENABLE_ACTIVE_USER_FUNCTION",
        b'\x99': "RESPONSE_ACTIVE_USER_FUNCTION",
        b'\x9A': "REQUEST_ENABLE_SPEED_SCALING",
        b'\x9B': "RESPONSE_SPEED_SCALING",
    }

    SERIAL_DELIMITER = b'\xF0'

    @classmethod
    def is_start_byte(cls, byte_value):
        return byte_value == cls.SERIAL_DELIMITER

    @classmethod
    def get_type_name(cls, type_id):
        return cls.type_ids[type_id]

    @classmethod
    def get_module_type(cls, module_type):
        return cls.module_types[module_type]

    @classmethod
    def get_type_id(cls, type_name):
        try:
            return key_for_value(cls.type_ids, type_name)
        except KeyError:
            return None

    def __init__(self, type_id):
        self.type_id = type_id  # byte
        self.data_len = b'\x00'
        self.data = b''  # bytes
        self.data_buffer = []  # array of byte things
        self.crc = b''
        self.verbose = False

    def calculate_crc(self):
        if self.data is not None:
            if len(self.data) == 0 and len(self.data_buffer) > 0:
                self.data = b"".join(self.data_buffer)
            content = b"".join([self.type_id, self.data_len, self.data])
        else:
            content = b"".join([self.type_id, self.data_len])
        self.crc = bytes([crc8.crc_of_bytes(content)])

    @property
    def valid(self):
        # Override with a custom check!
        return len(self.crc) == 1

    def __str__(self):
        if self.verbose:
            return "{} ({}): data ({} bytes): {}, crc: {}, valid: {}. \t{}".format(self.get_type_name(self.type_id), self.type_id, self.data_len, self.data, self.crc, self.valid, self.data_string)
        else:
            return "{}: \t{}".format(self.get_type_name(self.type_id), self.data_string)

class TPIPacketDecoder(TPIPacket):
    def __init__(self, type_id):
        super().__init__(type_id)
        self.data_len_int = 0  # data length as an integer value
        self.rx_crc = b'\x00'  # received crc
        self.read_idx = 0
        self.end_delimiter = 0

    def decode_data(self):
        ''' Dispatch method '''
        if self.valid:
            method_name = "decode_{}".format(self.type_ids[self.type_id])
            method = getattr(self, method_name, self.decode_generic)
            # Call the method
            method()
        else:
            self.data_string = "Invalid Packet"

    def read_byte(self, byte_value):
        '''
        :param byte_value: integer value of next byte in packet
        :return: number of bytes still to read in this packet
        '''
        if self.read_idx == 0: # i.e. zero'th byte after type id
            self.data_len_int = int.from_bytes(byte_value, byteorder='big', signed=False)
            self.data_len = byte_value
        elif self.read_idx <= self.data_len_int:
            self.data_buffer.append(byte_value)
        elif self.read_idx == self.data_len_int + 1:
            self.rx_crc = byte_value
        elif self.read_idx == self.data_len_int + 2:
            self.end_delimiter = byte_value
            self.decode_data()
        else:
            raise IndexError("Packet finished already")
        self.read_idx += 1

        return self.data_len_int + 3 - self.read_idx

    @property
    def valid(self):
        valid = self.end_delimiter == self.SERIAL_DELIMITER and len(self.data_buffer) == self.data_len_int
        self.calculate_crc()
        # print("got crc:", self.rx_crc, "expected:", self.crc, self.rx_crc == self.crc)
        valid = valid and self.rx_crc == self.crc
        return valid

    def decode_generic(self):
        self.data_string = " ".join([b.hex() for b in self.data])

    def decode_RESPONSE_STATUS(self):
        self.data_string = self.status_codes[self.data_buffer[0]]
        try:
            if self.type_ids[self.data_buffer[1]] != "NONE":
                self.data_string += ", in response to ({})".format(self.type_ids[self.data_buffer[1]])
        except KeyError:
            self.data_string += ", in response to  ({})".format([self.data_buffer[1]])

    def decode_RESPONSE_USER_INPUT(self):
        self.x = int.from_bytes(self.data_buffer[0], byteorder='big', signed=True)
        self.y = int.from_bytes(self.data_buffer[1], byteorder='big', signed=True)
        self.sp = int.from_bytes(self.data_buffer[2], byteorder='big', signed=False)
        self.data_string = "UI: x {}%, y {}%, sp {}%".format(self.x, self.y, self.sp)

    def decode_RESPONSE_MOTOR_SPEED(self):
        self.left = int.from_bytes(b"".join(self.data_buffer[0:2]), byteorder='big', signed=True) / 320.0
        self.right = int.from_bytes(b"".join(self.data_buffer[2::]), byteorder='big', signed=True) / 320.0
        self.data_string = "l {:.2f}%, r {:.2f}%".format(self.left, self.right)

    def decode_RESPONSE_BUTTON_PRESSES(self):
        self.button = int.from_bytes(self.data_buffer[0], byteorder='big', signed=True)
        self.state = int.from_bytes(self.data_buffer[1], byteorder='big', signed=True)
        self.data_string = "button {}, {}".format(self.button, "Pressed" if self.state == 1 else "Released")

    def decode_RESPONSE_CONNECTED_MODULES(self):
        self.modules = []
        for i in range(self.data_len_int):
            module_i = self.get_module_type(self.data_buffer[i])
            self.modules.append(module_i)
        self.data_string = "connected modules: {}".format(self.modules)

    def decode_RESPONSE_GYRO_TURN_SPEED(self):
        self.turn = int.from_bytes(b"".join(self.data_buffer[0::]), byteorder='big', signed=True) / 128
        self.data_string = "turn speed: {}".format(self.turn)

    def decode_RESPONSE_ACTIVE_USER_FUNCTION(self):
        self.active_user_function = int.from_bytes(self.data_buffer[0], byteorder='big', signed=True)
        self.data_string = "active user function: {}".format(self.active_user_function)

    def decode_RESPONSE_SPEED_SCALING(self):
        self.forward = int.from_bytes(self.data_buffer[0], byteorder='big', signed=True)
        self.reverse = int.from_bytes(self.data_buffer[1], byteorder='big', signed=True)
        self.left = int.from_bytes(self.data_buffer[2], byteorder='big', signed=False)
        self.right = int.from_bytes(self.data_buffer[2], byteorder='big', signed=False)
        self.data_string = "Speed Scaling: fwd {}%, rev {}%, l {}%, r {}%".format(self.forward, self.reverse, self.left, self.right)


class TPIPacketEncoder(TPIPacket):
    def __init__(self, type_name, data):
        '''Note Data should be an array of integers'''
        super().__init__(self.get_type_id(type_name))
        self.encode_data(data)
        self.calculate_crc()

    def encode_data(self, data):
        ''' Dispatch method '''
        self.data_string = data
        method_name = "encode_{}".format(self.type_ids[self.type_id])
        method = getattr(self, method_name, lambda x: print("encoder not implemented for {}".format(x)))
        # Call the method
        method(data)
        if self.data is not None:
            self.data_len = bytes([len(self.data)])
        else:
            self.data_len = bytes([0])

    def uint8_encoder(self, data):
        '''assumes data is an array of values which are in the range 0 to 255'''
        self.data = bytes(data)

    def percentage_encoder(self, data):
        assert data[0] >= 0 and data[0] <= 100
        self.uint8_encoder(data)

    def bool_encoder(self, data):
        '''assumes 1 boolean value in data array'''
        self.data = bytes([1 if data[0] else 0])

    def encode_RESPONSE_STATUS(self, data):
        if data[0]:  # signal OK
            status = "STATUS_OK"
        else:
            status = "OTHER_ERROR"
        self.data = key_for_value(self.status_codes, status) + key_for_value(self.type_ids, "NONE")  # concatenate bytes

    def encode_REQUEST_ENABLE_USER_INPUT(self, data):
        self.bool_encoder(data)

    def encode_REQUEST_ENABLE_MOTOR_SPEED(self, data):
        self.bool_encoder(data)

    def encode_REQUEST_ENABLE_BUTTON_PRESSES(self, data):
        self.bool_encoder(data)

    def encode_REQUEST_ENABLE_GYRO_TURN_SPEED(self, data):
        self.bool_encoder(data)

    def encode_REQUEST_ENABLE_ACTIVE_USER_FUNCTION(self, data):
        self.bool_encoder(data)

    def encode_REQUEST_ENABLE_SPEED_SCALING(self, data):
        self.bool_encoder(data)

    def encode_REQUEST_MODIFY_DEMAND(self, data):
        # data = [x, y]
        assert data[0] >= -100 and data[0] <= 100
        assert data[1] >= -100 and data[1] <= 100
        self.data = data[0].to_bytes(1, byteorder='big', signed=True) + data[1].to_bytes(1, byteorder='big', signed=True)


    def encode_REQUEST_CONNECTED_MODULES(self, data):
        self.data = None # No data needed for this message

    def get_bytes(self):
        ''' To send stuff '''
        if self.data is None:
            return [self.SERIAL_DELIMITER, self.type_id, self.data_len, self.crc, self.SERIAL_DELIMITER]
        else:
            return [self.SERIAL_DELIMITER, self.type_id, self.data_len, self.data, self.crc, self.SERIAL_DELIMITER]

