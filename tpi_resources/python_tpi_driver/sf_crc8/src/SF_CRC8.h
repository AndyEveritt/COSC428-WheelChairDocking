#ifndef SF_CRC8_H
#define SF_CRC8_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define SF_CRC8_INITIAL_VALUE       ((uint8_t)0xff)

/**
 * @brief SF_CRC8_CalculateCRC8
 * based on AUTOSAR interface for CRC Calculation of SAE-J1850 CRC8
 * @param dataPtr: pointer to data
 * @param length: length of data
 * @param initialValue: set to SF_CRC8_INITIAL_VALUE if this is the first call,
 * other wise previous result
 * @param isFinalValue: if this is the last call for this crc
 * @return the crc
 */
uint8_t SF_CRC8_CalculateCRC8(const uint8_t* dataPtr,
                              uint32_t length,
                              uint8_t initialValue,
                              bool isFinalValue);

#endif /* SF_CRC8_H */
