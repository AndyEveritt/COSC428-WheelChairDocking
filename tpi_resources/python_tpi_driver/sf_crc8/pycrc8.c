#include <Python.h>
#include "SF_CRC8.h"

// see https://docs.python.org/3.5/extending/extending.html

static PyObject *crc_of_bytes(PyObject *self, PyObject *args)
{
    Py_buffer buffer;
    uint8_t crc;

    // y* : bytes like object
    if (!PyArg_ParseTuple(args, "y*", &buffer))
        return NULL;

    crc = SF_CRC8_CalculateCRC8((uint8_t *) buffer.buf, (uint8_t) buffer.len, SF_CRC8_INITIAL_VALUE, true);

    PyBuffer_Release(&buffer);
    return PyLong_FromUnsignedLong((long) crc);
}

static PyMethodDef crc8Methods[] = {
    {"crc_of_bytes",  crc_of_bytes, METH_VARARGS, "One shot crc calculation on a bytes like thing."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef crc8module = {
    PyModuleDef_HEAD_INIT,
    "crc8",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    crc8Methods
};

PyMODINIT_FUNC PyInit_crc8(void)
{
    return PyModule_Create(&crc8module);
}
