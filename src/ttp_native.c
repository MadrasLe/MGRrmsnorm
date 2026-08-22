#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
typedef SOCKET ttp_socket_t;
#else
#include <errno.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/socket.h>
typedef int ttp_socket_t;
#endif

#define TTP_DEFAULT_MAX_FRAME_BYTES (2ULL * 1024ULL * 1024ULL * 1024ULL)

typedef enum {
    TTP_READ_OK = 0,
    TTP_READ_CLOSED = 1,
    TTP_READ_ERROR = 2,
    TTP_READ_TIMEOUT = 3
} ttp_read_status_t;

typedef struct {
    ttp_read_status_t status;
    int error_code;
} ttp_read_result_t;

static ttp_read_result_t ttp_result(ttp_read_status_t status, int error_code) {
    ttp_read_result_t result;
    result.status = status;
    result.error_code = error_code;
    return result;
}

static ttp_read_result_t ttp_wait_readable(ttp_socket_t sock, double timeout_seconds) {
    int rc;

    for (;;) {
        fd_set readfds;
        struct timeval timeout_value;
        struct timeval *timeout_ptr = NULL;

        FD_ZERO(&readfds);
        FD_SET(sock, &readfds);

        if (timeout_seconds >= 0.0) {
            long seconds = (long)timeout_seconds;
            double fractional = timeout_seconds - (double)seconds;
            long usec = (long)(fractional * 1000000.0);
            if (usec < 0) {
                usec = 0;
            } else if (usec > 999999) {
                usec = 999999;
            }
            timeout_value.tv_sec = seconds;
            timeout_value.tv_usec = usec;
            timeout_ptr = &timeout_value;
        }

#ifdef _WIN32
        rc = select(0, &readfds, NULL, NULL, timeout_ptr);
        if (rc == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAEINTR) {
                continue;
            }
            return ttp_result(TTP_READ_ERROR, err);
        }
#else
        rc = select(sock + 1, &readfds, NULL, NULL, timeout_ptr);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return ttp_result(TTP_READ_ERROR, errno);
        }
#endif
        if (rc == 0) {
            return ttp_result(TTP_READ_TIMEOUT, 0);
        }
        return ttp_result(TTP_READ_OK, 0);
    }
}

static ttp_read_result_t ttp_read_exact(
    ttp_socket_t sock,
    char *dst,
    size_t nbytes,
    double timeout_seconds) {
    size_t offset = 0;

    while (offset < nbytes) {
        size_t remaining = nbytes - offset;

#ifdef _WIN32
        int want = remaining > (size_t)INT_MAX ? INT_MAX : (int)remaining;
        int got = recv(sock, dst + offset, want, 0);
        if (got == 0) {
            return ttp_result(TTP_READ_CLOSED, 0);
        }
        if (got == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAEINTR) {
                continue;
            }
            if (err == WSAEWOULDBLOCK) {
                ttp_read_result_t wait_result = ttp_wait_readable(sock, timeout_seconds);
                if (wait_result.status != TTP_READ_OK) {
                    return wait_result;
                }
                continue;
            }
            return ttp_result(TTP_READ_ERROR, err);
        }
#else
        size_t want_size = remaining > (size_t)INT_MAX ? (size_t)INT_MAX : remaining;
        ssize_t got = recv(sock, dst + offset, want_size, 0);
        if (got == 0) {
            return ttp_result(TTP_READ_CLOSED, 0);
        }
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                ttp_read_result_t wait_result = ttp_wait_readable(sock, timeout_seconds);
                if (wait_result.status != TTP_READ_OK) {
                    return wait_result;
                }
                continue;
            }
            return ttp_result(TTP_READ_ERROR, errno);
        }
#endif

        offset += (size_t)got;
    }

    return ttp_result(TTP_READ_OK, 0);
}

static PyObject *ttp_raise_read_error(ttp_read_result_t result) {
    if (result.status == TTP_READ_CLOSED) {
        PyErr_SetString(PyExc_EOFError, "TTP connection closed");
        return NULL;
    }
    if (result.status == TTP_READ_TIMEOUT) {
        PyErr_SetString(PyExc_TimeoutError, "TTP receive timed out");
        return NULL;
    }

#ifdef _WIN32
    PyErr_Format(PyExc_OSError, "socket recv failed with WSA error %d", result.error_code);
    return NULL;
#else
    errno = result.error_code;
    return PyErr_SetFromErrno(PyExc_OSError);
#endif
}

static uint64_t ttp_read_le_u64(const unsigned char *src) {
    uint64_t value = 0;
    value |= ((uint64_t)src[0]) << 0;
    value |= ((uint64_t)src[1]) << 8;
    value |= ((uint64_t)src[2]) << 16;
    value |= ((uint64_t)src[3]) << 24;
    value |= ((uint64_t)src[4]) << 32;
    value |= ((uint64_t)src[5]) << 40;
    value |= ((uint64_t)src[6]) << 48;
    value |= ((uint64_t)src[7]) << 56;
    return value;
}

static PyObject *ttp_recv_packet(PyObject *self, PyObject *args, PyObject *kwargs) {
    unsigned long long fd_arg = 0;
    unsigned long long max_frame_bytes = TTP_DEFAULT_MAX_FRAME_BYTES;
    double timeout_seconds = -1.0;
    unsigned char prefix[8];
    uint64_t length = 0;
    ttp_read_result_t read_result;
    PyObject *packet = NULL;
    char *packet_data = NULL;
    static char *kwlist[] = {
        (char *)"fd",
        (char *)"max_frame_bytes",
        (char *)"timeout_seconds",
        NULL,
    };

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "K|Kd",
            kwlist,
            &fd_arg,
            &max_frame_bytes,
            &timeout_seconds)) {
        return NULL;
    }

#ifdef _WIN32
    ttp_socket_t sock = (ttp_socket_t)(uintptr_t)fd_arg;
#else
    if (fd_arg > (unsigned long long)INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "socket fd does not fit in int");
        return NULL;
    }
    ttp_socket_t sock = (ttp_socket_t)fd_arg;
#endif

    Py_BEGIN_ALLOW_THREADS
    read_result = ttp_read_exact(sock, (char *)prefix, sizeof(prefix), timeout_seconds);
    Py_END_ALLOW_THREADS

    if (read_result.status != TTP_READ_OK) {
        return ttp_raise_read_error(read_result);
    }

    length = ttp_read_le_u64(prefix);
    if (length > (uint64_t)max_frame_bytes) {
        PyErr_Format(
            PyExc_ValueError,
            "TTP frame too large: %llu bytes",
            (unsigned long long)length);
        return NULL;
    }
    if (length > (uint64_t)PY_SSIZE_T_MAX) {
        PyErr_SetString(PyExc_OverflowError, "TTP frame too large for Python bytearray");
        return NULL;
    }

    packet = PyByteArray_FromStringAndSize(NULL, (Py_ssize_t)length);
    if (packet == NULL) {
        return NULL;
    }

    if (length == 0) {
        return packet;
    }

    packet_data = PyByteArray_AS_STRING(packet);
    Py_BEGIN_ALLOW_THREADS
    read_result = ttp_read_exact(sock, packet_data, (size_t)length, timeout_seconds);
    Py_END_ALLOW_THREADS

    if (read_result.status != TTP_READ_OK) {
        Py_DECREF(packet);
        return ttp_raise_read_error(read_result);
    }

    return packet;
}

static PyMethodDef ttp_methods[] = {
    {
        "recv_packet",
        (PyCFunction)ttp_recv_packet,
        METH_VARARGS | METH_KEYWORDS,
        "recv_packet(fd, max_frame_bytes=2147483648, timeout_seconds=-1.0) -> bytearray\n"
        "\n"
        "Receive one length-prefixed TTP packet from an existing socket fd.",
    },
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef ttp_module = {
    PyModuleDef_HEAD_INIT,
    "megagemm_ttp_native",
    "Native MegaMesh TTP packet receive helpers.",
    -1,
    ttp_methods,
};

PyMODINIT_FUNC PyInit_megagemm_ttp_native(void) {
    PyObject *module = PyModule_Create(&ttp_module);
    if (module == NULL) {
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "SUPPORTS_TIMEOUT_SOCKETS", 1) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddStringConstant(module, "TTP_NATIVE_VERSION", "1") < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
