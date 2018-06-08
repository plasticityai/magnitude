/*
   Unix SMB/CIFS implementation.
   Python 3 compatibility macros
   Copyright (C) Petr Viktorin <pviktori@redhat.com> 2015

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _SAMBA_PY3COMPAT_H_
#define _SAMBA_PY3COMPAT_H_
#include <Python.h>

/* Quick docs:
 *
 * The IS_PY3 preprocessor constant is 1 on Python 3, and 0 on Python 2.
 *
 * "PyStr_*" works like PyUnicode_* on Python 3, but uses bytestrings (str)
 * under Python 2.
 *
 * "PyBytes_*" work like in Python 3; on Python 2 they are aliased to their
 * PyString_* names.
 *
 * "PyInt_*" works like PyLong_*
 *
 * Syntax for module initialization is as in Python 3, except the entrypoint
 * function definition and declaration:
 *     PyMODINIT_FUNC PyInit_modulename(void);
 *     PyMODINIT_FUNC PyInit_modulename(void)
 *     {
 *         ...
 *     }
 * is replaced by:
 *     MODULE_INIT_FUNC(modulename)
 *     {
 *         ...
 *     }
 *
 * In the entrypoint, create a module using PyModule_Create and PyModuleDef,
 * and return it. See Python 3 documentation for details.
 * For Python 2 compatibility, always set PyModuleDef.m_size to -1.
 *
 */

#if PY_MAJOR_VERSION >= 3

/***** Python 3 *****/

#define IS_PY3 1

/* Strings */

#define PyStr_Type PyUnicode_Type
#define PyStr_Check PyUnicode_Check
#define PyStr_CheckExact PyUnicode_CheckExact
#define PyStr_FromString PyUnicode_FromString
#define PyStr_FromStringAndSize PyUnicode_FromStringAndSize
#define PyStr_FromFormat PyUnicode_FromFormat
#define PyStr_FromFormatV PyUnicode_FromFormatV
#define PyStr_AsString PyUnicode_AsUTF8
#define PyStr_Concat PyUnicode_Concat
#define PyStr_Format PyUnicode_Format
#define PyStr_InternInPlace PyUnicode_InternInPlace
#define PyStr_InternFromString PyUnicode_InternFromString
#define PyStr_Decode PyUnicode_Decode

#define PyStr_AsUTF8String PyUnicode_AsUTF8String // returns PyBytes
#define PyStr_AsUTF8 PyUnicode_AsUTF8
#define PyStr_AsUTF8AndSize PyUnicode_AsUTF8AndSize

/* description of bytes and string objects */
#define PY_DESC_PY3_BYTES "bytes"
#define PY_DESC_PY3_STRING "string"

/* Determine if object is really bytes, for code that runs
 * in python2 & python3 (note: PyBytes_Check is replaced by
 * PyString_Check in python2) so care needs to be taken when
 * writing code that will check if incoming type is bytes that
 * will work as expected in python2 & python3
 */

#define IsPy3Bytes PyBytes_Check

#define IsPy3BytesOrString(pystr) \
    (PyStr_Check(pystr) || PyBytes_Check(pystr))


/* Ints */

#define PyInt_Type PyLong_Type
#define PyInt_Check PyLong_Check
#define PyInt_CheckExact PyLong_CheckExact
#define PyInt_FromString PyLong_FromString
#define PyInt_FromLong PyLong_FromLong
#define PyInt_FromSsize_t PyLong_FromSsize_t
#define PyInt_FromSize_t PyLong_FromSize_t
#define _PyInt_AsInt PyLong_AsLong
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyInt_AsUnsignedLongLongMask PyLong_AsUnsignedLongLongMask
#define PyInt_AsSsize_t PyLong_AsSsize_t

/* Strings */
#define PyString_Type PyUnicode_Type
#define PyBuffer_Type PyBytes_Type
#define PyString_Type PyUnicode_Type
#define PyString_Check PyUnicode_Check 
#define PyString_CheckExact PyUnicode_CheckExact
#define PyString_FromString PyUnicode_FromString 
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize 
#define PyString_FromFormat PyUnicode_FromFormat 
#define PyString_FromFormatV PyUnicode_FromFormatV 
#define PyString_AsString(s) (PyUnicode_AsEncodedString(s, "utf-8", "Error ~"))
#define PyString_AS_STRING(s) (PyUnicode_AsEncodedString(s, "utf-8", "Error ~"))
#define PyString_AsStringAndSize PyBytes_AsStringAndSize 
#define PyString_Format PyUnicode_Format 
#define PyString_InternInPlace PyUnicode_InternInPlace 
#define PyString_InternFromString PyUnicode_InternFromString 
#define PyString_Decode PyUnicode_Decode 
#define PyString_GET_SIZE PyUnicode_GET_SIZE
#define PyString_Concat PyUnicode_Concat
#define PyBuffer_Check PyBytes_Check

/* Exceptions */
#define PyExc_StandardError PyExc_Exception

/* Macros for ob_type */
#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

/* Other Macros */
#define RO READONLY
#define Py_TPFLAGS_HAVE_ITER 0
#define Py_TPFLAGS_HAVE_WEAKREFS 0

/* Module init */

#define MODULE_INIT_FUNC(name) \
    PyMODINIT_FUNC PyInit_ ## name(void); \
    PyMODINIT_FUNC PyInit_ ## name(void)

#define MOD_DEF(ob, name, doc, methods) \
  static struct PyModuleDef moduledef = { \
      PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
  ob = PyModule_Create(&moduledef);

/* PyArg_ParseTuple/Py_BuildValue argument */

#define PYARG_BYTES_LEN "y#"

#else

/***** Python 2 *****/

#define IS_PY3 0

/* Strings */

#define PyStr_Type PyString_Type
#define PyStr_Check PyString_Check
#define PyStr_CheckExact PyString_CheckExact
#define PyStr_FromString PyString_FromString
#define PyStr_FromStringAndSize PyString_FromStringAndSize
#define PyStr_FromFormat PyString_FromFormat
#define PyStr_FromFormatV PyString_FromFormatV
#define PyStr_AsString PyString_AsString
#define PyStr_Format PyString_Format
#define PyStr_InternInPlace PyString_InternInPlace
#define PyStr_InternFromString PyString_InternFromString
#define PyStr_Decode PyString_Decode

#define PyStr_AsUTF8String(str) (Py_INCREF(str), (str))
#define PyStr_AsUTF8 PyString_AsString
#define PyStr_AsUTF8AndSize(pystr, sizeptr) \
    ((*sizeptr=PyString_Size(pystr)), PyString_AsString(pystr))

#define PyBytes_Type PyString_Type
#define PyBytes_Check PyString_Check
#define PyBytes_CheckExact PyString_CheckExact
#define PyBytes_FromString PyString_FromString
#define PyBytes_FromStringAndSize PyString_FromStringAndSize
#define PyBytes_FromFormat PyString_FromFormat
#define PyBytes_FromFormatV PyString_FromFormatV
#define PyBytes_Size PyString_Size
#define PyBytes_GET_SIZE PyString_GET_SIZE
#define PyBytes_AsString PyString_AsString
#define PyBytes_AS_STRING PyString_AS_STRING
#define PyBytes_AsStringAndSize PyString_AsStringAndSize
#define PyBytes_Concat PyString_Concat
#define PyBytes_ConcatAndDel PyString_ConcatAndDel
#define _PyBytes_Resize _PyString_Resize

/* description of bytes and string objects */
#define PY_DESC_PY3_BYTES "string"
#define PY_DESC_PY3_STRING "unicode"

/* Determine if object is really bytes, for code that runs
 * in python2 & python3 (note: PyBytes_Check is replaced by
 * PyString_Check in python2) so care needs to be taken when
 * writing code that will check if incoming type is bytes that
 * will work as expected in python2 & python3
 */

#define IsPy3Bytes(pystr) false

#define IsPy3BytesOrString PyStr_Check

/* PyArg_ParseTuple/Py_BuildValue argument */

#define PYARG_BYTES_LEN "s#"

/* Module init */

#define PyModuleDef_HEAD_INIT 0

typedef struct PyModuleDef {
    int m_base;
    const char* m_name;
    const char* m_doc;
    Py_ssize_t m_size;
    PyMethodDef *m_methods;
} PyModuleDef;

#define PyModule_Create(def) \
    Py_InitModule3((def)->m_name, (def)->m_methods, (def)->m_doc)

#define MODULE_INIT_FUNC(name) \
    static PyObject *PyInit_ ## name(void); \
    void init ## name(void); \
    void init ## name(void) { PyInit_ ## name(); } \
    static PyObject *PyInit_ ## name(void)

#define MOD_DEF(ob, name, doc, methods) \
    ob = Py_InitModule3(name, methods, doc);


#endif

#endif
