/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file hal_backend_ml_test_wrapper.h
 * @brief Wrapper header for testing static functions in hal-backend-ml-*.cc files
 * 
 * This header provides a way to test static functions by undefining 'static'
 * when TESTING is defined. This allows the test to compile the source file
 * directly and access the functions.
 */

#ifdef TESTING

// Undefine static to make functions accessible during testing
#ifdef static
#undef static
#define static
#endif

// Note: The source files (hal-backend-ml-*.cc) will be included
// by the test files, and with TESTING defined, static functions will
// become regular functions that can be called from tests.

#endif // TESTING
