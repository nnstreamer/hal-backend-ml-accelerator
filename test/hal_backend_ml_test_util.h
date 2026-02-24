/* SPDX-License-Identifier: Apache-2.0 */

#ifndef __HAL_BACKEND_ML_TEST_UTIL_H__
#define __HAL_BACKEND_ML_TEST_UTIL_H__

#include "nnstreamer_plugin_api_filter.h"
#include <glib.h>
#include <gtest/gtest.h>
#include "hal-backend-ml-util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test-specific extension of GstTensorFilterProperties
 *
 * This struct extends the original GstTensorFilterProperties with test-specific
 * fields needed for unit testing. It uses composition to include all original
 * fields and adds test-specific extensions.
 */
typedef struct _TestGstTensorFilterProperties
{
  GstTensorFilterProperties base;  /**< Original GstTensorFilterProperties struct */

  /* Test-specific extensions */
  const char **input_data_files;  /**< Array of input data file paths for testing */
  guint num_input_files;          /**< Number of input data files */
} TestGstTensorFilterProperties;

/**
 * @brief Parse JSON configuration file for test setup
 *
 * Parses a JSON configuration file and populates the TestGstTensorFilterProperties
 * structure with the configuration parameters.
 *
 * @param json_path Path to the JSON configuration file
 * @param prop Pointer to TestGstTensorFilterProperties structure to populate
 * @return HAL_ML_ERROR_NONE on success, error code on failure
 */
int parse_json_file(char * json_path, TestGstTensorFilterProperties *prop);

/**
 * @brief Get the test configuration
 *
 * Returns a pointer to the global test configuration structure.
 * This is used by test cases to access the parsed configuration.
 *
 * @return Pointer to TestGstTensorFilterProperties structure
 */
TestGstTensorFilterProperties* get_test_config();

/**
 * @brief Set the test configuration
 *
 * Sets the global test configuration structure pointer.
 * This is called by main() after parsing the JSON configuration.
 *
 * @param config Pointer to TestGstTensorFilterProperties structure to set as global config
 */
void set_test_config(TestGstTensorFilterProperties *config);

#ifdef __cplusplus
}

/**
 * @brief Base test fixture for ML backend tests
 *
 * This fixture provides a common base class for ML backend tests.
 * Tests access the global test configuration through get_test_config().
 */
class MLBackendTest : public ::testing::Test {
protected:
    // No setup/teardown needed - tests use get_test_config() for configuration
};

#endif // __cplusplus

/**
 * @brief Allocate and load test buffers for inference
 *
 * Allocates input and output buffers, loads data from files if available,
 * or fills with zeros if no input files are provided.
 *
 * @param input Input buffer array to allocate and load
 * @param output Output buffer array to allocate
 * @param in_info Input tensor info
 * @param out_info Output tensor info
 * @param prop Test properties containing input data file paths
 */
static inline void
allocate_and_load_test_buffers (GstTensorMemory *input, GstTensorMemory *output,
                                 GstTensorsInfo *in_info, GstTensorsInfo *out_info,
                                 TestGstTensorFilterProperties *prop)
{
  /* Allocate and load input buffers */
  for (guint i = 0; i < in_info->num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (in_info, i);
    input[i].size = gst_tensor_info_get_size (info);
    input[i].data = g_malloc (input[i].size);
    ASSERT_NE (input[i].data, nullptr);

    /* Load raw data from file if provided */
    if (prop->input_data_files && i < prop->num_input_files) {
      const gchar *input_file = prop->input_data_files[i];
      FILE *fp = fopen (input_file, "rb");
      if (fp) {
        size_t bytes_read = fread (input[i].data, 1, input[i].size, fp);
        fclose (fp);

        ASSERT_EQ (bytes_read, input[i].size)
            << "Read " << bytes_read << " bytes, expected " << input[i].size
            << " bytes from file: " << input_file;
      } else {
        /* File open failed, use zero-filled buffer */
        memset (input[i].data, 0, input[i].size);
      }
    } else {
      /* No input file provided, use zero-filled buffer */
      memset (input[i].data, 0, input[i].size);
    }
  }

  /* Allocate output buffers */
  for (guint i = 0; i < out_info->num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (out_info, i);
    output[i].size = gst_tensor_info_get_size (info);
    output[i].data = g_malloc0 (output[i].size);
    ASSERT_NE (output[i].data, nullptr);
  }
}

/**
 * @brief Free allocated test buffers
 *
 * Frees all input and output buffers that were allocated for testing.
 * Safely handles null pointers.
 *
 * @param input Input buffer array to free
 * @param output Output buffer array to free
 * @param in_info Input tensor info (for num_tensors)
 * @param out_info Output tensor info (for num_tensors)
 */
static inline void
free_test_buffers (GstTensorMemory *input, GstTensorMemory *output,
                   GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  /* Free input buffers */
  if (input) {
    for (guint i = 0; i < in_info->num_tensors; i++) {
      if (input[i].data) {
        g_free (input[i].data);
        input[i].data = nullptr;
      }
    }
  }

  /* Free output buffers */
  if (output) {
    for (guint i = 0; i < out_info->num_tensors; i++) {
      if (output[i].data) {
        g_free (output[i].data);
        output[i].data = nullptr;
      }
    }
  }
}

#endif /* __HAL_BACKEND_ML_TEST_UTIL_H__ */
