/* SPDX-License-Identifier: Apache-2.0 */

#define TESTING 1
#include <stdio.h>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <glib.h>
#include "hal-backend-ml-util.h"
#include "hal_backend_ml_test_util.h"
#include "hal-backend-ml-util.cc"
#include "hal_backend_ml_test_wrapper.h"
#include "hal-backend-ml-dummy-passthrough.cc"

// ===================================================================
// Basic Lifecycle Tests
// ===================================================================

TEST_F(MLBackendTest, DummyPassthrough_InitAndExit) {
    void* hal_data = nullptr;

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    EXPECT_NE(hal_data, nullptr);

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}

TEST_F(MLBackendTest, DummyPassthrough_configure_instance) {
    void* hal_data = nullptr;
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    // Initialize the backend
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    EXPECT_NE(hal_data, nullptr);

    // Configure the instance with properties
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_configure_instance(hal_data, &test_config->base));

    // Deinitialize the backend
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}

TEST_F(MLBackendTest, DummyPassthrough_inference) {
    void* hal_data = nullptr;
    GstTensorMemory input[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorMemory output[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    // Initialize the backend
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    ASSERT_NE(hal_data, nullptr);

    // Configure the instance with properties
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_configure_instance(hal_data, &test_config->base));

    // Get input/output tensor info
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_get_model_info(hal_data, GET_IN_OUT_INFO, &in_info, &out_info));

    // Allocate and load input/output buffers
    allocate_and_load_test_buffers(input, output, &in_info, &out_info, test_config);

    // Perform inference
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_invoke(hal_data, input, output));

    // Free input and output buffers
    free_test_buffers(input, output, &in_info, &out_info);

    // Free tensor info memory
    gst_tensors_info_free(&in_info);
    gst_tensors_info_free(&out_info);

    // Deinitialize the backend
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}

// ===================================================================
// Framework Info Tests
// ===================================================================

TEST_F(MLBackendTest, DummyPassthrough_get_framework_info) {
    void* hal_data = nullptr;
    GstTensorFilterFrameworkInfo fw_info = {0};

    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_get_framework_info(hal_data, &fw_info));
    EXPECT_STREQ("dummy-passthrough", fw_info.name);
    EXPECT_FALSE(fw_info.allow_in_place);
    EXPECT_FALSE(fw_info.allocate_in_invoke);
    EXPECT_FALSE(fw_info.run_without_model);
    EXPECT_FALSE(fw_info.verify_model_path);
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}

// ===================================================================
// Model Info Tests
// ===================================================================

TEST_F(MLBackendTest, DummyPassthrough_get_model_info) {
    void* hal_data = nullptr;
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_configure_instance(hal_data, &test_config->base));

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_get_model_info(hal_data, GET_IN_OUT_INFO, &in_info, &out_info));
    EXPECT_GT(in_info.num_tensors, 0);
    EXPECT_GT(out_info.num_tensors, 0);

    // Free tensor info memory
    gst_tensors_info_free(&in_info);
    gst_tensors_info_free(&out_info);

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}

// ===================================================================
// Event Handler Tests
// ===================================================================

TEST_F(MLBackendTest, DummyPassthrough_event_handler_not_supported) {
    void* hal_data = nullptr;
    
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    
    EXPECT_EQ(HAL_ML_ERROR_NOT_SUPPORTED, ml_dummy_passthrough_event_handler(hal_data, 0, nullptr));
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}

// ===================================================================
// Error Handling Tests - NULL Parameters
// ===================================================================

TEST(DummyPassthroughTest, DeinitWithNullParameter) {
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_dummy_passthrough_deinit(nullptr));
}

TEST(DummyPassthroughTest, ConfigureInstanceWithNullBackend) {
    GstTensorFilterProperties test_prop = {0};
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_dummy_passthrough_configure_instance(nullptr, &test_prop));
}

TEST(DummyPassthroughTest, ConfigureInstanceWithNullProperties) {
    void* hal_data = nullptr;
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_dummy_passthrough_configure_instance(hal_data, nullptr));
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}


TEST(DummyPassthroughTest, InvokeWithNullBackend) {
    GstTensorMemory input[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorMemory output[NNS_TENSOR_MEMORY_MAX] = {0};
    
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_dummy_passthrough_invoke(nullptr, input, output));
}


TEST(DummyPassthroughTest, GetModelInfoWithNullBackend) {
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, 
              ml_dummy_passthrough_get_model_info(nullptr, GET_IN_OUT_INFO, &in_info, &out_info));
}

TEST(DummyPassthroughTest, GetFrameworkInfoWithNullBackend) {
    GstTensorFilterFrameworkInfo fw_info = {0};
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_get_framework_info(nullptr, &fw_info));
    EXPECT_STREQ("dummy-passthrough", fw_info.name);
}

// ===================================================================
// Multiple Inference Tests
// ===================================================================

TEST_F(MLBackendTest, DummyPassthrough_multiple_inferences) {
    void* hal_data = nullptr;
    GstTensorMemory input[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorMemory output[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_init(&hal_data));
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_configure_instance(hal_data, &test_config->base));
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_get_model_info(hal_data, GET_IN_OUT_INFO, &in_info, &out_info));

    // Perform multiple inferences
    for (int i = 0; i < 3; i++) {
        allocate_and_load_test_buffers(input, output, &in_info, &out_info, test_config);
        EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_invoke(hal_data, input, output));
        free_test_buffers(input, output, &in_info, &out_info);
    }

    // Free tensor info memory
    gst_tensors_info_free(&in_info);
    gst_tensors_info_free(&out_info);

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_dummy_passthrough_deinit(hal_data));
}
