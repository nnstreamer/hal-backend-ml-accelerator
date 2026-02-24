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
#include "hal-backend-ml-vivante.cc"

// ===================================================================
// Basic Lifecycle Tests
// ===================================================================

TEST_F(MLBackendTest, Vivante_InitAndExit) {
    void* hal_data = nullptr;

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    EXPECT_NE(hal_data, nullptr);

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

TEST_F(MLBackendTest, Vivante_configure_instance) {
    void* hal_data = nullptr;
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    // Initialize the backend
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    EXPECT_NE(hal_data, nullptr);

    // Configure the instance with properties
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_configure_instance(hal_data, &test_config->base));

    // Deinitialize the backend
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

TEST_F(MLBackendTest, Vivante_inference) {
    void* hal_data = nullptr;
    GstTensorMemory input[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorMemory output[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    // Initialize the backend
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    ASSERT_NE(hal_data, nullptr);

    // Configure the instance with properties
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_configure_instance(hal_data, &test_config->base));

    // Get input/output tensor info
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_get_model_info(hal_data, GET_IN_OUT_INFO, &in_info, &out_info));

    // Allocate and load input/output buffers
    allocate_and_load_test_buffers(input, output, &in_info, &out_info, test_config);

    // Perform inference
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_invoke(hal_data, input, output));

    // Free input and output buffers
    free_test_buffers(input, output, &in_info, &out_info);

    // Free tensor info memory
    gst_tensors_info_free(&in_info);
    gst_tensors_info_free(&out_info);

    // Deinitialize the backend
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

// ===================================================================
// Framework Info Tests
// ===================================================================

TEST_F(MLBackendTest, Vivante_get_framework_info) {
    void* hal_data = nullptr;
    GstTensorFilterFrameworkInfo fw_info = {0};

    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_get_framework_info(hal_data, &fw_info));
    EXPECT_STREQ("vivante", fw_info.name);
    EXPECT_FALSE(fw_info.allow_in_place);
    EXPECT_FALSE(fw_info.allocate_in_invoke);
    EXPECT_FALSE(fw_info.run_without_model);
    EXPECT_FALSE(fw_info.verify_model_path);
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

// ===================================================================
// Model Info Tests
// ===================================================================

TEST_F(MLBackendTest, Vivante_get_model_info) {
    void* hal_data = nullptr;
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_configure_instance(hal_data, &test_config->base));

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_get_model_info(hal_data, GET_IN_OUT_INFO, &in_info, &out_info));
    EXPECT_GT(in_info.num_tensors, 0);
    EXPECT_GT(out_info.num_tensors, 0);

    // Free tensor info memory
    gst_tensors_info_free(&in_info);
    gst_tensors_info_free(&out_info);

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

// ===================================================================
// Event Handler Tests
// ===================================================================

TEST_F(MLBackendTest, Vivante_event_handler_not_supported) {
    void* hal_data = nullptr;
    
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    
    EXPECT_EQ(HAL_ML_ERROR_NOT_SUPPORTED, ml_vivante_event_handler(hal_data, 0, nullptr));
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

// ===================================================================
// Error Handling Tests - NULL Parameters
// ===================================================================

TEST(VivanteTest, DeinitWithNullParameter) {
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_vivante_deinit(nullptr));
}

TEST(VivanteTest, ConfigureInstanceWithNullBackend) {
    GstTensorFilterProperties test_prop = {0};
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_vivante_configure_instance(nullptr, &test_prop));
}

TEST(VivanteTest, ConfigureInstanceWithNullProperties) {
    void* hal_data = nullptr;
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_vivante_configure_instance(hal_data, nullptr));
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}

TEST(VivanteTest, InvokeWithNullBackend) {
    GstTensorMemory input[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorMemory output[NNS_TENSOR_MEMORY_MAX] = {0};
    
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, ml_vivante_invoke(nullptr, input, output));
}

TEST(VivanteTest, GetModelInfoWithNullBackend) {
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    
    EXPECT_EQ(HAL_ML_ERROR_INVALID_PARAMETER, 
              ml_vivante_get_model_info(nullptr, GET_IN_OUT_INFO, &in_info, &out_info));
}

TEST(VivanteTest, GetFrameworkInfoWithNullBackend) {
    GstTensorFilterFrameworkInfo fw_info = {0};
    
    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_get_framework_info(nullptr, &fw_info));
    EXPECT_STREQ("vivante", fw_info.name);
}

// ===================================================================
// Static Helper Function Tests - Type Conversions
// ===================================================================

TEST(VivanteTest, VivanteVsiTypeFromString) {
    EXPECT_EQ(VSI_NN_TYPE_INT8, vivante_vsi_type_from_string("VSI_NN_TYPE_INT8"));
    EXPECT_EQ(VSI_NN_TYPE_UINT8, vivante_vsi_type_from_string("VSI_NN_TYPE_UINT8"));
    EXPECT_EQ(VSI_NN_TYPE_INT16, vivante_vsi_type_from_string("VSI_NN_TYPE_INT16"));
    EXPECT_EQ(VSI_NN_TYPE_UINT16, vivante_vsi_type_from_string("VSI_NN_TYPE_UINT16"));
    EXPECT_EQ(VSI_NN_TYPE_INT32, vivante_vsi_type_from_string("VSI_NN_TYPE_INT32"));
    EXPECT_EQ(VSI_NN_TYPE_UINT32, vivante_vsi_type_from_string("VSI_NN_TYPE_UINT32"));
    EXPECT_EQ(VSI_NN_TYPE_INT64, vivante_vsi_type_from_string("VSI_NN_TYPE_INT64"));
    EXPECT_EQ(VSI_NN_TYPE_UINT64, vivante_vsi_type_from_string("VSI_NN_TYPE_UINT64"));
    EXPECT_EQ(VSI_NN_TYPE_FLOAT16, vivante_vsi_type_from_string("VSI_NN_TYPE_FLOAT16"));
    EXPECT_EQ(VSI_NN_TYPE_FLOAT32, vivante_vsi_type_from_string("VSI_NN_TYPE_FLOAT32"));
    EXPECT_EQ(VSI_NN_TYPE_FLOAT64, vivante_vsi_type_from_string("VSI_NN_TYPE_FLOAT64"));
    EXPECT_EQ(VSI_NN_TYPE_BFLOAT16, vivante_vsi_type_from_string("VSI_NN_TYPE_BFLOAT16"));
    EXPECT_EQ(VSI_NN_TYPE_BOOL8, vivante_vsi_type_from_string("VSI_NN_TYPE_BOOL8"));
    
    // Case insensitive
    EXPECT_EQ(VSI_NN_TYPE_FLOAT32, vivante_vsi_type_from_string("vsi_nn_type_float32"));
    EXPECT_EQ(VSI_NN_TYPE_FLOAT32, vivante_vsi_type_from_string("VSI_NN_TYPE_FLOAT32"));
    
    // NULL or unknown
    EXPECT_EQ(VSI_NN_TYPE_NONE, vivante_vsi_type_from_string(nullptr));
    EXPECT_EQ(VSI_NN_TYPE_NONE, vivante_vsi_type_from_string("UNKNOWN_TYPE"));
}

TEST(VivanteTest, VivanteQntTypeFromString) {
    EXPECT_EQ(VSI_NN_QNT_TYPE_NONE, vivante_qnt_type_from_string("VSI_NN_QNT_TYPE_NONE"));
    EXPECT_EQ(VSI_NN_QNT_TYPE_DFP, vivante_qnt_type_from_string("VSI_NN_QNT_TYPE_DFP"));
    EXPECT_EQ(VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC, vivante_qnt_type_from_string("VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC"));
    EXPECT_EQ(VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC, vivante_qnt_type_from_string("VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC"));
    EXPECT_EQ(VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC, vivante_qnt_type_from_string("VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC"));
    
    // Case insensitive
    EXPECT_EQ(VSI_NN_QNT_TYPE_DFP, vivante_qnt_type_from_string("vsi_nn_qnt_type_dfp"));
    
    // NULL or unknown
    EXPECT_EQ(VSI_NN_QNT_TYPE_NONE, vivante_qnt_type_from_string(nullptr));
    EXPECT_EQ(VSI_NN_QNT_TYPE_NONE, vivante_qnt_type_from_string("UNKNOWN_QNT_TYPE"));
}

TEST(VivanteTest, ConvertToTensorType) {
    EXPECT_EQ(_NNS_INT8, convert_to_tensor_type(VSI_NN_TYPE_INT8));
    EXPECT_EQ(_NNS_INT8, convert_to_tensor_type(VSI_NN_TYPE_BOOL8));
    EXPECT_EQ(_NNS_UINT8, convert_to_tensor_type(VSI_NN_TYPE_UINT8));
    EXPECT_EQ(_NNS_INT16, convert_to_tensor_type(VSI_NN_TYPE_INT16));
    EXPECT_EQ(_NNS_UINT16, convert_to_tensor_type(VSI_NN_TYPE_UINT16));
    EXPECT_EQ(_NNS_INT32, convert_to_tensor_type(VSI_NN_TYPE_INT32));
    EXPECT_EQ(_NNS_UINT32, convert_to_tensor_type(VSI_NN_TYPE_UINT32));
    EXPECT_EQ(_NNS_INT64, convert_to_tensor_type(VSI_NN_TYPE_INT64));
    EXPECT_EQ(_NNS_UINT64, convert_to_tensor_type(VSI_NN_TYPE_UINT64));
    EXPECT_EQ(_NNS_FLOAT16, convert_to_tensor_type(VSI_NN_TYPE_FLOAT16));
    EXPECT_EQ(_NNS_FLOAT16, convert_to_tensor_type(VSI_NN_TYPE_BFLOAT16));
    EXPECT_EQ(_NNS_FLOAT32, convert_to_tensor_type(VSI_NN_TYPE_FLOAT32));
    EXPECT_EQ(_NNS_FLOAT64, convert_to_tensor_type(VSI_NN_TYPE_FLOAT64));
    
    // Unknown type
    EXPECT_EQ(_NNS_END, convert_to_tensor_type(VSI_NN_TYPE_NONE));
}

// ===================================================================
// Multiple Inference Tests
// ===================================================================

TEST_F(MLBackendTest, Vivante_multiple_inferences) {
    void* hal_data = nullptr;
    GstTensorMemory input[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorMemory output[NNS_TENSOR_MEMORY_MAX] = {0};
    GstTensorsInfo in_info = {0};
    GstTensorsInfo out_info = {0};
    TestGstTensorFilterProperties* test_config = get_test_config();
    ASSERT_NE(test_config, nullptr) << "Test configuration not initialized";

    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_init(&hal_data));
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_configure_instance(hal_data, &test_config->base));
    ASSERT_EQ(HAL_ML_ERROR_NONE, ml_vivante_get_model_info(hal_data, GET_IN_OUT_INFO, &in_info, &out_info));

    // Perform multiple inferences
    for (int i = 0; i < 3; i++) {
        allocate_and_load_test_buffers(input, output, &in_info, &out_info, test_config);
        EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_invoke(hal_data, input, output));
        free_test_buffers(input, output, &in_info, &out_info);
    }

    // Free tensor info memory
    gst_tensors_info_free(&in_info);
    gst_tensors_info_free(&out_info);

    EXPECT_EQ(HAL_ML_ERROR_NONE, ml_vivante_deinit(hal_data));
}
