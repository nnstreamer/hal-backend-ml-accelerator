/* SPDX-License-Identifier: Apache-2.0 */

#include <glib.h>
#include <stdexcept>
#include <vector>

#include <hal-common-interface.h>
#include <hal-ml-interface.h>

#include <DlContainer/DlContainer.h>
#include <DlSystem/DlEnums.h>
#include <DlSystem/DlError.h>
#include <DlSystem/DlVersion.h>
#include <DlSystem/IUserBuffer.h>
#include <DlSystem/RuntimeList.h>
#include <DlSystem/UserBufferMap.h>
#include <SNPE/SNPE.h>
#include <SNPE/SNPEBuilder.h>
#include <SNPE/SNPEUtil.h>

#include "hal-backend-ml-util.h"


typedef struct _snpe_handle_s {
  char *model_path;
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  Snpe_SNPE_Handle_t snpe_h;
  Snpe_UserBufferMap_Handle_t inputMap_h;
  Snpe_UserBufferMap_Handle_t outputMap_h;
  std::vector<Snpe_IUserBuffer_Handle_t> user_buffers;
} snpe_handle_s;

static int
ml_snpe_init (void **backend_private)
{
  snpe_handle_s *snpe = g_new0 (snpe_handle_s, 1);

  gst_tensors_info_init (&snpe->inputInfo);
  gst_tensors_info_init (&snpe->outputInfo);

  *backend_private = snpe;
  return 0;
}

static int
ml_snpe_deinit (void *backend_private)
{
  snpe_handle_s *snpe = (snpe_handle_s *) backend_private;
  if (!snpe) {
    g_critical ("[snpe backend] ml_snpe_deinit called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (snpe->inputMap_h)
    Snpe_UserBufferMap_Delete (snpe->inputMap_h);

  if (snpe->outputMap_h)
    Snpe_UserBufferMap_Delete (snpe->outputMap_h);

  for (auto &ub : snpe->user_buffers)
    if (ub)
      Snpe_IUserBuffer_Delete (ub);

  snpe->user_buffers.clear ();

  if (snpe->snpe_h)
    Snpe_SNPE_Delete (snpe->snpe_h);

  if (snpe->model_path)
    g_free (snpe->model_path);

  gst_tensors_info_free (&snpe->inputInfo);
  gst_tensors_info_free (&snpe->outputInfo);

  g_free (snpe);
  return 0;
}

static int
ml_snpe_configure_instance (void *backend_private, const void *prop_)
{
  const GstTensorFilterProperties *prop = (const GstTensorFilterProperties *) prop_;
  snpe_handle_s *snpe = (snpe_handle_s *) backend_private;
  if (!snpe) {
    g_critical ("[snpe backend] ml_snpe_configure_instance called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  Snpe_DlVersion_Handle_t lib_version_h = NULL;
  Snpe_RuntimeList_Handle_t runtime_list_h = NULL;
  Snpe_DlContainer_Handle_t container_h = NULL;
  Snpe_SNPEBuilder_Handle_t snpebuilder_h = NULL;
  Snpe_StringList_Handle_t inputstrListHandle = NULL;
  Snpe_StringList_Handle_t outputstrListHandle = NULL;
  std::vector<Snpe_UserBufferEncoding_ElementType_t> inputTypeVec;
  std::vector<Snpe_UserBufferEncoding_ElementType_t> outputTypeVec;

  auto _clean_handles = [&] () {
    if (lib_version_h)
      Snpe_DlVersion_Delete (lib_version_h);
    if (runtime_list_h)
      Snpe_RuntimeList_Delete (runtime_list_h);
    if (container_h)
      Snpe_DlContainer_Delete (container_h);
    if (snpebuilder_h)
      Snpe_SNPEBuilder_Delete (snpebuilder_h);
    if (inputstrListHandle)
      Snpe_StringList_Delete (inputstrListHandle);
    if (outputstrListHandle)
      Snpe_StringList_Delete (outputstrListHandle);
  };

  /* default runtime is CPU */
  Snpe_Runtime_t runtime = SNPE_RUNTIME_CPU;

  /* lambda function to handle tensor */
  auto handleTensor = [&] (const char *tensorName, GstTensorInfo *info,
                          Snpe_UserBufferMap_Handle_t bufferMapHandle,
                          Snpe_UserBufferEncoding_ElementType_t type) {
    Snpe_IBufferAttributes_Handle_t bufferAttributesOpt
        = Snpe_SNPE_GetInputOutputBufferAttributes (snpe->snpe_h, tensorName);
    if (!bufferAttributesOpt)
      throw std::runtime_error ("Error obtaining buffer attributes");

    auto default_type = Snpe_IBufferAttributes_GetEncodingType (bufferAttributesOpt);

    /* parse tensor data type with user given element type */
    switch (type) {
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN:
        /* If the type is not provided by user, use default type */
        type = default_type;
        if (default_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) {
          info->type = _NNS_FLOAT32;
        } else if (default_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8) {
          info->type = _NNS_UINT8;
        } else {
          throw std::invalid_argument ("Unsupported data type");
        }
        break;
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
        info->type = _NNS_FLOAT32;
        break;
      case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
        info->type = _NNS_UINT8;
        if (default_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) {
          throw std::invalid_argument (
              "ERROR: Quantization parameters are not present in model. Use TF8 type.");
        }
        break;
      default:
        throw std::invalid_argument ("Unsupported data type");
    }

    /* parse tensor dimension */
    auto shapeHandle = Snpe_IBufferAttributes_GetDims (bufferAttributesOpt);
    auto rank = Snpe_TensorShape_Rank (shapeHandle);
    const size_t *sdims = Snpe_TensorShape_GetDimensions (shapeHandle);
    for (size_t j = 0; j < rank; j++) {
      info->dimension[rank - 1 - j] = sdims[j];
    }

    /* calculate strides */
    std::vector<size_t> strides (rank);
    strides[rank - 1] = gst_tensor_get_element_size (info->type);
    for (size_t j = rank - 1; j > 0; j--) {
      strides[j - 1] = strides[j] * sdims[j];
    }

    auto stride_h = Snpe_TensorShape_CreateDimsSize (strides.data (), strides.size ());
    Snpe_TensorShape_Delete (shapeHandle);
    Snpe_IBufferAttributes_Delete (bufferAttributesOpt);

    /* assign user_buffermap */
    size_t bufsize = gst_tensor_info_get_size (info);
    Snpe_UserBufferEncoding_Handle_t ube_h = NULL;
    if (type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8) {
      Snpe_IBufferAttributes_Handle_t bufferAttributesOpt
          = Snpe_SNPE_GetInputOutputBufferAttributes (snpe->snpe_h, tensorName);
      Snpe_UserBufferEncoding_Handle_t ubeTfNHandle
          = Snpe_IBufferAttributes_GetEncoding_Ref (bufferAttributesOpt);
      uint64_t stepEquivalentTo0 = Snpe_UserBufferEncodingTfN_GetStepExactly0 (ubeTfNHandle);
      float quantizedStepSize
          = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize (ubeTfNHandle);
      ube_h = Snpe_UserBufferEncodingTfN_Create (stepEquivalentTo0, quantizedStepSize, 8);
      Snpe_IBufferAttributes_Delete (bufferAttributesOpt);
      Snpe_UserBufferEncodingTfN_Delete (ubeTfNHandle);
    } else if (type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) {
      ube_h = Snpe_UserBufferEncodingFloat_Create ();
    }
    auto iub = Snpe_Util_CreateUserBuffer (NULL, bufsize, stride_h, ube_h);
    snpe->user_buffers.push_back (iub);

    if (type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8)
      Snpe_UserBufferEncodingTfN_Delete (ube_h);
    else if (type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT)
      Snpe_UserBufferEncodingFloat_Delete (ube_h);
    Snpe_TensorShape_Delete (stride_h);

    Snpe_UserBufferMap_Add (bufferMapHandle, tensorName, iub);
  };

  auto parse_custom_prop = [&runtime, &outputstrListHandle, &inputTypeVec,
                               &outputTypeVec] (const char *custom_prop) {
    if (!custom_prop)
      return;

    gchar **options = g_strsplit (custom_prop, ",", -1);

    for (guint op = 0; op < g_strv_length (options); ++op) {
      gchar **option = g_strsplit (options[op], ":", -1);

      if (g_strv_length (option) > 1) {
        g_strstrip (option[0]);
        g_strstrip (option[1]);

        if (g_ascii_strcasecmp (option[0], "Runtime") == 0) {
          if (g_ascii_strcasecmp (option[1], "CPU") == 0) {
            runtime = SNPE_RUNTIME_CPU;
          } else if (g_ascii_strcasecmp (option[1], "GPU") == 0) {
            runtime = SNPE_RUNTIME_GPU;
          } else if (g_ascii_strcasecmp (option[1], "DSP") == 0) {
            runtime = SNPE_RUNTIME_DSP;
          } else if (g_ascii_strcasecmp (option[1], "NPU") == 0
                     || g_ascii_strcasecmp (option[1], "AIP") == 0) {
            runtime = SNPE_RUNTIME_AIP_FIXED8_TF;
          } else {
            g_warning ("Unknown runtime (%s), set CPU as default.", options[op]);
          }
        } else if (g_ascii_strcasecmp (option[0], "OutputTensor") == 0) {
          /* the tensor name may contain ':' */
          gchar *_ot_str = g_strjoinv (":", &option[1]);
          gchar **names = g_strsplit (_ot_str, ";", -1);
          guint num_names = g_strv_length (names);
          outputstrListHandle = Snpe_StringList_Create ();
          for (guint i = 0; i < num_names; ++i) {
            if (g_strcmp0 (names[i], "") == 0) {
              throw std::invalid_argument ("Given tensor name is invalid.");
            }

            g_info ("Add output tensor name of %s", names[i]);
            if (Snpe_StringList_Append (outputstrListHandle, names[i]) != SNPE_SUCCESS) {
              const std::string err_msg = "Failed to append output tensor name: "
                                          + (const std::string) names[i];
              throw std::runtime_error (err_msg);
            }
          }
          g_free (_ot_str);
          g_strfreev (names);
        } else if (g_ascii_strcasecmp (option[0], "OutputType") == 0) {
          gchar **types = g_strsplit (option[1], ";", -1);
          guint num_types = g_strv_length (types);
          for (guint i = 0; i < num_types; ++i) {
            if (g_ascii_strcasecmp (types[i], "FLOAT32") == 0) {
              outputTypeVec.push_back (SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT);
            } else if (g_ascii_strcasecmp (types[i], "TF8") == 0) {
              outputTypeVec.push_back (SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8);
            } else {
              g_warning ("Ignore unknown output type (%s)", types[i]);
            }
          }
          g_strfreev (types);
        } else if (g_ascii_strcasecmp (option[0], "InputType") == 0) {
          gchar **types = g_strsplit (option[1], ";", -1);
          guint num_types = g_strv_length (types);
          for (guint i = 0; i < num_types; ++i) {
            if (g_ascii_strcasecmp (types[i], "FLOAT32") == 0) {
              inputTypeVec.push_back (SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT);
            } else if (g_ascii_strcasecmp (types[i], "TF8") == 0) {
              inputTypeVec.push_back (SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8);
            } else {
              g_warning ("Ignore unknown input type (%s)", types[i]);
            }
          }
          g_strfreev (types);
        } else {
          g_warning ("Unknown option (%s).", options[op]);
        }
      }

      g_strfreev (option);
    }

    g_strfreev (options);
  };

  try {
    /* Log SNPE version */
    lib_version_h = Snpe_Util_GetLibraryVersion ();
    if (!lib_version_h)
      throw std::runtime_error ("Failed to get SNPE library version");

    g_info ("SNPE Version: %s", Snpe_DlVersion_ToString (lib_version_h));

    int32_t ver_major = Snpe_DlVersion_GetMajor (lib_version_h);
    if (ver_major < 2) {
      const std::string err_msg = "Invalid SNPE version, version 2.x is supported but has "
                                  + std::to_string (ver_major) + ".x.";
      g_critical ("%s", err_msg.c_str ());
      throw std::runtime_error (err_msg);
    }

    /* parse custom properties */
    parse_custom_prop (prop->custom_properties);

    /* Check the given Runtime is available */
    std::string runtime_str = std::string (Snpe_RuntimeList_RuntimeToString (runtime));
    if (Snpe_Util_IsRuntimeAvailable (runtime) == 0)
      throw std::runtime_error ("Given runtime " + runtime_str + " is not available");

    g_info ("Given runtime %s is available", runtime_str.c_str ());

    /* set runtimelist config */
    runtime_list_h = Snpe_RuntimeList_Create ();
    if (Snpe_RuntimeList_Add (runtime_list_h, runtime) != SNPE_SUCCESS)
      throw std::runtime_error ("Failed to add given runtime to Snpe_RuntimeList");

    /* Load network (dlc file) */
    if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
      const std::string err_msg
          = "Given file " + (std::string) prop->model_files[0] + " is not valid";
      throw std::invalid_argument (err_msg);
    }

    snpe->model_path = g_strdup (prop->model_files[0]);
    container_h = Snpe_DlContainer_Open (snpe->model_path);
    if (!container_h)
      throw std::runtime_error (
          "Failed to open the model file " + std::string (snpe->model_path));

    /* Build SNPE handle */
    snpebuilder_h = Snpe_SNPEBuilder_Create (container_h);
    if (!snpebuilder_h)
      throw std::runtime_error ("Failed to create SNPE builder");

    if (Snpe_SNPEBuilder_SetRuntimeProcessorOrder (snpebuilder_h, runtime_list_h) != SNPE_SUCCESS)
      throw std::runtime_error ("Failed to set runtime processor order");

    /* set UserBuffer mode */
    if (Snpe_SNPEBuilder_SetUseUserSuppliedBuffers (snpebuilder_h, true) != SNPE_SUCCESS)
      throw std::runtime_error ("Failed to set use user supplied buffers");

    /* Set Output Tensors (if given by custom prop) */
    if (outputstrListHandle) {
      if (Snpe_SNPEBuilder_SetOutputTensors (snpebuilder_h, outputstrListHandle) != SNPE_SUCCESS) {
        throw std::runtime_error ("Failed to set output tensors");
      }
    }

    snpe->snpe_h = Snpe_SNPEBuilder_Build (snpebuilder_h);
    if (!snpe->snpe_h)
      throw std::runtime_error ("Failed to build SNPE handle");

    /* set inputTensorsInfo and inputMap */
    snpe->inputMap_h = Snpe_UserBufferMap_Create ();
    inputstrListHandle = Snpe_SNPE_GetInputTensorNames (snpe->snpe_h);
    if (!snpe->inputMap_h || !inputstrListHandle)
      throw std::runtime_error ("Error while setting Input tensors");

    snpe->inputInfo.num_tensors = Snpe_StringList_Size (inputstrListHandle);
    for (size_t i = 0; i < snpe->inputInfo.num_tensors; i++) {
      GstTensorInfo *info
          = gst_tensors_info_get_nth_info (std::addressof (snpe->inputInfo), i);
      const char *inputName = Snpe_StringList_At (inputstrListHandle, i);
      info->name = g_strdup (inputName);

      auto inputType = SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN;

      /* set input type from custom prop if it is provided */
      if (inputTypeVec.size () > i)
        inputType = inputTypeVec[i];
      handleTensor (inputName, info, snpe->inputMap_h, inputType);
    }

    /* set outputTensorsInfo and outputMap */
    snpe->outputMap_h = Snpe_UserBufferMap_Create ();

    /* Get default output tensor names (if not provided by custom prop) */
    if (outputstrListHandle == NULL)
      outputstrListHandle = Snpe_SNPE_GetOutputTensorNames (snpe->snpe_h);

    if (!snpe->outputMap_h || !outputstrListHandle)
      throw std::runtime_error ("Error while setting Output tensors");

    snpe->outputInfo.num_tensors = Snpe_StringList_Size (outputstrListHandle);
    for (size_t i = 0; i < snpe->outputInfo.num_tensors; i++) {
      GstTensorInfo *info
          = gst_tensors_info_get_nth_info (std::addressof (snpe->outputInfo), i);
      const char *outputName = Snpe_StringList_At (outputstrListHandle, i);
      info->name = g_strdup (outputName);

      /* set output type from custom prop if it is provided */
      auto outputType = SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN;
      if (outputTypeVec.size () > i) {
        outputType = outputTypeVec[i];
      }
      handleTensor (outputName, info, snpe->outputMap_h, outputType);
    }

    _clean_handles ();
  } catch (const std::exception &e) {
    _clean_handles ();
    g_critical ("%s", e.what ());
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  return HAL_ML_ERROR_NONE;
}

static int
ml_snpe_invoke (void *backend_private, const void *input_, void *output_)
{
  const GstTensorMemory *input = (const GstTensorMemory *) input_;
  GstTensorMemory *output = (GstTensorMemory *) output_;
  snpe_handle_s *snpe = (snpe_handle_s *) backend_private;
  for (unsigned int i = 0; i < snpe->inputInfo.num_tensors; i++) {
    GstTensorInfo *info
        = gst_tensors_info_get_nth_info (std::addressof (snpe->inputInfo), i);
    auto iub = Snpe_UserBufferMap_GetUserBuffer_Ref (snpe->inputMap_h, info->name);
    Snpe_IUserBuffer_SetBufferAddress (iub, input[i].data);
  }

  for (unsigned int i = 0; i < snpe->outputInfo.num_tensors; i++) {
    GstTensorInfo *info
        = gst_tensors_info_get_nth_info (std::addressof (snpe->outputInfo), i);
    auto iub = Snpe_UserBufferMap_GetUserBuffer_Ref (snpe->outputMap_h, info->name);
    Snpe_IUserBuffer_SetBufferAddress (iub, output[i].data);
  }

  Snpe_SNPE_ExecuteUserBuffers (snpe->snpe_h, snpe->inputMap_h, snpe->outputMap_h);

  return HAL_ML_ERROR_NONE;
}

static int
ml_snpe_get_framework_info (void *backend_private, void *fw_info)
{
  GstTensorFilterFrameworkInfo *info = (GstTensorFilterFrameworkInfo *) fw_info;
  info->name = "snpe";
  info->allow_in_place = FALSE;
  info->allocate_in_invoke = FALSE;
  info->run_without_model = FALSE;
  info->verify_model_path = FALSE;

  return HAL_ML_ERROR_NONE;
}

static int
ml_snpe_get_model_info (void *backend_private, int ops_, void *in_info_, void *out_info_)
{
  int ops = (model_info_ops) ops_;
  GstTensorsInfo *in_info = (GstTensorsInfo *) in_info_;
  GstTensorsInfo *out_info = (GstTensorsInfo *) out_info_;
  snpe_handle_s *snpe = (snpe_handle_s *) backend_private;
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (in_info, &snpe->inputInfo);
    gst_tensors_info_copy (out_info, &snpe->outputInfo);

    return HAL_ML_ERROR_NONE;
  }

  return -2;
}

static int
ml_snpe_event_handler (void *backend_private, int ops_, void *data_)
{
  int ops = (event_ops) ops_;
  GstTensorFilterFrameworkEventData *data = (GstTensorFilterFrameworkEventData *) data_;

  return HAL_ML_ERROR_NOT_SUPPORTED;
}

static int
ml_snpe_hal_backend_init (void **data)
{
  hal_backend_ml_funcs *funcs = NULL;

  if (*data) {
    funcs = (hal_backend_ml_funcs *) *data;
  } else {
    funcs = g_new0 (hal_backend_ml_funcs, 1);
  }
  *data = (void *) funcs;

  funcs->init = ml_snpe_init;
  funcs->deinit = ml_snpe_deinit;
  funcs->configure_instance = ml_snpe_configure_instance;
  funcs->invoke = ml_snpe_invoke;
  funcs->get_framework_info = ml_snpe_get_framework_info;
  funcs->get_model_info = ml_snpe_get_model_info;
  funcs->event_handler = ml_snpe_event_handler;

  return 0;
}

static int
ml_snpe_hal_backend_exit (void *data)
{
  memset (data, 0x0, sizeof (hal_backend_ml_funcs));
  return 0;
}

hal_backend hal_backend_ml_data = {
  .name = "ml-snpe",
  .vendor = "YONGJOO",
  .init = ml_snpe_hal_backend_init,
  .exit = ml_snpe_hal_backend_exit,
  .major_version = 1,
  .minor_version = 1,
};
