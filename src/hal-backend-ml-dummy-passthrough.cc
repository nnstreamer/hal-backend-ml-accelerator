/* SPDX-License-Identifier: Apache-2.0 */

#include <glib.h>
#include <stdexcept>

#include <hal-common-interface.h>
#include <hal-ml-interface.h>

#include "hal-backend-ml-util.h"


typedef struct _pass_handle_s {
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;
} pass_handle_s;

static int
ml_dummy_passthrough_init (void **backend_private)
{
  pass_handle_s *pass = g_new0 (pass_handle_s, 1);
  gst_tensors_info_init (&pass->inputInfo);
  gst_tensors_info_init (&pass->outputInfo);
  *backend_private = pass;

  return HAL_ML_ERROR_NONE;
}

static int
ml_dummy_passthrough_deinit (void *backend_private)
{
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[dummy backend] ml_dummy_passthrough_deinit called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  gst_tensors_info_free (&pass->inputInfo);
  gst_tensors_info_free (&pass->outputInfo);

  g_free (pass);

  return HAL_ML_ERROR_NONE;
}

static int
ml_dummy_passthrough_configure_instance (void *backend_private, const void *prop_)
{
  const GstTensorFilterProperties *prop = (const GstTensorFilterProperties *) prop_;
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[dummy backend] ml_dummy_passthrough_configure_instance called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  gst_tensors_info_copy (&pass->inputInfo, &prop->input_meta);
  gst_tensors_info_copy (&pass->outputInfo, &prop->output_meta);

  return HAL_ML_ERROR_NONE;
}

static int
ml_dummy_passthrough_get_framework_info (void *backend_private, void *fw_info)
{
  GstTensorFilterFrameworkInfo *info = (GstTensorFilterFrameworkInfo *) fw_info;
  info->name = "dummy-passthrough";
  info->allow_in_place = FALSE;
  info->allocate_in_invoke = FALSE;
  info->run_without_model = FALSE;
  info->verify_model_path = FALSE;

  return HAL_ML_ERROR_NONE;
}

static int
ml_dummy_passthrough_invoke (void *backend_private, const void *input_, void *output_)
{
  const GstTensorMemory *input = (const GstTensorMemory *) input_;
  GstTensorMemory *output = (GstTensorMemory *) output_;
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[dummy backend] ml_dummy_passthrough_invoke called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  for (unsigned int i = 0; i < pass->inputInfo.num_tensors; i++) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&pass->inputInfo, i);
    memcpy (output[i].data, input[i].data, gst_tensor_info_get_size (info));
  }

  return HAL_ML_ERROR_NONE;
}

static int
ml_dummy_passthrough_get_model_info (
    void *backend_private, int ops_, void *in_info_, void *out_info_)
{
  int ops = (model_info_ops) ops_;
  GstTensorsInfo *in_info = (GstTensorsInfo *) in_info_;
  GstTensorsInfo *out_info = (GstTensorsInfo *) out_info_;
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[dummy backend] ml_dummy_passthrough_get_model_info called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (in_info, &pass->inputInfo);
    gst_tensors_info_copy (out_info, &pass->outputInfo);

    return HAL_ML_ERROR_NONE;
  }

  return HAL_ML_ERROR_NOT_SUPPORTED;
}

static int
ml_dummy_passthrough_event_handler (void *backend_private, int ops_, void *data_)
{
  return HAL_ML_ERROR_NOT_SUPPORTED;
}

static int
ml_dummy_passthrough_hal_backend_init (void **data)
{
  hal_backend_ml_funcs *funcs = NULL;

  if (*data) {
    funcs = (hal_backend_ml_funcs *) *data;
  } else {
    funcs = g_new0 (hal_backend_ml_funcs, 1);
  }
  *data = (void *) funcs;

  funcs->init = ml_dummy_passthrough_init;
  funcs->deinit = ml_dummy_passthrough_deinit;
  funcs->configure_instance = ml_dummy_passthrough_configure_instance;
  funcs->invoke = ml_dummy_passthrough_invoke;
  funcs->get_framework_info = ml_dummy_passthrough_get_framework_info;
  funcs->get_model_info = ml_dummy_passthrough_get_model_info;
  funcs->event_handler = ml_dummy_passthrough_event_handler;

  return 0;
}

static int
ml_dummy_passthrough_hal_backend_exit (void *data)
{
  memset (data, 0x0, sizeof (hal_backend_ml_funcs));
  return 0;
}

hal_backend hal_backend_ml_data = {
  .name = "ml-dummy-passthrough",
  .vendor = "NNStreamer",
  .init = ml_dummy_passthrough_hal_backend_init,
  .exit = ml_dummy_passthrough_hal_backend_exit,
  .major_version = 1,
  .minor_version = 0,
};
