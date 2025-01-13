/* SPDX-License-Identifier: Apache-2.0 */

#include <stdexcept>

#include <glib.h>

#include <hal-common-interface.h>
#include <hal-ml-interface.h>

#include "hal-backend-ml-util.h"


typedef struct _pass_handle_s
{
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;
} pass_handle_s;

static int ml_pass_init(void **backend_private)
{
  pass_handle_s *pass = g_new0 (pass_handle_s, 1);
  gst_tensors_info_init (&pass->inputInfo);
  gst_tensors_info_init (&pass->outputInfo);
  *backend_private = pass;
  return 0;
}

static int ml_pass_deinit(void *backend_private)
{
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[pass backend] ml_pass_deinit called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  gst_tensors_info_free (&pass->inputInfo);
  gst_tensors_info_free (&pass->outputInfo);

  g_free (pass);

  return HAL_ML_ERROR_NONE;
}

static int ml_pass_configure_instance(void *backend_private, const GstTensorFilterProperties *prop)
{
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[pass backend] ml_pass_configure_instance called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }


  gst_tensors_info_copy (&pass->inputInfo, &prop->input_meta);
  gst_tensors_info_copy (&pass->outputInfo, &prop->output_meta);

  return 0;
}

static int ml_pass_invoke(void *backend_private, const GstTensorMemory *input, GstTensorMemory *output)
{
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[pass backend] ml_pass_invoke called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }
  g_warning ("skip rungraph.. 됐다 치고 하하"); g_usleep (1000 * 100);
  // pass->result_vsi_nn_RunGraph (pass->graph);
  return 0;
}

static int ml_pass_get_model_info(void *backend_private, model_info_ops ops, GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  pass_handle_s *pass = (pass_handle_s *) backend_private;
  if (!pass) {
    g_critical ("[pass backend] ml_pass_get_model_info called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  gst_tensors_info_copy (in_info, &pass->inputInfo);
  gst_tensors_info_copy (out_info, &pass->outputInfo);

  return 0;
}

static int ml_pass_event_handler(void *backend_private, event_ops ops, GstTensorFilterFrameworkEventData *data)
{
  return -ENOENT;
}

static int ml_pass_hal_backend_init(void **data)
{
  hal_backend_ml_funcs *funcs = NULL;

  if (*data) {
    funcs = (hal_backend_ml_funcs *) *data;
  } else {
    funcs = g_new0 (hal_backend_ml_funcs, 1);
  }
  *data = (void *) funcs;

  funcs->init = ml_pass_init;
  funcs->deinit = ml_pass_deinit;
  funcs->configure_instance = ml_pass_configure_instance;
  funcs->invoke = ml_pass_invoke;
  funcs->get_model_info = ml_pass_get_model_info;
  funcs->event_handler = ml_pass_event_handler;

  return 0;
}

static int ml_pass_hal_backend_exit(void *data)
{
  memset (data, 0x0, sizeof(hal_backend_ml_funcs));
  return 0;
}

hal_backend hal_backend_ml_data = {
  .name = "ml-pass",
  .vendor = "YONGJOO",
  .init = ml_pass_hal_backend_init,
  .exit = ml_pass_hal_backend_exit,
  .major_version = 1,
  .minor_version = 1,
};
