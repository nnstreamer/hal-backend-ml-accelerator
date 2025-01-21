/* SPDX-License-Identifier: Apache-2.0 */

#include <stdexcept>

#include <glib.h>
#include <dlfcn.h>

#include <hal-common-interface.h>
#include <hal-ml-interface.h>

#include <ovx/vsi_nn_pub.h>

#include "hal-backend-ml-util.h"


static tensor_type convert_tensortype (vsi_nn_type_e tensor_type)
{
  switch (tensor_type) {
    case VSI_NN_TYPE_INT8:
      return _NNS_INT8;
    case VSI_NN_TYPE_UINT8:
      return _NNS_UINT8;
    case VSI_NN_TYPE_INT16:
      return _NNS_INT16;
    case VSI_NN_TYPE_UINT16:
      return _NNS_UINT16;
    case VSI_NN_TYPE_FLOAT16:
#ifdef FLOAT16_SUPPORT
      return _NNS_FLOAT16;
#else
      return _NNS_UINT16;
#endif
    case VSI_NN_TYPE_FLOAT32:
      return _NNS_FLOAT32;
    default:
      break;
  }
  return _NNS_END;
}

typedef struct _vivante_handle_s
{
  char *model_path;
  char *so_path;
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  vsi_nn_graph_t *graph;
  void *handle; /* dlopened model so */
  vsi_status (*result_vsi_nn_CopyDataToTensor) (vsi_nn_graph_t *, vsi_nn_tensor_t *, uint8_t *);
  void (*result_vnn_ReleaseNeuralNetwork) (vsi_nn_graph_t *);
  vsi_nn_graph_t *(*result_vnn_CreateNeuralNetwork) (const char *);
  vsi_status (*result_vsi_nn_RunGraph) (vsi_nn_graph_t *);
  int postProcess;
  vsi_status (*postProcessFunc) (vsi_nn_graph_t * graph);
} vivante_handle_s;

static int ml_vivante_init(void **backend_private)
{
  vivante_handle_s *vivante = g_new0 (vivante_handle_s, 1);
  gst_tensors_info_init (&vivante->inputInfo);
  gst_tensors_info_init (&vivante->outputInfo);
  *backend_private = vivante;
  return 0;
}

static int ml_vivante_deinit(void *backend_private)
{
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;
  if (!vivante) {
    g_error ("[vivante backend] ml_vivante_deinit called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (vivante->graph)
    vivante->result_vnn_ReleaseNeuralNetwork (vivante->graph);

  if (vivante->handle)
    dlclose (vivante->handle);

  gst_tensors_info_free (&vivante->inputInfo);
  gst_tensors_info_free (&vivante->outputInfo);

  g_free (vivante->model_path);
  g_free (vivante->so_path);
  g_free (vivante);

  return HAL_ML_ERROR_NONE;
}

static int ml_vivante_configure_instance(void *backend_private, const void *prop_)
{
  const GstTensorFilterProperties *prop = (const GstTensorFilterProperties *) prop_;
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;
  if (!vivante) {
    g_error ("[vivante backend] ml_vivante_configure_instance called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  vivante->model_path = g_strdup(prop->model_files[0]);
  vivante->so_path = g_strdup(prop->model_files[1]);

  vivante->handle = dlopen (vivante->so_path, RTLD_NOW);
  if (!vivante->handle) {
    g_error ("Failed to load shared library: %s", vivante->so_path);
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  vivante->result_vsi_nn_CopyDataToTensor = (vsi_status (*)(vsi_nn_graph_t *, vsi_nn_tensor_t *, uint8_t *)) dlsym(vivante->handle, "vsi_nn_CopyDataToTensor");
  vivante->result_vnn_ReleaseNeuralNetwork = (void (*)(vsi_nn_graph_t *)) dlsym(vivante->handle, "vnn_ReleaseNeuralNetwork");
  vivante->result_vnn_CreateNeuralNetwork = (vsi_nn_graph_t *(*)(const char *)) dlsym(vivante->handle, "vnn_CreateNeuralNetwork");
  vivante->result_vsi_nn_RunGraph = (vsi_status (*)(vsi_nn_graph_t *)) dlsym(vivante->handle, "vsi_nn_RunGraph");

  if (vivante->postProcess) {
    vivante->postProcessFunc = (vsi_status (*)(vsi_nn_graph_t *)) dlsym(vivante->handle, "vnn_PostProcessNeuralNetwork");
  }

  vivante->graph = vivante->result_vnn_CreateNeuralNetwork(vivante->model_path);

  /* setting input and output tensors info */
  gst_tensors_info_init (&vivante->inputInfo);
  gst_tensors_info_init (&vivante->outputInfo);

  vivante->inputInfo.num_tensors = vivante->graph->input.num;
  for (unsigned int i = 0; i < vivante->graph->input.num; i++) {
    vsi_nn_tensor_t *i_tensor = vsi_nn_GetTensor (vivante->graph, vivante->graph->input.tensors[i]);
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&vivante->inputInfo, i);

    info->type = convert_tensortype (i_tensor->attr.dtype.vx_type);
    info->name = g_strdup_printf ("%i", vivante->graph->input.tensors[i]);
    for (unsigned int j = 0; j < i_tensor->attr.dim_num; ++j) {
      info->dimension[j] = i_tensor->attr.size[j];
    }
  }

  vivante->outputInfo.num_tensors = vivante->graph->output.num;
  for (unsigned int i = 0; i < vivante->graph->output.num; i++) {
    vsi_nn_tensor_t *o_tensor = vsi_nn_GetTensor (vivante->graph, vivante->graph->output.tensors[i]);
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&vivante->outputInfo, i);

    info->type = convert_tensortype (o_tensor->attr.dtype.vx_type);
    info->name = g_strdup_printf ("%i", vivante->graph->output.tensors[i]);
    for (unsigned int j = 0; j < o_tensor->attr.dim_num; ++j) {
      info->dimension[j] = o_tensor->attr.size[j];
    }
  }

  return 0;
}

static int ml_vivante_invoke(void *backend_private, const void *input_, void *output_)
{
  const GstTensorMemory *input = (const GstTensorMemory *) input_;
  GstTensorMemory *output = (GstTensorMemory *) output_;
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;
  if (!vivante) {
    g_error ("[vivante backend] ml_vivante_invoke called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  for (unsigned int i = 0; i < vivante->graph->input.num; i++) {
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor (vivante->graph, vivante->graph->input.tensors[i]);
    vivante->result_vsi_nn_CopyDataToTensor (vivante->graph, tensor, (uint8_t *) input[i].data);
  }

  vivante->result_vsi_nn_RunGraph (vivante->graph);

  if (vivante->postProcess)
    vivante->postProcessFunc (vivante->graph);

  for (unsigned int i = 0; i < vivante->graph->output.num; i++) {
    vsi_nn_tensor_t *out_tensor = vsi_nn_GetTensor (vivante->graph, vivante->graph->output.tensors[i]);
    vsi_nn_CopyTensorToBuffer (vivante->graph, out_tensor, output[i].data);
  }

  return 0;
}

static int
ml_vivante_get_framework_info (void *backend_private, void *fw_info)
{
  GstTensorFilterFrameworkInfo *info = (GstTensorFilterFrameworkInfo *) fw_info;
  info->name = "vivante-tizen-hal";
  info->allow_in_place = FALSE;
  info->allocate_in_invoke = FALSE;
  info->run_without_model = FALSE;
  info->verify_model_path = FALSE;

  return HAL_ML_ERROR_NONE;
}

static int ml_vivante_get_model_info(void *backend_private, int ops_, void *in_info_, void *out_info_)
{
  int ops = (model_info_ops) ops_;
  GstTensorsInfo *in_info = (GstTensorsInfo *) in_info_;
  GstTensorsInfo *out_info = (GstTensorsInfo *) out_info_;

  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;
  if (!vivante) {
    g_error ("[vivante backend] ml_vivante_get_model_info called with invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  gst_tensors_info_copy (in_info, &vivante->inputInfo);
  gst_tensors_info_copy (out_info, &vivante->outputInfo);

  return 0;
}

static int ml_vivante_event_handler(void *backend_private, int ops_, void *data_)
{
  int ops = (event_ops) ops_;
  GstTensorFilterFrameworkEventData *data = (GstTensorFilterFrameworkEventData *) data_;

  return HAL_ML_ERROR_NOT_SUPPORTED;
}

static int ml_vivante_hal_backend_init(void **data)
{
  hal_backend_ml_funcs *funcs = NULL;

  if (*data) {
    funcs = (hal_backend_ml_funcs *) *data;
  } else {
    funcs = g_new0 (hal_backend_ml_funcs, 1);
  }
  *data = (void *) funcs;

  funcs->init = ml_vivante_init;
  funcs->deinit = ml_vivante_deinit;
  funcs->configure_instance = ml_vivante_configure_instance;
  funcs->invoke = ml_vivante_invoke;
  funcs->get_framework_info = ml_vivante_get_framework_info;
  funcs->get_model_info = ml_vivante_get_model_info;
  funcs->event_handler = ml_vivante_event_handler;

  return 0;
}

static int ml_vivante_hal_backend_exit(void *data)
{
  memset (data, 0x0, sizeof(hal_backend_ml_funcs));
  return 0;
}

hal_backend hal_backend_ml_data = {
  .name = "ml-vivante",
  .vendor = "YONGJOO",
  .init = ml_vivante_hal_backend_init,
  .exit = ml_vivante_hal_backend_exit,
  .major_version = 1,
  .minor_version = 1,
};
