/* SPDX-License-Identifier: Apache-2.0 */

#include <stdexcept>

#include <dlfcn.h>
#include <glib.h>
#include <json-glib/json-glib.h>

#include <hal-common-interface.h>
#include <hal-ml-interface.h>

#include <ovx/vsi_nn_pub.h>

#include "hal-backend-ml-util.h"


/* Helper to convert VSI type string from JSON to vsi_nn_type_e */
static vsi_nn_type_e
vivante_vsi_type_from_string (const gchar * vsi_type_str)
{
  if (!vsi_type_str) return VSI_NN_TYPE_NONE;

  /** @todo Add 4-bit types when it's available */

  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT8") == 0) return VSI_NN_TYPE_INT8;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT16") == 0) return VSI_NN_TYPE_INT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT32") == 0) return VSI_NN_TYPE_INT32;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT64") == 0) return VSI_NN_TYPE_INT64;

  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT8") == 0) return VSI_NN_TYPE_UINT8;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT16") == 0) return VSI_NN_TYPE_UINT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT32") == 0) return VSI_NN_TYPE_UINT32;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT64") == 0) return VSI_NN_TYPE_UINT64;

  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_FLOAT16") == 0) return VSI_NN_TYPE_FLOAT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_FLOAT32") == 0) return VSI_NN_TYPE_FLOAT32;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_FLOAT64") == 0) return VSI_NN_TYPE_FLOAT64;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_BFLOAT16") == 0) return VSI_NN_TYPE_BFLOAT16;

  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_BOOL8") == 0) return VSI_NN_TYPE_BOOL8;

  g_warning ("[vivante backend] Unknown VSI tensor type string from JSON: %s", vsi_type_str);
  return VSI_NN_TYPE_NONE;
}

/* Helper to convert VSI quantization type string from JSON to vsi_nn_qnt_type_e */
static vsi_nn_qnt_type_e
vivante_qnt_type_from_string (const gchar * qnt_str)
{
  if (!qnt_str) return VSI_NN_QNT_TYPE_NONE;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_NONE") == 0) return VSI_NN_QNT_TYPE_NONE;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_DFP") == 0) return VSI_NN_QNT_TYPE_DFP;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC") == 0) return VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC") == 0) return VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC") == 0) return VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC;

  /** @todo Add VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC when it's available */

  g_warning ("[vivante backend] Unknown VSI quantization type string from JSON: %s", qnt_str);
  return VSI_NN_QNT_TYPE_NONE;
}

static tensor_type
convert_tensortype (vsi_nn_type_e tensor_type)
{
  switch (tensor_type) {
    case VSI_NN_TYPE_BOOL8:
    case VSI_NN_TYPE_INT8:
      return _NNS_INT8;
    case VSI_NN_TYPE_UINT8:
      return _NNS_UINT8;
    case VSI_NN_TYPE_INT16:
      return _NNS_INT16;
    case VSI_NN_TYPE_UINT16:
      return _NNS_UINT16;
    case VSI_NN_TYPE_INT32:
      return _NNS_INT32;
    case VSI_NN_TYPE_UINT32:
      return _NNS_UINT32;
    case VSI_NN_TYPE_INT64:
      return _NNS_INT64;
    case VSI_NN_TYPE_UINT64:
      return _NNS_UINT64;
    case VSI_NN_TYPE_FLOAT16:
    case VSI_NN_TYPE_BFLOAT16:
#ifdef FLOAT16_SUPPORT
      return _NNS_FLOAT16;
#else
      return _NNS_UINT16;
#endif
    case VSI_NN_TYPE_FLOAT32:
      return _NNS_FLOAT32;
    case VSI_NN_TYPE_FLOAT64:
      return _NNS_FLOAT64;
    default:
      break;
  }
  return _NNS_END;
}

typedef struct _vivante_handle_s {
  char *model_path; /* .nb file path */
  char *so_path;    /* .so file path (for .so based model loading) */
  char *json_path;  /* .json file path (for JSON based model loading) */
  gboolean using_json_for_graph_setup;

  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  vsi_nn_graph_t *graph;

  /* handles for json based model loading */
  vsi_nn_context_t ctx;

  /* handles for .so based model loading */
  void *handle; /* dlopened model so */
  vsi_nn_graph_t *(*result_vnn_CreateNeuralNetwork) (const char *);
  void (*result_vnn_ReleaseNeuralNetwork) (vsi_nn_graph_t *);
  vsi_status (*postProcessFunc) (vsi_nn_graph_t *);
  int postProcess; /** @todo Add postProcess custom prop */
} vivante_handle_s;

static int
ml_vivante_init (void **backend_private)
{
  vivante_handle_s *vivante = g_new0 (vivante_handle_s, 1);

  gst_tensors_info_init (&vivante->inputInfo);
  gst_tensors_info_init (&vivante->outputInfo);

  vivante->using_json_for_graph_setup = TRUE;

  *backend_private = vivante;
  return HAL_ML_ERROR_NONE;
}

static int
_json_create_neural_network (vivante_handle_s *self)
{
  self->ctx = vsi_nn_CreateContext ();
  if (self->ctx == NULL) {
    g_critical ("[vivante backend] Failed to create context");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  // 1. Get string from the json file
  g_autofree gchar *json_string = NULL;
  if (!g_file_get_contents (self->json_path, &json_string, NULL, NULL)) {
    g_critical ("[vivante backend] Failed to read JSON file %s", self->json_path);
    return HAL_ML_ERROR_IO_ERROR;
  }

  // 2. Parse json string
  g_autoptr (GError) err = NULL;
  g_autoptr (JsonParser) parser = json_parser_new ();
  if (!parser) {
    g_critical ("[vivante backend] Failed to create JsonParser. OOM?");
    return HAL_ML_ERROR_OUT_OF_MEMORY;
  }

  if (!json_parser_load_from_data (parser, json_string, -1, &err)) {
    g_critical ("[vivante backend] Failed to parse json. error: %s", err ? err->message : "Unknown error");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  // 3. Get the root node and the object
  JsonNode *root;
  JsonObject *root_obj;
  root = json_parser_get_root (parser);
  if (!root) {
    g_critical ("[vivante backend] Failed to get root node");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  if (!JSON_NODE_HOLDS_OBJECT (root)) {
    g_critical ("[vivante backend] Root node is not an object");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  root_obj = json_node_get_object (root);
  if (!root_obj) {
    g_critical ("[vivante backend] Failed to get object from root node");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  // 4. Get the number of inputs and outputs
  guint input_tensors_num = 0U;
  guint output_tensors_num = 0U;

  if (!json_object_has_member (root_obj, "input_tensors")) {
    g_critical ("[vivante backend] Missing necessary key 'input_tensors' in JSON");;
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  JsonArray *input_array = json_object_get_array_member (root_obj, "input_tensors");
  if (!input_array) {
    g_critical ("[vivante backend] Failed to get array from 'input_tensors'");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  input_tensors_num = json_array_get_length (input_array);

  if (!json_object_has_member (root_obj, "output_tensors")) {
    g_critical ("[vivante backend] Missing necessary array 'output_tensors' in JSON");;
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  JsonArray *output_array = json_object_get_array_member (root_obj, "output_tensors");
  if (!output_array) {
    g_critical ("[vivante backend] Failed to get array from 'output_tensors'");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  output_tensors_num = json_array_get_length (output_array);

  // 5. Set up graphs
  vsi_nn_graph_t *graph = vsi_nn_CreateGraph (self->ctx, input_tensors_num + output_tensors_num + output_tensors_num, 1 /* NBG node */);
  if (graph == NULL) {
    g_critical ("[vivante backend] Failed to create graph");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  if (!vsi_nn_SetGraphInputs (graph, NULL, input_tensors_num)) {
    g_critical ("[vivante backend] Failed to set graph inputs");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }
  if (!vsi_nn_SetGraphOutputs (graph, NULL, output_tensors_num)) {
    g_critical ("[vivante backend] Failed to set graph outputs");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  // 6. Set a single vsi_nn_node
  /** @todo Assuming there is only 1 node for each nb model. Find any exceptions. */
  vsi_nn_node_t *node = NULL;
  node = vsi_nn_AddNode(graph, VSI_NN_OP_NBG, input_tensors_num, output_tensors_num, NULL);
  if (!node) {
    g_critical ("[vivante backend] Failed to add node");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  node->uid = 0;
  node->nn_param.nbg.type = VSI_NN_NBG_FILE;
  node->nn_param.nbg.url = self->model_path;

  auto _parse_tensor_attributes_from_json = [&] (JsonObject *tensor_obj, vsi_nn_tensor_attr_t *vsi_attr) -> int {
    memset (vsi_attr, 0, sizeof (vsi_nn_tensor_attr_t));
    if (json_object_has_member (tensor_obj, "id")) {
      // do something?
    }

    // initialize some boolean attributes
    vsi_attr->vtl = FALSE;
    vsi_attr->is_const = FALSE;
    vsi_attr->is_created_from_handle = FALSE;
    vsi_attr->is_handle_malloc_by_ovxlib = FALSE;

    // parse size and dim_num
    if (!json_object_has_member (tensor_obj, "size")) {
    }
    JsonArray *size_array = json_object_get_array_member (tensor_obj, "size");
    if (!size_array) {
      g_critical ("[vivante backend] Failed to get array from 'size' in JSON");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }

    vsi_attr->dim_num = json_array_get_length (size_array);
    if (vsi_attr->dim_num == 0U || vsi_attr->dim_num > VSI_NN_MAX_DIM_NUM) {
      g_critical ("[vivante backend] Invalid value for 'size' in JSON");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }

    for (guint i = 0; i < vsi_attr->dim_num; ++i) {
      gint64 dim = json_array_get_int_element (size_array, i);
      if (dim <= 0) {
        g_critical ("[vivante backend] Invalid value for 'dim' in JSON");
        return HAL_ML_ERROR_INVALID_PARAMETER;
      }

      vsi_attr->size[i] = (vsi_size_t) dim;
    }

    // parse dtype
    if (!json_object_has_member (tensor_obj, "dtype")) {
      g_critical ("[vivante backend] Missing necessary key 'dtype' in JSON");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }

    JsonObject *dtype_obj = json_object_get_object_member (tensor_obj, "dtype");

    // parse dtype.fmt
    vsi_attr->dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    if (!json_object_has_member (dtype_obj, "fmt")) {
      g_warning ("[vivante backend] Missing key 'fmt' in 'dtype' in JSON. Default: 'VSI_NN_DIM_FMT_NCHW'");
    } else {
      // @todo parse dtype.fmt
    }

    // parse dtype.vx_type
    if (!json_object_has_member (dtype_obj, "vx_type")) {
      g_critical ("[vivante backend] Missing necessary key 'vx_type' in 'dtype' in JSON");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    } else {
      vsi_attr->dtype.vx_type = vivante_vsi_type_from_string (json_object_get_string_member (dtype_obj, "vx_type"));
    }

    // parse dtype.qnt_type
    vsi_attr->dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    if (!json_object_has_member (dtype_obj, "qnt_type")) {
      g_warning ("[vivante backend] Missing key 'qnt_type' in 'dtype' in JSON. Default: 'VSI_NN_QNT_TYPE_NONE'");
    } else {
      vsi_attr->dtype.qnt_type = vivante_qnt_type_from_string (json_object_get_string_member (dtype_obj, "qnt_type"));
    }

    // parse dtype.fl for dynamic fixed point
    vsi_attr->dtype.fl = 0;
    if (!json_object_has_member (dtype_obj, "fl")) {
      g_warning ("[vivante backend] Missing key 'fl' in 'dtype' in JSON. Default: 0");
    } else {
      vsi_attr->dtype.fl = (int8_t) json_object_get_int_member (dtype_obj, "fl");
    }

    // parse dtype.zero_point for affine asymmetric
    vsi_attr->dtype.zero_point = 0;
    if (!json_object_has_member (dtype_obj, "zero_point")) {
      g_warning ("[vivante backend] Missing key 'zero_point' in 'dtype' in JSON. Default: 0");
    } else {
      vsi_attr->dtype.zero_point = (int32_t) json_object_get_int_member (dtype_obj, "zero_point");
    }

    // parse dtype.scale for affine asymmetric
    vsi_attr->dtype.scale = 0.0f;
    if (!json_object_has_member (dtype_obj, "scale")) {
      g_warning ("[vivante backend] Missing key 'scale' in 'dtype' in JSON. Default: 0.0f");
    } else {
      vsi_attr->dtype.scale = (float) json_object_get_double_member (dtype_obj, "scale");
    }

    /** @todo parse scales, scale_dim, channel_dim, zero_points, zero_points_dim for AFFINE_PERCHANNEL_SYMMETRIC */

    return HAL_ML_ERROR_NONE;
  };

  // 7. Set up input tensors
  for (guint i = 0; i < input_tensors_num; ++i) {
    vsi_nn_tensor_attr_t tensor_attr;

    // parse attr data from json
    JsonObject *tensor_obj = json_array_get_object_element (input_array, i);
    if (_parse_tensor_attributes_from_json (tensor_obj, &tensor_attr) != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante backend] Failed to parse tensor attributes from JSON");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }

    // Add the tensor to the graph
    vsi_nn_tensor_id_t vsi_input_id = vsi_nn_AddTensor (graph, VSI_NN_TENSOR_ID_AUTO, &tensor_attr, NULL);
    if (vsi_input_id == VSI_NN_TENSOR_ID_NA) {
      g_critical ("[vivante backend] Failed to add input tensor #%u", i);
      return HAL_ML_ERROR_RUNTIME_ERROR;
    }

    g_info ("[vivante backend] Added input tensor #%u with id %u", i, vsi_input_id);
    node->input.tensors[i] = vsi_input_id;
    graph->input.tensors[i] = vsi_input_id;
  }

  for (guint i = 0; i < input_tensors_num; ++i) {
    g_info ("[vivante backend] print Input tensor #%u:", graph->input.tensors[i]);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor (graph, graph->input.tensors[i]);
    vsi_nn_PrintTensor (tensor, graph->input.tensors[i]);
  }

  // 8. Set up output tensors
  for (guint i = 0; i < output_tensors_num; ++i) {
    vsi_nn_tensor_attr_t tensor_attr;

    // parse attr data from json
    JsonObject *tensor_obj = json_array_get_object_element (output_array, i);
    if (_parse_tensor_attributes_from_json (tensor_obj, &tensor_attr) != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante backend] Failed to parse tensor attributes from JSON");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }

    // Add the tensor to the graph
    vsi_nn_tensor_id_t vsi_output_id = vsi_nn_AddTensor (graph, VSI_NN_TENSOR_ID_AUTO, &tensor_attr, NULL);
    if (vsi_output_id == VSI_NN_TENSOR_ID_NA) {
      g_critical ("[vivante backend] Failed to add output tensor #%u", i);
      return HAL_ML_ERROR_RUNTIME_ERROR;
    }

    g_info ("[vivante backend] Added output tensor #%u with id %u", i, vsi_output_id);
    node->output.tensors[i] = vsi_output_id;
    graph->output.tensors[i] = vsi_output_id;
  }

  for (guint i = 0; i < output_tensors_num; ++i) {
    g_info ("[vivante backend] print output tensor #%u:", graph->output.tensors[i]);
    vsi_nn_tensor_t *tensor = vsi_nn_GetTensor (graph, graph->output.tensors[i]);
    vsi_nn_PrintTensor (tensor, graph->output.tensors[i]);
  }

  // 9. Setup graph
  if (vsi_nn_SetupGraph (graph, FALSE) != VSI_SUCCESS) {
    g_critical ("[vivante backend] Failed to setup graph");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  self->graph = graph;

  return HAL_ML_ERROR_NONE;
}

static int
_json_release_neural_network (vivante_handle_s *self)
{
  if (self->ctx) {
    vsi_nn_ReleaseContext (&self->ctx);
    self->ctx = NULL;
  }

  if (self->graph) {
    vsi_nn_ReleaseGraph (&self->graph);
    self->graph = NULL;
  }
}

static int
ml_vivante_deinit (void *backend_private)
{
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;

  if (!vivante) {
    g_critical ("[vivante backend] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (vivante->using_json_for_graph_setup) {
    _json_release_neural_network (vivante);
  } else {
    if (vivante->graph)
      vivante->result_vnn_ReleaseNeuralNetwork (vivante->graph);

    if (vivante->handle)
      dlclose (vivante->handle);
  }

  gst_tensors_info_free (&vivante->inputInfo);
  gst_tensors_info_free (&vivante->outputInfo);

  g_free (vivante->model_path);
  g_free (vivante->json_path);
  g_free (vivante->so_path);
  g_free (vivante);

  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_configure_instance (void *backend_private, const void *prop_)
{
  const GstTensorFilterProperties *prop = (const GstTensorFilterProperties *) prop_;
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;

  if (!vivante) {
    g_critical ("[vivante backend] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  /* Parse custom properties */
  if (prop->custom_properties) {
    gchar **options = g_strsplit (prop->custom_properties, ",", -1);

    for (guint op = 0; op < g_strv_length (options); ++op) {
      gchar **option = g_strsplit (options[op], ":", -1);

      if (g_strv_length (option) > 1) {
        g_strstrip (option[0]);
        g_strstrip (option[1]);

        if (g_ascii_strcasecmp (option[0], "json") == 0) {
          vivante->using_json_for_graph_setup = TRUE;
          vivante->json_path = g_strdup (option[1]);
          g_info ("[vivante backend] Using JSON for graph setup with JSON file: %s", option[1]);
        } else {
          g_warning ("Unknown option (%s).", options[op]);
        }
      }

      g_strfreev (option);
    }

    g_strfreev (options);
  }

  vivante->model_path = g_strdup (prop->model_files[0]);
  if (prop->num_models > 1) {
    vivante->using_json_for_graph_setup = FALSE;
  }


  /* do model loading */
  if (vivante->using_json_for_graph_setup) {
    /* .json based initializing */
    int error = _json_create_neural_network (vivante);
    if (error != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante backend] Failed to create neural network from JSON");
      _json_release_neural_network (vivante);
      return error;
    }
    g_info ("[vivante backend] Model loaded successfully from JSON!");
  } else {
    /* setup .so */
    vivante->so_path = g_strdup (prop->model_files[1]);

    vivante->handle = dlopen (vivante->so_path, RTLD_NOW);
    if (!vivante->handle) {
      g_critical ("Failed to load shared library: %s", vivante->so_path);
      return HAL_ML_ERROR_RUNTIME_ERROR;
    }

    /* get symbols */
    vivante->result_vnn_ReleaseNeuralNetwork = (void (*) (vsi_nn_graph_t *)) dlsym (
        vivante->handle, "vnn_ReleaseNeuralNetwork");
    vivante->result_vnn_CreateNeuralNetwork = (vsi_nn_graph_t * (*) (const char *) )
        dlsym (vivante->handle, "vnn_CreateNeuralNetwork");

    if (vivante->postProcess) {
      vivante->postProcessFunc = (vsi_status (*) (vsi_nn_graph_t *)) dlsym (
          vivante->handle, "vnn_PostProcessNeuralNetwork");
    }

    /* .so based initializing */
    vivante->graph = vivante->result_vnn_CreateNeuralNetwork (vivante->model_path);
  }

  /* setting input and output tensors info */
  gst_tensors_info_init (&vivante->inputInfo);
  gst_tensors_info_init (&vivante->outputInfo);

  vivante->inputInfo.num_tensors = vivante->graph->input.num;
  for (unsigned int i = 0; i < vivante->graph->input.num; i++) {
    vsi_nn_tensor_t *i_tensor
        = vsi_nn_GetTensor (vivante->graph, vivante->graph->input.tensors[i]);
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&vivante->inputInfo, i);

    info->type = convert_tensortype (i_tensor->attr.dtype.vx_type);
    info->name = g_strdup_printf ("%i", vivante->graph->input.tensors[i]);
    for (unsigned int j = 0; j < i_tensor->attr.dim_num; ++j) {
      info->dimension[j] = i_tensor->attr.size[j];
    }
  }

  vivante->outputInfo.num_tensors = vivante->graph->output.num;
  for (unsigned int i = 0; i < vivante->graph->output.num; i++) {
    vsi_nn_tensor_t *o_tensor
        = vsi_nn_GetTensor (vivante->graph, vivante->graph->output.tensors[i]);
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&vivante->outputInfo, i);

    info->type = convert_tensortype (o_tensor->attr.dtype.vx_type);
    info->name = g_strdup_printf ("%i", vivante->graph->output.tensors[i]);
    for (unsigned int j = 0; j < o_tensor->attr.dim_num; ++j) {
      info->dimension[j] = o_tensor->attr.size[j];
    }
  }

  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_invoke (void *backend_private, const void *input_, void *output_)
{
  const GstTensorMemory *input = (const GstTensorMemory *) input_;
  GstTensorMemory *output = (GstTensorMemory *) output_;
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;

  if (!vivante) {
    g_critical ("[vivante backend] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  for (unsigned int i = 0; i < vivante->graph->input.num; i++) {
    vsi_nn_tensor_t *tensor
        = vsi_nn_GetTensor (vivante->graph, vivante->graph->input.tensors[i]);
    if (vsi_nn_CopyDataToTensor (vivante->graph, tensor, (uint8_t *) input[i].data) != VSI_SUCCESS) {
      g_critical ("[vivante backend] Failed to copy data to tensor");
      return HAL_ML_ERROR_RUNTIME_ERROR;
    }
  }

  if (vsi_nn_RunGraph (vivante->graph) != VSI_SUCCESS) {
    g_critical ("[vivante backend] Failed to run graph");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  if (vivante->postProcess)
    vivante->postProcessFunc (vivante->graph);

  for (unsigned int i = 0; i < vivante->graph->output.num; i++) {
    vsi_nn_tensor_t *out_tensor
        = vsi_nn_GetTensor (vivante->graph, vivante->graph->output.tensors[i]);
    /* Do not check return value of vsi_nnCopyTensorToBuffer. It returns error in normal case */
    vsi_nn_CopyTensorToBuffer (vivante->graph, out_tensor, output[i].data);
  }

  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_get_framework_info (void *backend_private, void *fw_info)
{
  GstTensorFilterFrameworkInfo *info = (GstTensorFilterFrameworkInfo *) fw_info;

  info->name = "vivante";
  info->allow_in_place = FALSE;
  info->allocate_in_invoke = FALSE;
  info->run_without_model = FALSE;
  info->verify_model_path = FALSE;

  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_get_model_info (void *backend_private, int ops_, void *in_info_, void *out_info_)
{
  int ops = (model_info_ops) ops_;
  GstTensorsInfo *in_info = (GstTensorsInfo *) in_info_;
  GstTensorsInfo *out_info = (GstTensorsInfo *) out_info_;
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;

  if (!vivante) {
    g_critical ("[vivante backend] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  gst_tensors_info_copy (in_info, &vivante->inputInfo);
  gst_tensors_info_copy (out_info, &vivante->outputInfo);

  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_event_handler (void *backend_private, int ops_, void *data_)
{
  return HAL_ML_ERROR_NOT_SUPPORTED;
}

static int
ml_vivante_hal_backend_init (void **data)
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

static int
ml_vivante_hal_backend_exit (void *data)
{
  memset (data, 0x0, sizeof (hal_backend_ml_funcs));
  return 0;
}

hal_backend hal_backend_ml_data = {
  .name = "ml-vivante",
  .vendor = "VeriSilicon",
  .init = ml_vivante_hal_backend_init,
  .exit = ml_vivante_hal_backend_exit,
  .major_version = 1,
  .minor_version = 0,
};
