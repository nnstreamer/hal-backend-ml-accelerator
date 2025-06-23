/* SPDX-License-Identifier: Apache-2.0 */

#include <dlfcn.h>
#include <glib.h>
#include <json-glib/json-glib.h>

#include <hal-common-interface.h>
#include <hal-ml-interface.h>

#include <ovx/vsi_nn_pub.h>

#include "hal-backend-ml-util.h"


/**
 * @brief Private handle for the Vivante instance.
 */
typedef struct _vivante_handle_s {
  char *model_path; /* .nb file path */
  char *so_path; /* .so file path (for .so based model loading) */
  char *json_path; /* .json file path (for JSON based model loading) */
  gboolean use_json_for_graph;
  gboolean has_post_process; /** @deprecated Do not use it. */

  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  vsi_nn_graph_t *graph;

  /* Handles for JSON based model loading */
  vsi_nn_context_t ctx;

  /* Handles for .so based model loading */
  void *dl_handle; /* dlopened model so */
  vsi_nn_graph_t *(*model_specific_vnn_CreateNeuralNetwork) (const char *);
  void (*model_specific_vnn_ReleaseNeuralNetwork) (vsi_nn_graph_t *);
  vsi_status (*model_specific_vnn_PostProcessNeuralNetwork) (vsi_nn_graph_t *); /** @deprecated */
} vivante_handle_s;

/* ===================================================================
 * Forward Declarations of Static Helper Functions
 * ===================================================================
 */
static void _json_release_neural_network (vivante_handle_s *self);
static int _json_create_neural_network (vivante_handle_s *self);
static int _so_create_neural_network (vivante_handle_s *self);

/* ===================================================================
 * Type Conversion Helpers
 * ===================================================================
 */
/** @brief Converts VSI type string from JSON to vsi_nn_type_e enum. */
static vsi_nn_type_e
vivante_vsi_type_from_string (const gchar *vsi_type_str)
{
  if (!vsi_type_str)
    return VSI_NN_TYPE_NONE;

  /** @todo Add 4-bit types when available */
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT8") == 0)
    return VSI_NN_TYPE_INT8;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT8") == 0)
    return VSI_NN_TYPE_UINT8;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT16") == 0)
    return VSI_NN_TYPE_INT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT16") == 0)
    return VSI_NN_TYPE_UINT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT32") == 0)
    return VSI_NN_TYPE_INT32;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT32") == 0)
    return VSI_NN_TYPE_UINT32;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_INT64") == 0)
    return VSI_NN_TYPE_INT64;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_UINT64") == 0)
    return VSI_NN_TYPE_UINT64;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_FLOAT16") == 0)
    return VSI_NN_TYPE_FLOAT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_FLOAT32") == 0)
    return VSI_NN_TYPE_FLOAT32;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_FLOAT64") == 0)
    return VSI_NN_TYPE_FLOAT64;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_BFLOAT16") == 0)
    return VSI_NN_TYPE_BFLOAT16;
  if (g_ascii_strcasecmp (vsi_type_str, "VSI_NN_TYPE_BOOL8") == 0)
    return VSI_NN_TYPE_BOOL8;

  g_warning ("[vivante] Unknown VSI tensor type string from JSON: %s", vsi_type_str);
  return VSI_NN_TYPE_NONE;
}

/** @brief Converts VSI quantization type string from JSON to vsi_nn_qnt_type_e enum. */
static vsi_nn_qnt_type_e
vivante_qnt_type_from_string (const gchar *qnt_str)
{
  if (!qnt_str)
    return VSI_NN_QNT_TYPE_NONE;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_NONE") == 0)
    return VSI_NN_QNT_TYPE_NONE;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_DFP") == 0)
    return VSI_NN_QNT_TYPE_DFP;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC") == 0)
    return VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC") == 0)
    return VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC;
  if (g_ascii_strcasecmp (qnt_str, "VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC") == 0)
    return VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC;

  /** @todo Add VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_ASYMMETRIC when available */
  g_warning ("[vivante] Unknown VSI quantization type string from JSON: %s", qnt_str);
  return VSI_NN_QNT_TYPE_NONE;
}

/** @brief Converts vsi_nn_type_e to the framework's tensor_type. */
static tensor_type
convert_to_tensor_type (vsi_nn_type_e vsi_type)
{
  switch (vsi_type) {
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
      return _NNS_FLOAT16;
    case VSI_NN_TYPE_FLOAT32:
      return _NNS_FLOAT32;
    case VSI_NN_TYPE_FLOAT64:
      return _NNS_FLOAT64;
    default:
      g_warning ("[vivante] Unsupported vsi_nn_type_e: %d", vsi_type);
      return _NNS_END;
  }
}

/* ===================================================================
 * JSON Parsing and Graph Creation Helpers
 * ===================================================================
 */
/** @brief Parses tensor attributes from a JSON object into a vsi_nn_tensor_attr_t struct. */
static int
_helper_parse_tensor_attributes (JsonObject *tensor_obj, vsi_nn_tensor_attr_t *vsi_attr)
{
  memset (vsi_attr, 0, sizeof (vsi_nn_tensor_attr_t));
  vsi_attr->vtl = FALSE;
  vsi_attr->is_const = FALSE;

  // Size and dim_num
  JsonArray *size_array = json_object_get_array_member (tensor_obj, "size");
  if (!size_array) {
    g_critical ("[vivante] Tensor in JSON missing 'size' array.");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }
  vsi_attr->dim_num = json_array_get_length (size_array);
  if (vsi_attr->dim_num == 0 || vsi_attr->dim_num > VSI_NN_MAX_DIM_NUM) {
    g_critical ("[vivante] Invalid tensor 'dim_num': %u", vsi_attr->dim_num);
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }
  for (guint i = 0; i < vsi_attr->dim_num; ++i) {
    vsi_attr->size[i] = json_array_get_int_element (size_array, i);
  }

  // Dtype object
  JsonObject *dtype_obj = json_object_get_object_member (tensor_obj, "dtype");
  if (!dtype_obj) {
    g_critical ("[vivante] Tensor in JSON missing 'dtype' object.");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  // Required: vx_type
  const gchar *vx_type_str = json_object_get_string_member (dtype_obj, "vx_type");
  if (!vx_type_str) {
    g_critical ("[vivante] 'dtype' missing required 'vx_type' key.");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }
  vsi_attr->dtype.vx_type = vivante_vsi_type_from_string (vx_type_str);

  // Required: qnt_type
  vsi_attr->dtype.qnt_type
      = vivante_qnt_type_from_string (json_object_get_string_member_with_default (
          dtype_obj, "qnt_type", "VSI_NN_QNT_TYPE_NONE"));

  // Optional fields with defaults
  if (json_object_has_member (dtype_obj, "fl")) {
    vsi_attr->dtype.fl = json_object_get_int_member_with_default (dtype_obj, "fl", 0);
  }

  if (json_object_has_member (dtype_obj, "zero_point")) {
    vsi_attr->dtype.zero_point
        = json_object_get_int_member_with_default (dtype_obj, "zero_point", 0);
  }

  if (json_object_has_member (dtype_obj, "scale")) {
    vsi_attr->dtype.scale
        = json_object_get_double_member_with_default (dtype_obj, "scale", 0.0f);
  }

  /** @todo Parse scales, scale_dim, etc. for PERCHANNEL quantization */

  return HAL_ML_ERROR_NONE;
}

/** @brief Creates and sets up the neural network graph using a JSON definition file. */
static int
_json_create_neural_network (vivante_handle_s *self)
{
  g_autofree gchar *json_string = NULL;
  g_autoptr (GError) err = NULL;
  g_autoptr (JsonParser) parser = NULL;
  JsonNode *root_node = NULL;
  JsonObject *root_obj = NULL;
  JsonArray *input_array = NULL, *output_array = NULL;
  guint input_tensors_num = 0, output_tensors_num = 0;
  vsi_nn_node_t *node = NULL;

  self->ctx = vsi_nn_CreateContext ();
  if (!self->ctx) {
    g_critical ("[vivante] Failed to create VSI context.");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  if (!g_file_get_contents (self->json_path, &json_string, NULL, &err)) {
    g_critical ("[vivante] Failed to read JSON file '%s': %s", self->json_path,
        err ? err->message : "Unknown error");
    return HAL_ML_ERROR_IO_ERROR;
  }

  parser = json_parser_new ();
  if (!json_parser_load_from_data (parser, json_string, -1, &err)) {
    g_critical ("[vivante] Failed to parse JSON: %s", err ? err->message : "Unknown error");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  root_node = json_parser_get_root (parser);
  if (!root_node || !JSON_NODE_HOLDS_OBJECT (root_node)) {
    g_critical ("[vivante] JSON root is not a valid object.");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }
  root_obj = json_node_get_object (root_node);

  input_array = json_object_get_array_member (root_obj, "input_tensors");
  output_array = json_object_get_array_member (root_obj, "output_tensors");
  if (!input_array || !output_array) {
    g_critical ("[vivante] JSON must contain 'input_tensors' and 'output_tensors' arrays.");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  input_tensors_num = json_array_get_length (input_array);
  output_tensors_num = json_array_get_length (output_array);

  const guint node_num = 1U; /* single NBG node */
  const guint const_tensors_num = 0U; /** @todo support this */
  const guint normal_tensors_num = input_tensors_num + output_tensors_num;
  const guint virtual_tensors_num = output_tensors_num;

  self->graph = vsi_nn_CreateGraph (self->ctx,
      normal_tensors_num + virtual_tensors_num + const_tensors_num, node_num);
  if (!self->graph) {
    g_critical ("[vivante] Failed to create VSI graph.");
    goto cleanup;
  }

  if (!vsi_nn_SetGraphInputs (self->graph, NULL, input_tensors_num)
      || !vsi_nn_SetGraphOutputs (self->graph, NULL, output_tensors_num)) {
    g_critical ("[vivante] Failed to set graph inputs/outputs.");
    goto cleanup;
  }

  node = vsi_nn_AddNode (
      self->graph, VSI_NN_OP_NBG, input_tensors_num, output_tensors_num, NULL);
  if (!node) {
    g_critical ("[vivante] Failed to add NBG node to graph.");
    goto cleanup;
  }
  node->uid = 0;
  node->nn_param.nbg.type = VSI_NN_NBG_FILE;
  node->nn_param.nbg.url = self->model_path;

  // Set up input tensors
  for (guint i = 0; i < input_tensors_num; ++i) {
    vsi_nn_tensor_attr_t tensor_attr;

    // parse attr data from json
    JsonObject *tensor_obj = json_array_get_object_element (input_array, i);
    if (_helper_parse_tensor_attributes (tensor_obj, &tensor_attr) != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante] Failed to parse tensor attributes from JSON");
      goto cleanup;
    }

    // Add the tensor to the graph
    vsi_nn_tensor_id_t vsi_input_id
        = vsi_nn_AddTensor (self->graph, VSI_NN_TENSOR_ID_AUTO, &tensor_attr, NULL);
    if (vsi_input_id == VSI_NN_TENSOR_ID_NA) {
      g_critical ("[vivante] Failed to add input tensor #%u", i);
      goto cleanup;
    }

    g_info ("[vivante] Added input tensor #%u with id %u", i, vsi_input_id);
    node->input.tensors[i] = vsi_input_id;
    self->graph->input.tensors[i] = vsi_input_id;
  }

  for (guint i = 0; i < input_tensors_num; ++i) {
    g_info ("[vivante] print Input tensor #%u:", self->graph->input.tensors[i]);
    vsi_nn_tensor_t *tensor
        = vsi_nn_GetTensor (self->graph, self->graph->input.tensors[i]);
    vsi_nn_PrintTensor (tensor, self->graph->input.tensors[i]);
  }

  // Set up output tensors
  for (guint i = 0; i < output_tensors_num; ++i) {
    vsi_nn_tensor_attr_t tensor_attr;

    // parse attr data from json
    JsonObject *tensor_obj = json_array_get_object_element (output_array, i);
    if (_helper_parse_tensor_attributes (tensor_obj, &tensor_attr) != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante] Failed to parse tensor attributes from JSON");
      goto cleanup;
    }

    // Add the tensor to the graph
    vsi_nn_tensor_id_t vsi_output_id
        = vsi_nn_AddTensor (self->graph, VSI_NN_TENSOR_ID_AUTO, &tensor_attr, NULL);
    if (vsi_output_id == VSI_NN_TENSOR_ID_NA) {
      g_critical ("[vivante] Failed to add output tensor #%u", i);
      goto cleanup;
    }

    g_info ("[vivante] Added output tensor #%u with id %u", i, vsi_output_id);
    node->output.tensors[i] = vsi_output_id;
    self->graph->output.tensors[i] = vsi_output_id;
  }

  for (guint i = 0; i < output_tensors_num; ++i) {
    g_info ("[vivante] print output tensor #%u:", self->graph->output.tensors[i]);
    vsi_nn_tensor_t *tensor
        = vsi_nn_GetTensor (self->graph, self->graph->output.tensors[i]);
    vsi_nn_PrintTensor (tensor, self->graph->output.tensors[i]);
  }

  // setup graph
  if (vsi_nn_SetupGraph (self->graph, FALSE) != VSI_SUCCESS) {
    g_critical ("[vivante] Failed to setup VSI graph.");
    goto cleanup;
  }

  return HAL_ML_ERROR_NONE;

cleanup:
  _json_release_neural_network (self);
  return HAL_ML_ERROR_RUNTIME_ERROR;
}

/** @brief Releases all resources associated with a JSON-based graph. */
static void
_json_release_neural_network (vivante_handle_s *self)
{
  if (self->graph) {
    vsi_nn_ReleaseGraph (&self->graph);
    self->graph = NULL;
  }
  if (self->ctx) {
    vsi_nn_ReleaseContext (&self->ctx);
    self->ctx = NULL;
  }
}

/* ===================================================================
 * Shared Library (.so) Loading Helpers
 * ===================================================================
 */

/** @brief Creates the neural network from a pre-compiled shared library. */
static int
_so_create_neural_network (vivante_handle_s *self)
{
  self->dl_handle = dlopen (self->so_path, RTLD_NOW);
  if (!self->dl_handle) {
    g_critical ("[vivante] Failed to load shared library '%s': %s",
        self->so_path, dlerror ());
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  self->model_specific_vnn_CreateNeuralNetwork = (vsi_nn_graph_t * (*) (const char *) )
      dlsym (self->dl_handle, "vnn_CreateNeuralNetwork");
  self->model_specific_vnn_ReleaseNeuralNetwork
      = (void (*) (vsi_nn_graph_t *)) dlsym (self->dl_handle, "vnn_ReleaseNeuralNetwork");

  if (!self->model_specific_vnn_CreateNeuralNetwork
      || !self->model_specific_vnn_ReleaseNeuralNetwork) {
    g_critical ("[vivante] Could not find required symbols in '%s'", self->so_path);
    dlclose (self->dl_handle);
    self->dl_handle = NULL;
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  if (self->has_post_process) {
    self->model_specific_vnn_PostProcessNeuralNetwork = (vsi_status (*) (
        vsi_nn_graph_t *)) dlsym (self->dl_handle, "vnn_PostProcessNeuralNetwork");
    if (!self->model_specific_vnn_PostProcessNeuralNetwork) {
      g_warning ("[vivante] 'postProcess' was requested, but symbol 'vnn_PostProcessNeuralNetwork' not found.");
      self->has_post_process = FALSE;
    }
  }

  self->graph = self->model_specific_vnn_CreateNeuralNetwork (self->model_path);
  if (!self->graph) {
    g_critical ("[vivante] vnn_CreateNeuralNetwork failed for model '%s'", self->model_path);
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  return HAL_ML_ERROR_NONE;
}

/* ===================================================================
 * Main HAL Implementation Functions
 * ===================================================================
 */
static int
ml_vivante_init (void **backend_private)
{
  vivante_handle_s *vivante = g_new0 (vivante_handle_s, 1);

  gst_tensors_info_init (&vivante->inputInfo);
  gst_tensors_info_init (&vivante->outputInfo);

  vivante->use_json_for_graph = TRUE;
  vivante->has_post_process = FALSE;

  *backend_private = vivante;
  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_deinit (void *backend_private)
{
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;

  if (!vivante) {
    g_critical ("[vivante] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (vivante->use_json_for_graph) {
    _json_release_neural_network (vivante);
  } else {
    if (vivante->graph && vivante->model_specific_vnn_ReleaseNeuralNetwork) {
      vivante->model_specific_vnn_ReleaseNeuralNetwork (vivante->graph);
    }
    if (vivante->dl_handle) {
      dlclose (vivante->dl_handle);
    }
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

  if (!vivante || !prop) {
    g_critical ("[vivante] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  vivante->model_path = g_strdup (prop->model_files[0]);
  // Default loading strategy: if more than one model file is given, assume .so loading.
  if (prop->num_models > 1) {
    vivante->use_json_for_graph = FALSE;
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
          vivante->use_json_for_graph = TRUE;
          vivante->json_path = g_strdup (option[1]);
          g_info ("[vivante] Using JSON for graph setup: %s", vivante->json_path);
        } else {
          g_warning ("Unknown option (%s).", options[op]);
        }
      }

      g_strfreev (option);
    }

    g_strfreev (options);
  }

  /* Load model based on the determined strategy JSON vs so */
  if (vivante->use_json_for_graph) {
    if (!vivante->json_path || !g_file_test (vivante->json_path, G_FILE_TEST_IS_REGULAR)) {
      g_critical ("[vivante] JSON loading was selected, but no JSON path was provided via 'json:' custom property.");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }

    int status = _json_create_neural_network (vivante);
    if (status != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante] Failed to create VSI graph.");
      return status;
    }
  } else {
    if (prop->num_models <= 1) {
      g_critical ("[vivante] .so loading requires a second model file (the .so path).");
      return HAL_ML_ERROR_INVALID_PARAMETER;
    }
    vivante->so_path = g_strdup (prop->model_files[1]);

    int status = _so_create_neural_network (vivante);
    if (status != HAL_ML_ERROR_NONE) {
      g_critical ("[vivante] Failed to create VSI graph.");
      return status;
    }
  }

  /* setting input and output tensors info */
  vivante->inputInfo.num_tensors = vivante->graph->input.num;
  for (unsigned int i = 0; i < vivante->graph->input.num; i++) {
    vsi_nn_tensor_t *i_tensor
        = vsi_nn_GetTensor (vivante->graph, vivante->graph->input.tensors[i]);
    GstTensorInfo *info = gst_tensors_info_get_nth_info (&vivante->inputInfo, i);

    info->type = convert_to_tensor_type (i_tensor->attr.dtype.vx_type);
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

    info->type = convert_to_tensor_type (o_tensor->attr.dtype.vx_type);
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
    g_critical ("[vivante] invalid backend_private");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  for (unsigned int i = 0; i < vivante->graph->input.num; i++) {
    vsi_nn_tensor_t *tensor
        = vsi_nn_GetTensor (vivante->graph, vivante->graph->input.tensors[i]);
    if (vsi_nn_CopyDataToTensor (vivante->graph, tensor, (uint8_t *) input[i].data) != VSI_SUCCESS) {
      g_critical ("[vivante] Failed to copy data to tensor");
      return HAL_ML_ERROR_RUNTIME_ERROR;
    }
  }

  if (vsi_nn_RunGraph (vivante->graph) != VSI_SUCCESS) {
    g_critical ("[vivante] Failed to run graph");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  if (vivante->has_post_process)
    vivante->model_specific_vnn_PostProcessNeuralNetwork (vivante->graph);

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
ml_vivante_get_model_info (void *backend_private, int ops, void *in_info, void *out_info)
{
  vivante_handle_s *vivante = (vivante_handle_s *) backend_private;
  if (!vivante)
    return HAL_ML_ERROR_INVALID_PARAMETER;

  gst_tensors_info_copy ((GstTensorsInfo *) in_info, &vivante->inputInfo);
  gst_tensors_info_copy ((GstTensorsInfo *) out_info, &vivante->outputInfo);

  return HAL_ML_ERROR_NONE;
}

static int
ml_vivante_event_handler (void *backend_private, int ops, void *data)
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
