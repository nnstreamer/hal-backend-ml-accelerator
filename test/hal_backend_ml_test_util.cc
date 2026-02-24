#include "hal_backend_ml_test_util.h"
#include <json-glib/json-glib.h>
#include <stdio.h>
#include <iostream>
#include "hal-backend-ml-util.h"
#include <hal-ml-interface.h>

/* Global test configuration - accessed through getter/setter functions */
static TestGstTensorFilterProperties* g_test_config = nullptr;

TestGstTensorFilterProperties*
get_test_config()
{
  return g_test_config;
}

void
set_test_config(TestGstTensorFilterProperties *config)
{
  g_test_config = config;
}

int
parse_json_file (char * json_path, TestGstTensorFilterProperties *prop)
{
  JsonParser *parser = NULL;
  GError *error = NULL;
  JsonNode *rootNode = NULL;
  JsonObject *rootObject = NULL;
  JsonObject *metadata = NULL;
  JsonArray *configParametersArray = NULL;
  int ret = HAL_ML_ERROR_NONE;

  /* Input validation */
  if (!json_path) {
    g_warning ("JSON file path is NULL");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  if (!prop) {
    g_warning ("TestGstTensorFilterProperties pointer is NULL");
    return HAL_ML_ERROR_INVALID_PARAMETER;
  }

  /* Free existing allocations */
  if (prop->base.model_files) {
    for (int i = 0; i < prop->base.num_models; ++i) {
      g_free ((char *) prop->base.model_files[i]);
    }
    delete[] prop->base.model_files;
    prop->base.model_files = NULL;
  }

  if (prop->input_data_files) {
    for (guint i = 0; i < prop->num_input_files; ++i) {
      g_free ((char *) prop->input_data_files[i]);
    }
    delete[] prop->input_data_files;
    prop->input_data_files = NULL;
  }

  g_free ((char *) prop->base.fwname);
  prop->base.fwname = NULL;

  g_free ((char *) prop->base.custom_properties);
  prop->base.custom_properties = NULL;

  /* Declare variables before gotos to avoid crossing initialization */
  guint elements;
  guint elem;
  JsonNode *configNode;
  JsonObject *configObject;
  guint num_models;
  guint i;

  /* Initialize parser */
  parser = json_parser_new ();
  if (!parser) {
    g_warning ("Failed to create JSON parser");
    return HAL_ML_ERROR_RUNTIME_ERROR;
  }

  /* Load JSON file */
  if (!json_parser_load_from_file (parser, json_path, &error)) {
    g_warning ("Failed to load JSON file: %s", error->message);
    ret = HAL_ML_ERROR_IO_ERROR;
    goto cleanup;
  }

  /* Get root node */
  rootNode = json_parser_get_root (parser);
  if (!rootNode || JSON_NODE_TYPE (rootNode) != JSON_NODE_OBJECT) {
    g_warning ("JSON root is not an object");
    ret = HAL_ML_ERROR_INVALID_PARAMETER;
    goto cleanup;
  }

  /* Get root object */
  rootObject = json_node_get_object (rootNode);
  if (!rootObject) {
    g_warning ("Failed to get JSON object from root node");
    ret = HAL_ML_ERROR_INVALID_PARAMETER;
    goto cleanup;
  }

  /* Get metadata object */
  metadata = json_object_get_object_member (rootObject, "metadata");
  if (!metadata) {
    g_warning ("'metadata' object not found in JSON");
    ret = HAL_ML_ERROR_INVALID_PARAMETER;
    goto cleanup;
  }

  /* Get configParameters array */
  configParametersArray = json_object_get_array_member (metadata, "configParameters");
  if (!configParametersArray) {
    g_warning ("'configParameters' array not found in 'metadata'");
    ret = HAL_ML_ERROR_INVALID_PARAMETER;
    goto cleanup;
  }

  /* Process each config parameter element */
  elements = json_array_get_length (configParametersArray);
  for (elem = 0; elem < elements; ++elem) {
    configNode = json_array_get_element (configParametersArray, elem);
    configObject = NULL;

    if (!configNode || JSON_NODE_TYPE (configNode) != JSON_NODE_OBJECT) {
      g_warning ("Invalid element at index %u in 'configParameters'", elem);
      continue;
    }

    configObject = json_node_get_object (configNode);
    if (!configObject) {
      g_warning ("Failed to get object from element at index %u", elem);
      continue;
    }

    /* Parse 'fwname' */
    if (json_object_has_member (configObject, "fwname")) {
      const gchar *fwname = json_object_get_string_member (configObject, "fwname");
      if (fwname) {
        g_free ((char *) prop->base.fwname);
        prop->base.fwname = g_strdup (fwname);
        g_info ("fwname: %s", prop->base.fwname);
      } else {
        g_warning ("'fwname' is not a string");
      }
    }

    /* Parse 'fw_opened' */
    prop->base.fw_opened = json_object_get_int_member_with_default (configObject, "fw_opened", 0);

    /* Parse 'num_models' (will be updated based on model_files array) */
    prop->base.num_models = json_object_get_int_member_with_default (configObject, "num_models", 0);
    g_info ("num_models: %d", prop->base.num_models);

    /* Parse 'model_files' array */
    JsonArray *model_files_array = json_object_get_array_member (configObject, "model_files");
    if (model_files_array) {
      num_models = json_array_get_length (model_files_array);

      if (num_models > 0) {
        prop->base.num_models = num_models;
        prop->base.model_files = new const char*[num_models];

        for (i = 0; i < num_models; ++i) {
          const gchar *file_name = json_array_get_string_element (model_files_array, i);
          if (file_name) {
            prop->base.model_files[i] = g_strdup (file_name);
            g_info ("Model file %u: %s", i, prop->base.model_files[i]);
          } else {
            g_warning ("Failed to get model file name at index %u", i);
            prop->base.model_files[i] = NULL;
          }
        }
      }
    } else {
      g_warning ("'model_files' array not found in configParameters");
    }

    /* Parse 'input_configured' */
    prop->base.input_configured = json_object_get_int_member_with_default (configObject, "input_configured", 0);

    /* Parse 'output_configured' */
    prop->base.output_configured = json_object_get_int_member_with_default (configObject, "output_configured", 0);

    /* Parse 'custom_properties' */
    if (json_object_has_member (configObject, "custom_properties")) {
      const gchar *custom_properties = json_object_get_string_member (configObject, "custom_properties");
      if (custom_properties) {
        g_free ((char *) prop->base.custom_properties);
        prop->base.custom_properties = g_strdup (custom_properties);
        g_info ("Custom properties: %s", prop->base.custom_properties);
      }
    }

    /* Parse 'input_file' array */
    JsonArray *input_file_array = json_object_get_array_member (configObject, "input_file");
    if (input_file_array) {
      guint num_input_files = json_array_get_length (input_file_array);

      if (num_input_files > 0) {
        prop->input_data_files = new const char*[num_input_files];
        prop->num_input_files = num_input_files;

        for (i = 0; i < num_input_files; ++i) {
          const gchar *file_path = json_array_get_string_element (input_file_array, i);
          if (file_path) {
            prop->input_data_files[i] = g_strdup (file_path);
            g_info ("Input data file %u: %s", i, prop->input_data_files[i]);
          } else {
            g_warning ("Failed to get input file path at index %u", i);
            prop->input_data_files[i] = NULL;
          }
        }
      }
    } else {
      prop->num_input_files = 0;
      prop->input_data_files = nullptr;
    }
  }

cleanup:
  g_clear_error (&error);
  g_clear_object (&parser);

  return ret;
}
