#include "gtest/gtest.h"
#include <stdio.h>
#include "hal_backend_ml_test_util.h"
#include "hal-backend-ml-util.h"
#include <hal-ml-interface.h>
#include <iostream>

/**
 * @brief Display usage information for the test executable
 *
 * Shows how to run the test executable with the required configuration file.
 *
 * @param program_name Name of the test executable
 */
static void
show_usage (const char *program_name)
{
  std::cerr << "\n";
  std::cerr << "=====================================================================\n";
  std::cerr << "  HAL Backend ML Accelerator Test Runner\n";
  std::cerr << "=====================================================================\n\n";
  std::cerr << "Usage: " << program_name << " <config.json>\n\n";
  std::cerr << "Required Arguments:\n";
  std::cerr << "  config.json    Path to JSON configuration file for test setup\n\n";
  std::cerr << "JSON Configuration File Format:\n";
  std::cerr << "  The JSON file must contain the following structure:\n";
  std::cerr << "  {\n";
  std::cerr << "    \"metadata\": {\n";
  std::cerr << "      \"configParameters\": [\n";
  std::cerr << "        {\n";
  std::cerr << "          \"fwname\": \"snpe|vivante|dummy-passthrough\",\n";
  std::cerr << "          \"model_files\": [\"/path/to/model.dlc\"],\n";
  std::cerr << "          \"input_file\": [\"/path/to/input1.bin\", ...],\n";
  std::cerr << "          \"custom_properties\": \"key=value;...\"\n";
  std::cerr << "        }\n";
  std::cerr << "      ]\n";
  std::cerr << "    }\n";
  std::cerr << "  }\n\n";
  std::cerr << "Example:\n";
  std::cerr << "  " << program_name << " /path/to/test_config.json\n\n";
  std::cerr << "=====================================================================\n\n";
}

int
main (int argc, char **argv)
{
  int ret = 0;
  TestGstTensorFilterProperties* prop = nullptr;

  /* Check for --help flag */
  if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    show_usage(argv[0]);
    return 0;
  }

  /* Validate command line arguments */
  if (argc < 2) {
    std::cerr << "Error: Missing JSON configuration file path\n";
    show_usage(argv[0]);
    return 1;
  }

  char *json_path;  
  json_path = argv[1];

  /* Allocate and initialize test configuration */
  prop = new TestGstTensorFilterProperties();
  memset(prop, 0, sizeof(TestGstTensorFilterProperties));

  /* Parse JSON configuration file */
  ret = parse_json_file(json_path, prop);
  if (ret != HAL_ML_ERROR_NONE) {
    std::cerr << "Error: Failed to parse JSON file (error code: " << ret << ")" << std::endl;
    g_free ((char *) prop->base.fwname);
    g_free ((char *) prop->base.custom_properties);
    delete prop;
    return 1;
  }

  /* Set the global test configuration */
  set_test_config(prop);

  /* Run all tests */
  testing::InitGoogleTest (&argc, argv);
  ret = RUN_ALL_TESTS ();

  /* Cleanup allocated memory */
  if (prop->base.model_files) {
    for (int i = 0; i < prop->base.num_models; ++i) {
      g_free ((char *) prop->base.model_files[i]);
    }
    delete[] prop->base.model_files;
  }
  
  if (prop->input_data_files) {
    for (guint i = 0; i < prop->num_input_files; ++i) {
      g_free ((char *) prop->input_data_files[i]);
    }
    delete[] prop->input_data_files;
  }
  
  g_free ((char *) prop->base.custom_properties);
  g_free ((char *) prop->base.fwname);
  delete prop;

  return ret;
}
