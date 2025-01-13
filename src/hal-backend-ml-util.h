/* SPDX-License-Identifier: Apache-2.0 */

#include <glib.h>

#include "tensor_typedef.h"
#include "nnstreamer_plugin_api_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

void gst_tensor_info_init (GstTensorInfo * info);
void gst_tensor_info_free (GstTensorInfo * info);
void gst_tensors_info_init (GstTensorsInfo * info);
void gst_tensors_info_free (GstTensorsInfo * info);
gsize gst_tensor_get_element_size (tensor_type type);
gulong gst_tensor_get_element_count (const tensor_dim dim);
gsize gst_tensor_info_get_size (const GstTensorInfo * info);
GstTensorInfo * gst_tensors_info_get_nth_info (GstTensorsInfo * info, guint index);
void gst_tensor_info_copy_n (GstTensorInfo * dest, const GstTensorInfo * src, const guint n);
void gst_tensor_info_copy (GstTensorInfo * dest, const GstTensorInfo * src);
void gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src);

#ifdef __cplusplus
}
#endif
