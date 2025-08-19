/* SPDX-License-Identifier: Apache-2.0 */

#include <glib.h>

#include "hal-backend-ml-util.h"

void gst_tensor_info_init (GstTensorInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  info->name = NULL;
  info->type = _NNS_END;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    info->dimension[i] = 0;
  }
}

void gst_tensor_info_free (GstTensorInfo * info)
{
  g_return_if_fail (info != NULL);

  g_free (info->name);

  /* Init default */
  gst_tensor_info_init (info);
}

void gst_tensors_info_init (GstTensorsInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  info->num_tensors = 0;
  info->extra = NULL;

  /** @note default format is */
  info->format = _NNS_TENSOR_FORMAT_STATIC;

  for (i = 0; i < NNS_TENSOR_MEMORY_MAX; i++) {
    gst_tensor_info_init (&info->info[i]);
  }
}

void gst_tensors_info_free (GstTensorsInfo * info)
{
  guint i;

  g_return_if_fail (info != NULL);

  for (i = 0; i < NNS_TENSOR_MEMORY_MAX; i++) {
    gst_tensor_info_free (&info->info[i]);
  }

  if (info->extra) {
    for (i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; ++i)
      gst_tensor_info_free (&info->extra[i]);

    g_free (info->extra);
    info->extra = NULL;
  }

  /* Init default */
  gst_tensors_info_init (info);
}

static const guint tensor_element_size[] = {
  [_NNS_INT32] = 4,
  [_NNS_UINT32] = 4,
  [_NNS_INT16] = 2,
  [_NNS_UINT16] = 2,
  [_NNS_INT8] = 1,
  [_NNS_UINT8] = 1,
  [_NNS_FLOAT64] = 8,
  [_NNS_FLOAT32] = 4,
  [_NNS_INT64] = 8,
  [_NNS_UINT64] = 8,
  [_NNS_FLOAT16] = 2,
  [_NNS_END] = 0,
};

gsize gst_tensor_get_element_size (tensor_type type)
{
  g_return_val_if_fail (type >= 0 && type <= _NNS_END, 0);

  return tensor_element_size[type];
}

gulong gst_tensor_get_element_count (const tensor_dim dim)
{
  gulong count = 1;
  guint i;

  for (i = 0; i < NNS_TENSOR_RANK_LIMIT; i++) {
    if (dim[i] == 0)
      break;

    count *= dim[i];
  }

  return (i > 0) ? count : 0;
}

gsize gst_tensor_info_get_size (const GstTensorInfo * info)
{
  gsize data_size;

  g_return_val_if_fail (info != NULL, 0);

  data_size = gst_tensor_get_element_count (info->dimension) *
      gst_tensor_get_element_size (info->type);

  return data_size;
}

GstTensorInfo * gst_tensors_info_get_nth_info (GstTensorsInfo * info, guint index)
{
  guint i;

  g_return_val_if_fail (info != NULL, NULL);

  if (index < NNS_TENSOR_MEMORY_MAX)
    return &info->info[index];

  if (index < NNS_TENSOR_SIZE_LIMIT) {
    if (!info->extra) {
      info->extra = g_new0 (GstTensorInfo, NNS_TENSOR_SIZE_EXTRA_LIMIT);

      for (i = 0; i < NNS_TENSOR_SIZE_EXTRA_LIMIT; ++i)
        gst_tensor_info_init (&info->extra[i]);
    }

    return &info->extra[index - NNS_TENSOR_MEMORY_MAX];
  }

  g_critical ("Failed to get the information, invalid index %u (max %d).",
      index, NNS_TENSOR_SIZE_LIMIT);
  return NULL;
}

void gst_tensor_info_copy_n (GstTensorInfo * dest, const GstTensorInfo * src,
    const guint n)
{
  guint i;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  dest->name = g_strdup (src->name);
  dest->type = src->type;

  for (i = 0; i < n; i++) {
    dest->dimension[i] = src->dimension[i];
  }
}

void gst_tensor_info_copy (GstTensorInfo * dest, const GstTensorInfo * src)
{
  gst_tensor_info_copy_n (dest, src, NNS_TENSOR_RANK_LIMIT);
}

void gst_tensors_info_copy (GstTensorsInfo * dest, const GstTensorsInfo * src)
{
  guint i, num;
  GstTensorInfo *_dest, *_src;

  g_return_if_fail (dest != NULL);
  g_return_if_fail (src != NULL);

  gst_tensors_info_init (dest);
  num = dest->num_tensors = src->num_tensors;
  dest->format = src->format;

  /* Try to copy tensor info even if its format is not static. */
  for (i = 0; i < num; i++) {
    _dest = gst_tensors_info_get_nth_info (dest, i);
    _src = gst_tensors_info_get_nth_info ((GstTensorsInfo *) src, i);

    gst_tensor_info_copy (_dest, _src);
  }
}
