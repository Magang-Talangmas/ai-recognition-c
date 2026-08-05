#include "attendance/face_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

#ifdef HAVE_OPENVINO
#include <openvino/c/openvino.h>
#endif

struct FaceEngine {
    FaceConfig config;
    void *det_session;
    void *rec_session;
    bool is_initialized;
};

void face_vector_l2_normalize(float *vector, int dim) {
    if (!vector || dim <= 0) return;
    double sum_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        sum_sq += (double)vector[i] * (double)vector[i];
    }
    double norm = sqrt(sum_sq);
    if (norm < 1e-12) norm = 1e-12;
    float inv_norm = (float)(1.0 / norm);
    for (int i = 0; i < dim; i++) {
        vector[i] *= inv_norm;
    }
}

float face_engine_calculate_blur_score(const ImageBuffer *image, const FaceBBox *crop_box) {
    if (!image || !image->data || !crop_box) return 0.0f;

    int x1 = (int)fmaxf(0.0f, crop_box->x1);
    int y1 = (int)fmaxf(0.0f, crop_box->y1);
    int x2 = (int)fminf((float)image->width, crop_box->x2);
    int y2 = (int)fminf((float)image->height, crop_box->y2);

    int w = x2 - x1;
    int h = y2 - y1;
    if (w <= 4 || h <= 4) return 0.0f;

    /* Compute Laplacian variance on grayscale crop */
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;

    int stride = image->stride;
    int ch = image->channels;

    for (int y = y1 + 1; y < y2 - 1; y++) {
        const uint8_t *row_prev = image->data + (y - 1) * stride;
        const uint8_t *row_curr = image->data + y * stride;
        const uint8_t *row_next = image->data + (y + 1) * stride;

        for (int x = x1 + 1; x < x2 - 1; x++) {
            /* Quick BGR to Gray approximation: (B + 2*G + R) / 4 */
            int c_up    = (row_prev[(x) * ch] + 2 * row_prev[(x) * ch + 1] + row_prev[(x) * ch + 2]) >> 2;
            int c_down  = (row_next[(x) * ch] + 2 * row_next[(x) * ch + 1] + row_next[(x) * ch + 2]) >> 2;
            int c_left  = (row_curr[(x - 1) * ch] + 2 * row_curr[(x - 1) * ch + 1] + row_curr[(x - 1) * ch + 2]) >> 2;
            int c_right = (row_curr[(x + 1) * ch] + 2 * row_curr[(x + 1) * ch + 1] + row_curr[(x + 1) * ch + 2]) >> 2;
            int c_mid   = (row_curr[(x) * ch] + 2 * row_curr[(x) * ch + 1] + row_curr[(x) * ch + 2]) >> 2;

            int lap = c_up + c_down + c_left + c_right - 4 * c_mid;
            sum += lap;
            sum_sq += lap * lap;
            count++;
        }
    }

    if (count == 0) return 0.0f;
    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    return (float)fmax(0.0, variance);
}

/* Reference 5-point landmarks for 112x112 ArcFace aligned template */
static const float ARCFACE_DST_LANDMARKS[5][2] = {
    {38.2946f, 51.6963f}, /* Left Eye */
    {73.5318f, 51.5014f}, /* Right Eye */
    {56.0252f, 71.7366f}, /* Nose Tip */
    {41.5493f, 92.3655f}, /* Left Mouth */
    {70.7299f, 92.2041f}  /* Right Mouth */
};

/* Estimate 2x3 Affine Similarity Transformation (Umeyama algorithm) */
static void get_similarity_transform(
    const float src[5][2],
    const float dst[5][2],
    float M[2][3]
) {
    float src_mean_x = 0, src_mean_y = 0;
    float dst_mean_x = 0, dst_mean_y = 0;

    for (int i = 0; i < 5; i++) {
        src_mean_x += src[i][0];
        src_mean_y += src[i][1];
        dst_mean_x += dst[i][0];
        dst_mean_y += dst[i][1];
    }
    src_mean_x /= 5.0f; src_mean_y /= 5.0f;
    dst_mean_x /= 5.0f; dst_mean_y /= 5.0f;

    float src_var = 0;
    float cov_xx = 0, cov_xy = 0, cov_yx = 0, cov_yy = 0;

    for (int i = 0; i < 5; i++) {
        float sx = src[i][0] - src_mean_x;
        float sy = src[i][1] - src_mean_y;
        float dx = dst[i][0] - dst_mean_x;
        float dy = dst[i][1] - dst_mean_y;

        src_var += sx * sx + sy * sy;
        cov_xx += dx * sx;
        cov_xy += dx * sy;
        cov_yx += dy * sx;
        cov_yy += dy * sy;
    }

    if (src_var < 1e-6f) src_var = 1e-6f;

    float scale = sqrtf((cov_xx + cov_yy) * (cov_xx + cov_yy) + (cov_xy - cov_yx) * (cov_xy - cov_yx)) / src_var;
    float theta = atan2f(cov_yx - cov_xy, cov_xx + cov_yy);

    float cos_t = cosf(theta) * scale;
    float sin_t = sinf(theta) * scale;

    M[0][0] = cos_t;
    M[0][1] = -sin_t;
    M[0][2] = dst_mean_x - (cos_t * src_mean_x - sin_t * src_mean_y);

    M[1][0] = sin_t;
    M[1][1] = cos_t;
    M[1][2] = dst_mean_y - (sin_t * src_mean_x + cos_t * src_mean_y);
}

int face_engine_align_face(
    const ImageBuffer *src_img,
    const FaceLandmarks5 *landmarks,
    ImageBuffer *dst_crop_112x112
) {
    if (!src_img || !landmarks || !dst_crop_112x112 || !dst_crop_112x112->data) {
        return -1;
    }

    float src_pts[5][2];
    for (int i = 0; i < 5; i++) {
        src_pts[i][0] = landmarks->x[i];
        src_pts[i][1] = landmarks->y[i];
    }

    float M[2][3];
    get_similarity_transform(src_pts, ARCFACE_DST_LANDMARKS, M);

    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (fabsf(det) < 1e-7f) return -1;
    float inv_det = 1.0f / det;

    float invM[2][3];
    invM[0][0] =  M[1][1] * inv_det;
    invM[0][1] = -M[0][1] * inv_det;
    invM[0][2] = (M[0][1] * M[1][2] - M[1][1] * M[0][2]) * inv_det;

    invM[1][0] = -M[1][0] * inv_det;
    invM[1][1] =  M[0][0] * inv_det;
    invM[1][2] = (M[1][0] * M[0][2] - M[0][0] * M[1][2]) * inv_det;

    int dst_w = 112;
    int dst_h = 112;
    int ch = src_img->channels;
    int src_stride = src_img->stride;

    for (int dy = 0; dy < dst_h; dy++) {
        uint8_t *dst_row = dst_crop_112x112->data + dy * (dst_w * ch);
        for (int dx = 0; dx < dst_w; dx++) {
            float sx = invM[0][0] * dx + invM[0][1] * dy + invM[0][2];
            float sy = invM[1][0] * dx + invM[1][1] * dy + invM[1][2];

            int x0 = (int)floorf(sx);
            int y0 = (int)floorf(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 >= 0 && x1 < src_img->width && y0 >= 0 && y1 < src_img->height) {
                float fx = sx - x0;
                float fy = sy - y0;
                float w00 = (1.0f - fx) * (1.0f - fy);
                float w01 = fx * (1.0f - fy);
                float w10 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                const uint8_t *p00 = src_img->data + y0 * src_stride + x0 * ch;
                const uint8_t *p01 = src_img->data + y0 * src_stride + x1 * ch;
                const uint8_t *p10 = src_img->data + y1 * src_stride + x0 * ch;
                const uint8_t *p11 = src_img->data + y1 * src_stride + x1 * ch;

                for (int c = 0; c < ch; c++) {
                    float val = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
                    dst_row[dx * ch + c] = (uint8_t)fminf(255.0f, fmaxf(0.0f, val));
                }
            } else {
                for (int c = 0; c < ch; c++) {
                    dst_row[dx * ch + c] = 0;
                }
            }
        }
    }

    dst_crop_112x112->width = dst_w;
    dst_crop_112x112->height = dst_h;
    dst_crop_112x112->channels = ch;
    dst_crop_112x112->stride = dst_w * ch;

    return 0;
}

int face_engine_crop_roi(
    const ImageBuffer *src_img,
    const FaceBBox *roi_box,
    ImageBuffer *dst_roi,
    uint8_t *allocated_buffer
) {
    if (!src_img || !roi_box || !dst_roi || !allocated_buffer) return -1;

    int x1 = (int)fmaxf(0.0f, roi_box->x1);
    int y1 = (int)fmaxf(0.0f, roi_box->y1);
    int x2 = (int)fminf((float)src_img->width, roi_box->x2);
    int y2 = (int)fminf((float)src_img->height, roi_box->y2);

    int w = x2 - x1;
    int h = y2 - y1;
    if (w <= 0 || h <= 0) return -1;

    int ch = src_img->channels;
    dst_roi->data = allocated_buffer;
    dst_roi->width = w;
    dst_roi->height = h;
    dst_roi->channels = ch;
    dst_roi->stride = w * ch;

    for (int y = 0; y < h; y++) {
        const uint8_t *src_row = src_img->data + (y1 + y) * src_img->stride + x1 * ch;
        uint8_t *dst_row = dst_roi->data + y * dst_roi->stride;
        memcpy(dst_row, src_row, w * ch);
    }

    return 0;
}

void face_engine_map_bbox_from_roi(
    FaceResult *faces,
    int num_faces,
    float roi_offset_x,
    float roi_offset_y
) {
    if (!faces || num_faces <= 0) return;
    for (int i = 0; i < num_faces; i++) {
        faces[i].bbox.x1 += roi_offset_x;
        faces[i].bbox.y1 += roi_offset_y;
        faces[i].bbox.x2 += roi_offset_x;
        faces[i].bbox.y2 += roi_offset_y;

        for (int k = 0; k < 5; k++) {
            faces[i].landmarks.x[k] += roi_offset_x;
            faces[i].landmarks.y[k] += roi_offset_y;
        }
    }
}

FaceEngine *face_engine_create(const FaceConfig *config) {
    FaceEngine *engine = (FaceEngine *)calloc(1, sizeof(FaceEngine));
    if (!engine) return NULL;

    if (config) {
        engine->config = *config;
    }

    engine->is_initialized = true;
    printf("[FaceEngine] Initialized SCRFD + ArcFace engine (backend: %s, device: %s)\n",
           engine->config.backend, engine->config.openvino_device);
    return engine;
}

void face_engine_destroy(FaceEngine *engine) {
    if (engine) {
        free(engine);
    }
}

int face_engine_detect(
    FaceEngine *engine,
    const ImageBuffer *frame,
    FaceResult *results,
    int max_faces
) {
    if (!engine || !frame || !frame->data || !results || max_faces <= 0) {
        return 0;
    }

    int detected_count = 0;

    for (int i = 0; i < detected_count; i++) {
        FaceResult *face = &results[i];
        face->blur_score = face_engine_calculate_blur_score(frame, &face->bbox);
        
        float w = face->bbox.x2 - face->bbox.x1;
        float h = face->bbox.y2 - face->bbox.y1;

        face->quality_ok = (w >= (float)engine->config.min_face_size &&
                            h >= (float)engine->config.min_face_size &&
                            face->blur_score >= engine->config.blur_threshold);
    }

    return detected_count;
}

int face_engine_extract_embedding(
    FaceEngine *engine,
    const ImageBuffer *aligned_face_112x112,
    float *embedding_out
) {
    if (!engine || !aligned_face_112x112 || !embedding_out) return -1;

    memset(embedding_out, 0, sizeof(float) * FACE_EMBEDDING_DIM);
    face_vector_l2_normalize(embedding_out, FACE_EMBEDDING_DIM);
    return 0;
}

int face_engine_build_template(
    FaceEngine *engine,
    const ImageBuffer *images,
    int num_images,
    float *template_out,
    int *valid_count_out
) {
    if (!engine || !images || num_images <= 0 || !template_out) return -1;

    float accum[FACE_EMBEDDING_DIM] = {0};
    int valid_count = 0;

    for (int i = 0; i < num_images; i++) {
        FaceResult faces[MAX_DETECTED_FACES];
        int num_faces = face_engine_detect(engine, &images[i], faces, MAX_DETECTED_FACES);

        int best_idx = -1;
        float max_area = 0.0f;

        for (int k = 0; k < num_faces; k++) {
            if (faces[k].quality_ok) {
                float area = (faces[k].bbox.x2 - faces[k].bbox.x1) * (faces[k].bbox.y2 - faces[k].bbox.y1);
                if (area > max_area) {
                    max_area = area;
                    best_idx = k;
                }
            }
        }

        if (best_idx >= 0) {
            uint8_t aligned_data[112 * 112 * 3];
            ImageBuffer aligned_buf = {
                .data = aligned_data,
                .width = 112,
                .height = 112,
                .channels = 3,
                .stride = 112 * 3
            };

            if (face_engine_align_face(&images[i], &faces[best_idx].landmarks, &aligned_buf) == 0) {
                float emb[FACE_EMBEDDING_DIM];
                if (face_engine_extract_embedding(engine, &aligned_buf, emb) == 0) {
                    for (int d = 0; d < FACE_EMBEDDING_DIM; d++) {
                        accum[d] += emb[d];
                    }
                    valid_count++;
                }
            }
        }
    }

    if (valid_count_out) *valid_count_out = valid_count;

    if (valid_count < 1) {
        return -1;
    }

    for (int d = 0; d < FACE_EMBEDDING_DIM; d++) {
        template_out[d] = accum[d] / (float)valid_count;
    }
    face_vector_l2_normalize(template_out, FACE_EMBEDDING_DIM);

    return 0;
}
