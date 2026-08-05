#include "attendance/vision_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

static int64_t global_track_counter = 1000;

double get_monotonic_time_seconds(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static BOOL initialized = FALSE;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = TRUE;
    }
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

void get_iso8601_timestamp(char *buf, size_t buf_size) {
    if (!buf || buf_size == 0) return;
    time_t raw_time = time(NULL);
    struct tm tm_utc;
#ifdef _WIN32
    gmtime_s(&tm_utc, &raw_time);
#else
    gmtime_r(&raw_time, &tm_utc);
#endif
    strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

void candidate_table_init(CandidateTable *table, float max_dist) {
    if (!table) return;
    memset(table, 0, sizeof(CandidateTable));
    table->max_match_dist = (max_dist > 0.0f) ? max_dist : 120.0f;
    table->round_robin_cursor = 0;
}

void centroid_tracker_init(CentroidTracker *tracker, float max_dist) {
    candidate_table_init(tracker, max_dist);
}

int64_t candidate_table_update_face(
    CandidateTable *table,
    const FaceResult *face,
    double now_time
) {
    if (!table || !face) return -1;

    float cx = (face->bbox.x1 + face->bbox.x2) * 0.5f;
    float cy = (face->bbox.y1 + face->bbox.y2) * 0.5f;

    int best_match_idx = -1;
    float min_dist_sq = table->max_match_dist * table->max_match_dist;

    for (int i = 0; i < table->count; i++) {
        TrackedCandidate *tc = &table->candidates[i];
        if (tc->state == CANDIDATE_STATE_EXPIRED) continue;

        float dx = tc->cx - cx;
        float dy = tc->cy - cy;
        float dist_sq = dx * dx + dy * dy;

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_match_idx = i;
        }
    }

    if (best_match_idx >= 0) {
        TrackedCandidate *tc = &table->candidates[best_match_idx];
        tc->cx = cx;
        tc->cy = cy;
        tc->last_bbox = face->bbox;
        tc->last_landmarks = face->landmarks;
        tc->last_seen_time = now_time;
        if (tc->state == CANDIDATE_STATE_NEW) {
            tc->state = CANDIDATE_STATE_VOTING;
        }
        return tc->track_id;
    }

    /* Assign new candidate */
    if (table->count < MAX_TRACKED_PEOPLE) {
        int idx = table->count++;
        TrackedCandidate *tc = &table->candidates[idx];
        memset(tc, 0, sizeof(TrackedCandidate));
        tc->track_id = ++global_track_counter;
        tc->state = CANDIDATE_STATE_NEW;
        tc->cx = cx;
        tc->cy = cy;
        tc->last_bbox = face->bbox;
        tc->last_landmarks = face->landmarks;
        tc->first_seen_time = now_time;
        tc->last_seen_time = now_time;
        tc->previous_side = 0;
        return tc->track_id;
    }

    return -1;
}

int64_t centroid_tracker_update_face(
    CentroidTracker *tracker,
    const FaceBBox *bbox,
    double now_time
) {
    if (!tracker || !bbox) return -1;
    FaceResult dummy;
    memset(&dummy, 0, sizeof(dummy));
    dummy.bbox = *bbox;
    return candidate_table_update_face(tracker, &dummy, now_time);
}

void candidate_table_clean_stale(CandidateTable *table, double now_time, double timeout_seconds) {
    if (!table) return;
    int write_idx = 0;
    for (int i = 0; i < table->count; i++) {
        TrackedCandidate *tc = &table->candidates[i];
        if ((now_time - tc->last_seen_time) <= timeout_seconds) {
            if (write_idx != i) {
                table->candidates[write_idx] = *tc;
            }
            write_idx++;
        }
    }
    table->count = write_idx;
    if (table->round_robin_cursor >= table->count) {
        table->round_robin_cursor = 0;
    }
}

void centroid_tracker_clean_stale(CentroidTracker *tracker, double now_time, double timeout_seconds) {
    candidate_table_clean_stale(tracker, now_time, timeout_seconds);
}

int candidate_table_select_for_arcface(
    CandidateTable *table,
    int max_budget,
    int64_t *out_track_ids,
    int max_out
) {
    if (!table || !out_track_ids || max_budget <= 0 || max_out <= 0 || table->count == 0) {
        return 0;
    }

    int limit = (max_budget < max_out) ? max_budget : max_out;
    int selected = 0;

    /* First pass: High priority for active candidates in VOTING or NEW state */
    int start = table->round_robin_cursor % table->count;
    for (int step = 0; step < table->count && selected < limit; step++) {
        int idx = (start + step) % table->count;
        TrackedCandidate *tc = &table->candidates[idx];

        if (tc->state == CANDIDATE_STATE_NEW || tc->state == CANDIDATE_STATE_VOTING) {
            out_track_ids[selected++] = tc->track_id;
            table->round_robin_cursor = (idx + 1) % table->count;
        }
    }

    /* Second pass if budget remains and unconfirmed candidates exist */
    if (selected < limit) {
        for (int step = 0; step < table->count && selected < limit; step++) {
            int idx = (start + step) % table->count;
            TrackedCandidate *tc = &table->candidates[idx];
            if (!tc->is_confirmed && tc->state != CANDIDATE_STATE_COOLDOWN) {
                bool already_added = false;
                for (int s = 0; s < selected; s++) {
                    if (out_track_ids[s] == tc->track_id) {
                        already_added = true;
                        break;
                    }
                }
                if (!already_added) {
                    out_track_ids[selected++] = tc->track_id;
                    table->round_robin_cursor = (idx + 1) % table->count;
                }
            }
        }
    }

    return selected;
}

void track_vote_push(TrackVoteQueue *q, const char *employee_id, float score, int capacity) {
    if (!q || capacity <= 0) return;
    if (capacity > MAX_VOTE_WINDOW) capacity = MAX_VOTE_WINDOW;
    q->capacity = capacity;

    int insert_idx;
    if (q->count < capacity) {
        insert_idx = q->count++;
    } else {
        insert_idx = q->head;
        q->head = (q->head + 1) % capacity;
    }

    IdentityVote *v = &q->votes[insert_idx];
    if (employee_id && employee_id[0]) {
        strncpy(v->employee_id, employee_id, sizeof(v->employee_id) - 1);
        v->employee_id[sizeof(v->employee_id) - 1] = '\0';
        v->has_id = true;
    } else {
        v->employee_id[0] = '\0';
        v->has_id = false;
    }
    v->score = score;
}

bool track_vote_get_stable(
    const TrackVoteQueue *q,
    int required_votes,
    char *out_employee_id,
    size_t id_buf_size,
    float *out_max_score
) {
    if (!q || q->count == 0 || required_votes <= 0) return false;

    typedef struct {
        char id[128];
        int count;
        float max_score;
    } Tally;

    Tally tallies[MAX_VOTE_WINDOW];
    int tally_count = 0;

    for (int i = 0; i < q->count; i++) {
        const IdentityVote *v = &q->votes[i];
        if (!v->has_id) continue;

        int found_idx = -1;
        for (int t = 0; t < tally_count; t++) {
            if (strcmp(tallies[t].id, v->employee_id) == 0) {
                found_idx = t;
                break;
            }
        }

        if (found_idx >= 0) {
            tallies[found_idx].count++;
            if (v->score > tallies[found_idx].max_score) {
                tallies[found_idx].max_score = v->score;
            }
        } else if (tally_count < MAX_VOTE_WINDOW) {
            strncpy(tallies[tally_count].id, v->employee_id, sizeof(tallies[tally_count].id) - 1);
            tallies[tally_count].count = 1;
            tallies[tally_count].max_score = v->score;
            tally_count++;
        }
    }

    int best_tally_idx = -1;
    int max_votes = 0;

    for (int t = 0; t < tally_count; t++) {
        if (tallies[t].count > max_votes) {
            max_votes = tallies[t].count;
            best_tally_idx = t;
        }
    }

    if (best_tally_idx >= 0 && max_votes >= required_votes) {
        if (out_employee_id && id_buf_size > 0) {
            strncpy(out_employee_id, tallies[best_tally_idx].id, id_buf_size - 1);
            out_employee_id[id_buf_size - 1] = '\0';
        }
        if (out_max_score) {
            *out_max_score = tallies[best_tally_idx].max_score;
        }
        return true;
    }

    return false;
}

bool face_quality_gate_check(
    const FaceResult *face,
    const FaceConfig *config,
    char *out_reason,
    size_t reason_size
) {
    if (!face || !config) return false;

    float width = face->bbox.x2 - face->bbox.x1;
    float height = face->bbox.y2 - face->bbox.y1;

    /* 1. Size check */
    if (width < (float)config->min_face_size || height < (float)config->min_face_size) {
        if (out_reason && reason_size > 0) {
            snprintf(out_reason, reason_size, "Face too small (%.0fx%.0f < %dpx)",
                     width, height, config->min_face_size);
        }
        return false;
    }

    /* 2. Sharpness / Blur score check */
    if (face->blur_score > 0.0f && face->blur_score < config->blur_threshold) {
        if (out_reason && reason_size > 0) {
            snprintf(out_reason, reason_size, "Face too blurry (score %.1f < %.1f)",
                     face->blur_score, config->blur_threshold);
        }
        return false;
    }

    /* 3. Pose / Landmark Symmetry check (Yaw angle) */
    float lx = face->landmarks.x[0];
    float ly = face->landmarks.y[0];
    float rx = face->landmarks.x[1];
    float ry = face->landmarks.y[1];
    float nx = face->landmarks.x[2];
    float ny = face->landmarks.y[2];

    float eye_dx = rx - lx;
    float eye_dy = ry - ly;
    float eye_dist = sqrtf(eye_dx * eye_dx + eye_dy * eye_dy);

    if (eye_dist > 5.0f) {
        float l_to_nose = sqrtf((nx - lx) * (nx - lx) + (ny - ly) * (ny - ly));
        float r_to_nose = sqrtf((rx - nx) * (rx - nx) + (ry - ny) * (ry - ny));
        float yaw_asymmetry = fabsf(l_to_nose - r_to_nose) / eye_dist;

        float max_yaw = (config->pose_max_yaw_ratio > 0.0f) ? config->pose_max_yaw_ratio : 0.35f;
        if (yaw_asymmetry > max_yaw) {
            if (out_reason && reason_size > 0) {
                snprintf(out_reason, reason_size, "Extreme face yaw angle (asymmetry %.2f > %.2f)",
                         yaw_asymmetry, max_yaw);
            }
            return false;
        }
    }

    if (out_reason && reason_size > 0) {
        snprintf(out_reason, reason_size, "Quality OK");
    }
    return true;
}

FaceBBox calculate_roi_bbox(int frame_width, int frame_height, const RoiConfig *roi_cfg) {
    FaceBBox box = {0, 0, (float)frame_width, (float)frame_height};
    if (!roi_cfg || !roi_cfg->enabled) return box;

    box.x1 = (float)frame_width * roi_cfg->x1_ratio;
    box.y1 = (float)frame_height * roi_cfg->y1_ratio;
    box.x2 = (float)frame_width * roi_cfg->x2_ratio;
    box.y2 = (float)frame_height * roi_cfg->y2_ratio;

    if (box.x1 < 0.0f) box.x1 = 0.0f;
    if (box.y1 < 0.0f) box.y1 = 0.0f;
    if (box.x2 > (float)frame_width) box.x2 = (float)frame_width;
    if (box.y2 > (float)frame_height) box.y2 = (float)frame_height;

    return box;
}

bool is_bbox_in_roi(const FaceBBox *face_box, const FaceBBox *roi_box, float min_overlap_ratio) {
    if (!face_box || !roi_box) return false;

    float inter_x1 = (face_box->x1 > roi_box->x1) ? face_box->x1 : roi_box->x1;
    float inter_y1 = (face_box->y1 > roi_box->y1) ? face_box->y1 : roi_box->y1;
    float inter_x2 = (face_box->x2 < roi_box->x2) ? face_box->x2 : roi_box->x2;
    float inter_y2 = (face_box->y2 < roi_box->y2) ? face_box->y2 : roi_box->y2;

    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) return false;

    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float face_area = (face_box->x2 - face_box->x1) * (face_box->y2 - face_box->y1);

    if (face_area <= 0.0f) return false;
    float ratio = inter_area / face_area;
    return (ratio >= min_overlap_ratio);
}

bool detect_roi_motion(
    const ImageBuffer *prev_frame,
    const ImageBuffer *curr_frame,
    const FaceBBox *roi_box,
    float intensity_threshold,
    int min_pixel_count
) {
    if (!prev_frame || !curr_frame || !prev_frame->data || !curr_frame->data) {
        return true; /* First frame, default to active */
    }
    if (prev_frame->width != curr_frame->width || prev_frame->height != curr_frame->height) {
        return true;
    }

    int x1 = (int)roi_box->x1;
    int y1 = (int)roi_box->y1;
    int x2 = (int)roi_box->x2;
    int y2 = (int)roi_box->y2;

    if (x1 < 0) x1 = 0;
    if (y1 < 0) y1 = 0;
    if (x2 > curr_frame->width) x2 = curr_frame->width;
    if (y2 > curr_frame->height) y2 = curr_frame->height;

    if (x2 <= x1 || y2 <= y1) return true;

    int diff_count = 0;
    int step = 4; /* Downsample step for blazing fast frame diff */
    int thresh_int = (int)intensity_threshold;

    for (int y = y1; y < y2; y += step) {
        const uint8_t *p_row = prev_frame->data + y * prev_frame->stride;
        const uint8_t *c_row = curr_frame->data + y * curr_frame->stride;

        for (int x = x1; x < x2; x += step) {
            int idx = x * 3;
            /* Grayscale luminance delta: Y = 0.299R + 0.587G + 0.114B (approx with int: (2B + 5G + 1R)/8) */
            int p_lum = (p_row[idx] * 2 + p_row[idx + 1] * 5 + p_row[idx + 2]) >> 3;
            int c_lum = (c_row[idx] * 2 + c_row[idx + 1] * 5 + c_row[idx + 2]) >> 3;
            int delta = abs(c_lum - p_lum);

            if (delta >= thresh_int) {
                diff_count++;
                if (diff_count >= min_pixel_count) {
                    return true;
                }
            }
        }
    }

    return (diff_count >= min_pixel_count);
}

const char *crossing_direction(
    int previous_side,
    int current_side,
    bool inside_is_below_line
) {
    if (previous_side == 0 || current_side == 0) return NULL;
    if (previous_side == current_side) return NULL;

    if (inside_is_below_line) {
        if (previous_side < 0 && current_side > 0) return "ENTER";
        if (previous_side > 0 && current_side < 0) return "EXIT";
    } else {
        if (previous_side > 0 && current_side < 0) return "ENTER";
        if (previous_side < 0 && current_side > 0) return "EXIT";
    }
    return NULL;
}
