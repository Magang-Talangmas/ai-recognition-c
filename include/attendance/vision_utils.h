#ifndef ATTENDANCE_VISION_UTILS_H
#define ATTENDANCE_VISION_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include "attendance/config.h"
#include "attendance/face_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VOTE_WINDOW 32
#define MAX_TRACKED_PEOPLE 128

typedef enum {
    CANDIDATE_STATE_NEW = 0,
    CANDIDATE_STATE_VOTING = 1,
    CANDIDATE_STATE_CONFIRMED = 2,
    CANDIDATE_STATE_COOLDOWN = 3,
    CANDIDATE_STATE_EXPIRED = 4
} CandidateState;

typedef struct {
    char employee_id[128];
    bool has_id;
    float score;
} IdentityVote;

typedef struct {
    IdentityVote votes[MAX_VOTE_WINDOW];
    int count;
    int head;
    int capacity;
} TrackVoteQueue;

typedef struct {
    int64_t track_id;
    CandidateState state;
    FaceBBox last_bbox;
    FaceLandmarks5 last_landmarks;
    float cx;
    float cy;
    double first_seen_time;
    double last_seen_time;
    double last_arcface_time;
    int arcface_eval_count;
    TrackVoteQueue vote_queue;
    char confirmed_id[128];
    float confirmed_score;
    bool is_confirmed;
    bool event_dispatched;
    int previous_side; /* -1 = above line, 1 = below line, 0 = unset */
} TrackedCandidate;

typedef struct {
    TrackedCandidate candidates[MAX_TRACKED_PEOPLE];
    int count;
    float max_match_dist;
    int round_robin_cursor;
} CandidateTable;

/* Backwards-compatible alias for existing tests */
typedef CandidateTable CentroidTracker;

/* Initialize candidate table */
void candidate_table_init(CandidateTable *table, float max_dist);
void centroid_tracker_init(CentroidTracker *tracker, float max_dist);

/* Update or create candidate for a detected face */
int64_t candidate_table_update_face(
    CandidateTable *table,
    const FaceResult *face,
    double now_time
);
int64_t centroid_tracker_update_face(
    CentroidTracker *tracker,
    const FaceBBox *bbox,
    double now_time
);

/* Clean candidates not seen for longer than timeout_seconds */
void candidate_table_clean_stale(CandidateTable *table, double now_time, double timeout_seconds);
void centroid_tracker_clean_stale(CentroidTracker *tracker, double now_time, double timeout_seconds);

/* Select high-priority candidates for ArcFace inference based on budget */
int candidate_table_select_for_arcface(
    CandidateTable *table,
    int max_budget,
    int64_t *out_track_ids,
    int max_out
);

/* Add identity vote to candidate queue */
void track_vote_push(TrackVoteQueue *q, const char *employee_id, float score, int capacity);

/* Evaluate stable identity from vote queue based on required majority count */
bool track_vote_get_stable(
    const TrackVoteQueue *q,
    int required_votes,
    char *out_employee_id,
    size_t id_buf_size,
    float *out_max_score
);

/* Face Quality Gate: tests size, Laplacian blur, and 5-point landmark pose/yaw */
bool face_quality_gate_check(
    const FaceResult *face,
    const FaceConfig *config,
    char *out_reason,
    size_t reason_size
);

/* Calculate pixel ROI bounding box from ratios */
FaceBBox calculate_roi_bbox(int frame_width, int frame_height, const RoiConfig *roi_cfg);

/* Check if face bounding box is inside the ROI zone */
bool is_bbox_in_roi(const FaceBBox *face_box, const FaceBBox *roi_box, float min_overlap_ratio);

/* Detect motion / pixel changes in ROI area */
bool detect_roi_motion(
    const ImageBuffer *prev_frame,
    const ImageBuffer *curr_frame,
    const FaceBBox *roi_box,
    float intensity_threshold,
    int min_pixel_count
);

/* Determine crossing direction across virtual line */
const char *crossing_direction(
    int previous_side,
    int current_side,
    bool inside_is_below_line
);

/* High-resolution monotonic time in seconds */
double get_monotonic_time_seconds(void);

/* Current UTC ISO8601 timestamp string (e.g. 2026-08-05T01:45:00Z) */
void get_iso8601_timestamp(char *buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif /* ATTENDANCE_VISION_UTILS_H */
