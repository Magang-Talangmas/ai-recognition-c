#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "attendance/config.h"
#include "attendance/database.h"
#include "attendance/face_engine.h"
#include "attendance/matcher.h"
#include "attendance/vision_utils.h"
#include "third_party/cJSON.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr, msg) \
    do { \
        if (expr) { \
            printf("  [PASS] %s\n", msg); \
            tests_passed++; \
        } else { \
            printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
            tests_failed++; \
        } \
    } while(0)

static void test_config(void) {
    printf("\n=== TEST 1: Config & Environment Expansion ===\n");
    AppConfig cfg;
    config_load(NULL, &cfg);

    ASSERT_TRUE(cfg.camera.process_every_n_frames > 0, "process_every_n_frames is set");
    ASSERT_TRUE(cfg.face.vote_window == 5, "Default vote_window is 5");
    ASSERT_TRUE(cfg.face.votes_required == 3, "Default votes_required is 3 (3/5 majority)");
    ASSERT_TRUE(cfg.roi.enabled == true, "ROI gating is enabled by default");
    ASSERT_TRUE(cfg.motion.enabled == true, "Motion gating is enabled by default");
    ASSERT_TRUE(cfg.scheduler.max_arcface_per_frame == 2, "Scheduler max_arcface_per_frame is 2");

    char expanded[256];
    config_expand_env("Hello ${PATH}", expanded, sizeof(expanded));
    ASSERT_TRUE(strlen(expanded) > 6, "Expanded ${PATH} environment variable");
}

static void test_roi_and_motion_gating(void) {
    printf("\n=== TEST 2: Recognition Zone (ROI) & Motion Gating ===\n");

    RoiConfig roi_cfg = {
        .enabled = true,
        .x1_ratio = 0.50f,
        .y1_ratio = 0.10f,
        .x2_ratio = 1.00f,
        .y2_ratio = 0.90f
    };

    FaceBBox roi_box = calculate_roi_bbox(1000, 800, &roi_cfg);
    ASSERT_TRUE(roi_box.x1 == 500.0f && roi_box.y1 == 80.0f &&
                roi_box.x2 == 1000.0f && roi_box.y2 == 720.0f,
                "Calculated pixel coordinates for ROI bounding box");

    /* Face inside right-side ROI */
    FaceBBox face_inside = {600.0f, 200.0f, 700.0f, 350.0f};
    ASSERT_TRUE(is_bbox_in_roi(&face_inside, &roi_box, 0.40f), "Face inside ROI is approved");

    /* Face outside left-side (e.g. employee sitting at desk) */
    FaceBBox face_outside = {100.0f, 200.0f, 200.0f, 350.0f};
    ASSERT_TRUE(!is_bbox_in_roi(&face_outside, &roi_box, 0.40f), "Face outside ROI (at desk) is rejected");

    /* Test Motion Detection */
    uint8_t prev_raw[640 * 480 * 3] = {0};
    uint8_t curr_raw[640 * 480 * 3] = {0};

    ImageBuffer prev_img = {prev_raw, 640, 480, 3, 640 * 3};
    ImageBuffer curr_img = {curr_raw, 640, 480, 3, 640 * 3};

    FaceBBox test_roi = {0, 0, 640, 480};

    /* Identical frames -> No motion */
    bool motion_static = detect_roi_motion(&prev_img, &curr_img, &test_roi, 15.0f, 50);
    ASSERT_TRUE(!motion_static, "Static identical frames report no motion");

    /* Add moving pixel block in current frame */
    for (int y = 100; y < 150; y++) {
        for (int x = 100; x < 150; x++) {
            curr_raw[(y * 640 + x) * 3] = 200;
        }
    }

    bool motion_active = detect_roi_motion(&prev_img, &curr_img, &test_roi, 15.0f, 50);
    ASSERT_TRUE(motion_active, "Frame difference correctly detects motion in ROI");
}

static void test_face_quality_gate(void) {
    printf("\n=== TEST 3: Face Quality Gate (Blur, Size, Yaw Angle) ===\n");

    FaceConfig face_cfg;
    memset(&face_cfg, 0, sizeof(face_cfg));
    face_cfg.min_face_size = 40;
    face_cfg.blur_threshold = 25.0f;
    face_cfg.pose_max_yaw_ratio = 0.35f;

    char reason[128] = {0};

    /* Case A: Face too small */
    FaceResult face_small = {
        .bbox = {10, 10, 30, 30}, /* 20x20px */
        .blur_score = 50.0f
    };
    ASSERT_TRUE(!face_quality_gate_check(&face_small, &face_cfg, reason, sizeof(reason)), "Small face (20x20) is rejected");

    /* Case B: Face blurry */
    FaceResult face_blurry = {
        .bbox = {10, 10, 100, 100}, /* 90x90px */
        .blur_score = 12.0f /* below 25.0 threshold */
    };
    ASSERT_TRUE(!face_quality_gate_check(&face_blurry, &face_cfg, reason, sizeof(reason)), "Blurry face is rejected");

    /* Case C: Extreme yaw / turned profile */
    FaceResult face_profile = {
        .bbox = {10, 10, 100, 100},
        .blur_score = 50.0f,
        .landmarks = {
            .x = {20.0f, 80.0f, 25.0f, 30.0f, 75.0f}, /* nose close to left eye */
            .y = {40.0f, 40.0f, 60.0f, 80.0f, 80.0f}
        }
    };
    ASSERT_TRUE(!face_quality_gate_check(&face_profile, &face_cfg, reason, sizeof(reason)), "Steep yaw profile face is rejected");

    /* Case D: Good frontal face */
    FaceResult face_good = {
        .bbox = {10, 10, 100, 100},
        .blur_score = 50.0f,
        .landmarks = {
            .x = {30.0f, 70.0f, 50.0f, 35.0f, 65.0f}, /* symmetric */
            .y = {40.0f, 40.0f, 60.0f, 80.0f, 80.0f}
        }
    };
    ASSERT_TRUE(face_quality_gate_check(&face_good, &face_cfg, reason, sizeof(reason)), "Good frontal face passes Quality Gate");
}

static void test_candidate_lifecycle_and_scheduler(void) {
    printf("\n=== TEST 4: Candidate Table & Priority Round-Robin Scheduler ===\n");

    CandidateTable table;
    candidate_table_init(&table, 120.0f);

    FaceResult f1 = {
        .bbox = {100, 100, 200, 200}
    };
    FaceResult f2 = {
        .bbox = {400, 100, 500, 200}
    };

    int64_t id1 = candidate_table_update_face(&table, &f1, 1.0);
    int64_t id2 = candidate_table_update_face(&table, &f2, 1.0);

    ASSERT_TRUE(id1 > 0 && id2 > 0 && id1 != id2, "Assigned unique track IDs for multiple candidates");
    ASSERT_TRUE(table.count == 2, "Candidate table contains 2 active people");

    /* Test Round-Robin scheduling budget */
    int64_t selected[4];
    int sel_count = candidate_table_select_for_arcface(&table, 1, selected, 4);
    ASSERT_TRUE(sel_count == 1 && selected[0] == id1, "Scheduler picked candidate 1 under budget of 1");

    sel_count = candidate_table_select_for_arcface(&table, 1, selected, 4);
    ASSERT_TRUE(sel_count == 1 && selected[0] == id2, "Scheduler rotated fairly to candidate 2 on next turn");

    /* Test Stale Candidate Cleanup */
    candidate_table_clean_stale(&table, 10.0, 5.0);
    ASSERT_TRUE(table.count == 0, "Stale candidates (inactive > 5s) cleaned up");
}

static void test_matcher_and_cooldown(void) {
    printf("\n=== TEST 5: Vector Normalization, Matcher & In-Memory Cooldown ===\n");

    float vec[FACE_EMBEDDING_DIM];
    for (int i = 0; i < FACE_EMBEDDING_DIM; i++) vec[i] = (float)(i + 1);
    face_vector_l2_normalize(vec, FACE_EMBEDDING_DIM);

    double sum_sq = 0.0;
    for (int i = 0; i < FACE_EMBEDDING_DIM; i++) sum_sq += vec[i] * vec[i];
    ASSERT_TRUE(fabs(sum_sq - 1.0) < 1e-4, "L2 normalization yields unit length vector");

    /* Test template database save & load */
    char ids[2][128] = {"EMP-001", "EMP-002"};
    float tmpls[2 * FACE_EMBEDDING_DIM];
    for (int i = 0; i < FACE_EMBEDDING_DIM; i++) {
        tmpls[i] = (i == 0) ? 1.0f : 0.0f; /* 100% direction 0 */
        tmpls[FACE_EMBEDDING_DIM + i] = (i == 1) ? 1.0f : 0.0f; /* 100% direction 1 */
    }

    const char *test_db_file = "test_embeddings.bin";
    int save_rc = face_matcher_save_database(test_db_file, ids, tmpls, 2);
    ASSERT_TRUE(save_rc == 0, "Saved test binary embedding database");

    FaceConfig fcfg;
    memset(&fcfg, 0, sizeof(fcfg));
    fcfg.match_threshold = 0.25f;
    fcfg.match_margin = 0.05f;

    FaceMatcher *matcher = face_matcher_create(test_db_file, &fcfg);
    ASSERT_TRUE(matcher != NULL, "Loaded test FaceMatcher from file");
    ASSERT_TRUE(face_matcher_get_count(matcher) == 2, "Matcher loaded exactly 2 templates");

    /* Query matching EMP-001 */
    float q1[FACE_EMBEDDING_DIM] = {0};
    q1[0] = 1.0f;
    MatchResult res1 = face_matcher_match(matcher, q1);
    ASSERT_TRUE(res1.is_matched && strcmp(res1.employee_id, "EMP-001") == 0, "Query matched EMP-001 with high score");
    ASSERT_TRUE(res1.score > 0.99f, "Score is ~1.0 for identical vector");

    /* Test In-Memory Cooldown */
    ASSERT_TRUE(!face_matcher_is_in_cooldown(matcher, "EMP-001", 100.0, 300.0), "EMP-001 not in cooldown initially");
    face_matcher_record_cooldown(matcher, "EMP-001", 100.0);
    ASSERT_TRUE(face_matcher_is_in_cooldown(matcher, "EMP-001", 150.0, 300.0), "EMP-001 is in cooldown 50s later");
    ASSERT_TRUE(!face_matcher_is_in_cooldown(matcher, "EMP-001", 450.0, 300.0), "EMP-001 cooldown expires after 350s");

    face_matcher_destroy(matcher);
    remove(test_db_file);
}

static void test_sqlite_database(void) {
    printf("\n=== TEST 6: SQLite Database Event Lifecycle ===\n");

    const char *test_db_file = "test_attendance.db";
    remove(test_db_file);

    AttendanceDB *db = attendance_db_open(test_db_file, 5);
    ASSERT_TRUE(db != NULL, "Opened test SQLite database");

    int64_t ev_id = attendance_db_create_pending_event(
        db,
        "test-cam",
        1001,
        "EMP-999",
        "ENTER",
        "PENDING_CONFIRMATION",
        0.88f
    );
    ASSERT_TRUE(ev_id > 0, "Created PENDING_CONFIRMATION attendance event");

    /* Check duplicate suppression */
    int64_t dup_id = attendance_db_create_pending_event(
        db,
        "test-cam",
        1001,
        "EMP-999",
        "ENTER",
        "PENDING_CONFIRMATION",
        0.89f
    );
    ASSERT_TRUE(dup_id == -1, "Duplicate event within cooldown properly suppressed");

    /* List events */
    AttendanceEvent events[10];
    int count = attendance_db_list_events(db, events, 10);
    ASSERT_TRUE(count == 1, "Listed 1 event from database");
    ASSERT_TRUE(strcmp(events[0].employee_id, "EMP-999") == 0, "Retrieved employee_id matches");
    ASSERT_TRUE(strcmp(events[0].status, "PENDING_CONFIRMATION") == 0, "Initial status is PENDING_CONFIRMATION");

    /* Respond with confirmation */
    int resp_rc = attendance_db_respond(db, ev_id, "confirm");
    ASSERT_TRUE(resp_rc == 0, "Responded to event with 'confirm'");

    count = attendance_db_list_events(db, events, 10);
    ASSERT_TRUE(strcmp(events[0].status, "CONFIRMED") == 0, "Status updated to CONFIRMED");
    ASSERT_TRUE(strcmp(events[0].event_type, "CHECK_IN") == 0, "Event type resolved to CHECK_IN");

    attendance_db_close(db);
    remove(test_db_file);
}

static void test_json(void) {
    printf("\n=== TEST 7: cJSON Serialization & Deserialization ===\n");

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "event_type", "CHECK_IN");
    cJSON_AddNumberToObject(root, "similarity", 0.954);
    cJSON_AddBoolToObject(root, "success", 1);

    char *json_str = cJSON_PrintUnformatted(root);
    ASSERT_TRUE(json_str != NULL, "Serialized cJSON object to string");

    cJSON *parsed = cJSON_Parse(json_str);
    ASSERT_TRUE(parsed != NULL, "Parsed JSON string back to cJSON object");
    cJSON *item = cJSON_GetObjectItem(parsed, "event_type");
    ASSERT_TRUE(item && strcmp(item->valuestring, "CHECK_IN") == 0, "Parsed field matches expected value");

    cJSON_Delete(parsed);
    cJSON_Delete(root);
    free(json_str);
}

int main(void) {
    printf("=================================================================\n");
    printf("     RUNNING ALL C UNIT & INTEGRATION TESTS                      \n");
    printf("=================================================================\n");

    test_config();
    test_roi_and_motion_gating();
    test_face_quality_gate();
    test_candidate_lifecycle_and_scheduler();
    test_matcher_and_cooldown();
    test_sqlite_database();
    test_json();

    printf("\n=================================================================\n");
    printf(" TEST SUMMARY: Passed: %d | Failed: %d\n", tests_passed, tests_failed);
    printf("=================================================================\n");

    return (tests_failed == 0) ? 0 : 1;
}
