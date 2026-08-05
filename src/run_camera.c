#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "attendance/config.h"
#include "attendance/database.h"
#include "attendance/face_engine.h"
#include "attendance/matcher.h"
#include "attendance/vision_utils.h"
#include "attendance/api_dispatcher.h"

#ifdef HAVE_OPENCV
#include <opencv2/c/opencv.h>
#endif

#include <signal.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

static volatile bool g_running = true;

static void handle_sigint(int sig) {
    (void)sig;
    g_running = false;
}

int main(int argc, char **argv) {
    signal(SIGINT, handle_sigint);
#ifdef SIGTERM
    signal(SIGTERM, handle_sigint);
#endif

    const char *config_file = "config.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }

    printf("=================================================================\n");
    printf("   TALANGMAS AI-RECOGNITION ATTENDANCE ENGINE (C EDITION)        \n");
    printf("   Featuring: ROI Gating | Motion Gating | Round-Robin ArcFace   \n");
    printf("=================================================================\n");

    AppConfig config;
    if (config_load(config_file, &config) != 0) {
        fprintf(stderr, "[Warning] Could not load '%s', using internal defaults.\n", config_file);
    }
    config_print(&config);

    /* 1. Open Local Event Database (SQLite WAL mode) */
    AttendanceDB *db = attendance_db_open(
        config.attendance.event_database,
        config.attendance.duplicate_cooldown_seconds
    );
    if (!db) {
        fprintf(stderr, "[Error] Failed to initialize SQLite database '%s'\n", config.attendance.event_database);
        return 1;
    }
    printf("[Init] Local SQLite database connected: %s\n", config.attendance.event_database);

    /* 2. Load Enrolled Face Templates */
    FaceMatcher *matcher = face_matcher_create(config.attendance.embedding_file, &config.face);
    if (!matcher) {
        fprintf(stderr, "[Error] Failed to initialize FaceMatcher.\n");
        attendance_db_close(db);
        return 1;
    }

    /* 3. Initialize Face Detection & Recognition Engine */
    FaceEngine *face_engine = face_engine_create(&config.face);
    if (!face_engine) {
        fprintf(stderr, "[Error] Failed to initialize FaceEngine.\n");
        face_matcher_destroy(matcher);
        attendance_db_close(db);
        return 1;
    }

    /* 4. Initialize Asynchronous Backend Dispatcher */
    BackendDispatcher *dispatcher = NULL;
    if (config.backend.enabled) {
        dispatcher = backend_dispatcher_create(&config.backend);
        if (dispatcher) {
            printf("[Init] Async HTTP Backend Dispatcher active: %s\n", config.backend.api_url);
        }
    }

    /* 5. Initialize Candidate Lifecycle Table */
    CandidateTable candidate_table;
    candidate_table_init(&candidate_table, 150.0f);

    printf("\n[System Ready] Starting video capture stream from: %s\n", config.camera.source);
    printf("Press Ctrl+C to stop.\n\n");

    int frame_index = 0;
    double last_fps_time = get_monotonic_time_seconds();
    int fps_frame_count = 0;

    /* Main Video Processing Loop */
    while (g_running) {
        double now = get_monotonic_time_seconds();
        frame_index++;
        fps_frame_count++;

        if (now - last_fps_time >= 2.0) {
            double current_fps = (double)fps_frame_count / (now - last_fps_time);
            fps_frame_count = 0;
            last_fps_time = now;
            printf("[Stream Active] FPS: %.1f | Frame: %d | Candidates in Zone: %d\n",
                   current_fps, frame_index, candidate_table.count);
            fflush(stdout);
        }

        /* Periodically clean stale candidates */
        if (frame_index % 60 == 0) {
            candidate_table_clean_stale(&candidate_table, now, config.scheduler.candidate_ttl_seconds);
        }

        /* Frame skipping logic */
        if (frame_index % config.camera.process_every_n_frames != 0) {
#ifdef _WIN32
            Sleep(15);
#else
            usleep(15000);
#endif
            continue;
        }

        /*
         * Pipeline Execution:
         * Step 1: Capture frame buffer (or dummy simulation buffer)
         * Step 2: Recognition Zone (ROI) Bounding Box calculation
         * Step 3: Motion / Activity Gating check
         * Step 4: SCRFD Face Detection on ROI sub-frame
         * Step 5: Face Quality Gate (blur, size, yaw angle)
         * Step 6: Candidate Table Update
         * Step 7: Priority Round-Robin ArcFace Scheduler
         * Step 8: Temporal Majority Voting (3 of 5)
         * Step 9: In-Memory Cooldown & Duplicate Check
         * Step 10: Event Generation (CHECK_IN_PENDING) -> SQLite + Async Backend Dispatch
         */

        ImageBuffer dummy_frame = {0};
        FaceBBox roi_box = calculate_roi_bbox(dummy_frame.width, dummy_frame.height, &config.roi);

        FaceResult faces[MAX_DETECTED_FACES];
        int num_faces = face_engine_detect(face_engine, &dummy_frame, faces, MAX_DETECTED_FACES);

        for (int i = 0; i < num_faces; i++) {
            FaceResult *face = &faces[i];

            /* Filter 1: Check if face is inside Recognition Zone (ROI) */
            if (config.roi.enabled && !is_bbox_in_roi(&face->bbox, &roi_box, 0.40f)) {
                continue;
            }

            /* Filter 2: Quality Gate */
            char quality_reason[64] = {0};
            if (!face_quality_gate_check(face, &config.face, quality_reason, sizeof(quality_reason))) {
                continue;
            }

            /* Update candidate tracking */
            candidate_table_update_face(&candidate_table, face, now);
        }

        /* Priority Round-Robin ArcFace Scheduler: allocate compute budget */
        int64_t candidate_budget_ids[MAX_TRACKED_PEOPLE];
        int selected_count = candidate_table_select_for_arcface(
            &candidate_table,
            config.scheduler.max_arcface_per_frame,
            candidate_budget_ids,
            MAX_TRACKED_PEOPLE
        );

        for (int s = 0; s < selected_count; s++) {
            int64_t track_id = candidate_budget_ids[s];

            /* Find candidate in table */
            for (int t = 0; t < candidate_table.count; t++) {
                TrackedCandidate *tc = &candidate_table.candidates[t];
                if (tc->track_id == track_id) {
                    tc->last_arcface_time = now;
                    tc->arcface_eval_count++;

                    /* Align face & Extract ArcFace embedding */
                    uint8_t crop_data[112 * 112 * 3];
                    ImageBuffer aligned_crop = {
                        .data = crop_data,
                        .width = 112,
                        .height = 112,
                        .channels = 3,
                        .stride = 112 * 3
                    };

                    float embedding[FACE_EMBEDDING_DIM];
                    if (face_engine_align_face(&dummy_frame, &tc->last_landmarks, &aligned_crop) == 0 &&
                        face_engine_extract_embedding(face_engine, &aligned_crop, embedding) == 0) {
                        
                        MatchResult match = face_matcher_match(matcher, embedding);

                        track_vote_push(
                            &tc->vote_queue,
                            match.is_matched ? match.employee_id : "",
                            match.score,
                            config.face.vote_window
                        );

                        /* Check 3 of 5 Majority Vote */
                        char stable_id[128] = {0};
                        float stable_score = 0.0f;
                        if (track_vote_get_stable(&tc->vote_queue,
                                                 config.face.votes_required,
                                                 stable_id, sizeof(stable_id),
                                                 &stable_score)) {
                            
                            if (!tc->is_confirmed) {
                                tc->is_confirmed = true;
                                tc->state = CANDIDATE_STATE_CONFIRMED;
                                strncpy(tc->confirmed_id, stable_id, sizeof(tc->confirmed_id) - 1);
                                tc->confirmed_score = stable_score;

                                /* Check cooldown to avoid redundant spam */
                                if (!face_matcher_is_in_cooldown(matcher, stable_id, now, (double)config.attendance.duplicate_cooldown_seconds)) {
                                    face_matcher_record_cooldown(matcher, stable_id, now);

                                    float line_y = (float)dummy_frame.height * config.attendance.line_y_ratio;
                                    int cur_side = (tc->cy >= line_y) ? 1 : -1;
                                    const char *dir = crossing_direction(tc->previous_side, cur_side, config.attendance.inside_is_below_line);
                                    if (!dir) dir = "ENTER";

                                    int64_t evt_id = attendance_db_create_pending_event(
                                        db,
                                        config.camera.camera_id,
                                        track_id,
                                        stable_id,
                                        dir,
                                        "PENDING_CONFIRMATION",
                                        stable_score
                                    );

                                    if (evt_id > 0) {
                                        printf("\n[EVENT CREATED] ID=%lld | Track=%lld | Employee=%s | Score=%.3f | Dir=%s\n",
                                               (long long)evt_id, (long long)track_id, stable_id, stable_score, dir);

                                        if (dispatcher) {
                                            char evt_str[64];
                                            snprintf(evt_str, sizeof(evt_str), "%lld", (long long)evt_id);
                                            backend_dispatcher_dispatch_checkin(
                                                dispatcher,
                                                stable_id,
                                                stable_score,
                                                config.camera.camera_id,
                                                evt_str,
                                                NULL
                                            );
                                        }
                                    }
                                    tc->previous_side = cur_side;
                                    tc->state = CANDIDATE_STATE_COOLDOWN;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }

#ifdef _WIN32
        Sleep(20);
#else
        usleep(20000);
#endif
    }

    printf("\n[Shutdown] Cleaning up resources...\n");
    if (dispatcher) backend_dispatcher_destroy(dispatcher);
    face_engine_destroy(face_engine);
    face_matcher_destroy(matcher);
    attendance_db_close(db);

    printf("[Shutdown] AI-Recognition engine exited cleanly.\n");
    return 0;
}
