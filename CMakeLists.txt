cmake_minimum_required(VERSION 3.16)
project(attendance_recognition_c C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
)

# Core Library Sources
set(CORE_SOURCES
    src/config.c
    src/face_engine.c
    src/matcher.c
    src/vision_utils.c
    src/api_dispatcher.c
    src/gui_preview.c
    src/video_capture.c
    src/third_party/cJSON.c
)

# System Libraries
set(EXTRA_LIBS "")
if (WIN32)
    list(APPEND EXTRA_LIBS winhttp ws2_32 gdi32 user32 ole32)
else()
    list(APPEND EXTRA_LIBS pthread m)
endif()

add_library(attendance_core STATIC ${CORE_SOURCES})
target_link_libraries(attendance_core ${EXTRA_LIBS})

# Main Executables
add_executable(run_camera src/run_camera.c)
target_link_libraries(run_camera attendance_core)

add_executable(enroll_employee src/enroll_employee.c)
target_link_libraries(enroll_employee attendance_core)

add_executable(export_embeddings src/export_embeddings.c)
target_link_libraries(export_embeddings attendance_core)

# Test Suite
add_executable(test_all tests/test_all.c)
target_link_libraries(test_all attendance_core)

enable_testing()
add_test(NAME UnitTests COMMAND test_all)
