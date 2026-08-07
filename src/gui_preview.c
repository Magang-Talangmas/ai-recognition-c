#include "attendance/gui_preview.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define CANONICAL_CANVAS_W 1280
#define CANONICAL_CANVAS_H 720

#ifdef _WIN32
#include <windows.h>

struct GuiWindow {
    char title[256];
    int width;
    int height;
    HWND hwnd;
    HDC hdc_mem;
    HBITMAP hbm_mem;
    HBITMAP hbm_old;
    HANDLE h_shm_map;
    uint8_t *shm_ptr;
    bool is_open;
};

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_ERASEBKGND:
            return 1; /* Avoid flicker during resize/fullscreen */
        case WM_GETMINMAXINFO: {
            LPMINMAXINFO lpMMI = (LPMINMAXINFO)lParam;
            lpMMI->ptMinTrackSize.x = 640;
            lpMMI->ptMinTrackSize.y = 360;
            return 0;
        }
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE || wParam == 'Q' || wParam == 'q') {
                PostQuitMessage(0);
            }
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}
#else
struct GuiWindow {
    char title[256];
    int width;
    int height;
    bool is_open;
};
#endif

GuiWindow *gui_window_create(const char *title, int width, int height) {
    GuiWindow *win = (GuiWindow *)calloc(1, sizeof(GuiWindow));
    if (!win) return NULL;

    strncpy(win->title, title ? title : "Talangmas AI Attendance - Live View", sizeof(win->title) - 1);
    win->width = (width > 0) ? width : CANONICAL_CANVAS_W;
    win->height = (height > 0) ? height : CANONICAL_CANVAS_H;

#ifdef _WIN32
    HINSTANCE hInst = GetModuleHandle(NULL);

    WNDCLASSA wc;
    memset(&wc, 0, sizeof(wc));
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = "TalangmasPreviewWindowClass";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassA(&wc);

    RECT wr = {0, 0, win->width, win->height};
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

    win->hwnd = CreateWindowExA(
        0,
        wc.lpszClassName,
        win->title,
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left, wr.bottom - wr.top,
        NULL, NULL, hInst, NULL
    );

    if (!win->hwnd) {
        free(win);
        return NULL;
    }

    /* Fixed canonical canvas size (1280x720) for consistent rendering & web stream */
    HDC hdc_screen = GetDC(win->hwnd);
    win->hdc_mem = CreateCompatibleDC(hdc_screen);
    win->hbm_mem = CreateCompatibleBitmap(hdc_screen, CANONICAL_CANVAS_W, CANONICAL_CANVAS_H);
    win->hbm_old = (HBITMAP)SelectObject(win->hdc_mem, win->hbm_mem);
    ReleaseDC(win->hwnd, hdc_screen);

    /* Setup Shared Memory for zero-latency Web/Mobile Live Streaming */
    win->h_shm_map = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        16 * 1024 * 1024,
        "Local\\TMAS_PREVIEW_SHM"
    );
    if (win->h_shm_map) {
        win->shm_ptr = (uint8_t *)MapViewOfFile(
            win->h_shm_map,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            16 * 1024 * 1024
        );
    }

    win->is_open = true;
#endif

    return win;
}

void gui_window_destroy(GuiWindow *win) {
    if (!win) return;
#ifdef _WIN32
    if (win->shm_ptr) {
        UnmapViewOfFile(win->shm_ptr);
        win->shm_ptr = NULL;
    }
    if (win->h_shm_map) {
        CloseHandle(win->h_shm_map);
        win->h_shm_map = NULL;
    }

    if (win->hdc_mem) {
        SelectObject(win->hdc_mem, win->hbm_old);
        DeleteDC(win->hdc_mem);
    }
    if (win->hbm_mem) DeleteObject(win->hbm_mem);

    if (win->hwnd) {
        DestroyWindow(win->hwnd);
    }
#endif
    free(win);
}

bool gui_window_process_events(GuiWindow *win) {
    if (!win) return false;
#ifdef _WIN32
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            win->is_open = false;
            return false;
        }
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return win->is_open;
#else
    return true;
#endif
}

void gui_window_render(
    GuiWindow *win,
    const ImageBuffer *frame,
    const FaceResult *faces,
    int num_faces,
    float display_fps,
    float inference_ms,
    const char *last_event_msg
) {
    (void)display_fps;
    (void)inference_ms;
    (void)last_event_msg;
    if (!win) return;

#ifdef _WIN32
    if (!win->hwnd || !win->hdc_mem) return;

    HDC hdc = win->hdc_mem;
    int cam_viewport_w = CANONICAL_CANVAS_W; /* 1280px Fullscreen Viewport */
    int cam_viewport_h = CANONICAL_CANVAS_H; /* 720px Fullscreen Viewport */

    /* 1. Paint Camera Viewport Background */
    RECT cam_vp_rect = {0, 0, cam_viewport_w, cam_viewport_h};
    HBRUSH cam_bg_brush = CreateSolidBrush(RGB(15, 18, 24));
    FillRect(hdc, &cam_vp_rect, cam_bg_brush);
    DeleteObject(cam_bg_brush);

    /* 1.1 Calculate Aspect-Ratio-Preserving Coordinates for Fullscreen Camera Frame */
    int cam_draw_w = cam_viewport_w;
    int cam_draw_h = cam_viewport_h;
    int cam_offset_x = 0;
    int cam_offset_y = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;

    if (frame && frame->data && frame->width > 0 && frame->height > 0) {
        float aspect_src = (float)frame->width / (float)frame->height;
        float aspect_dst = (float)cam_viewport_w / (float)cam_viewport_h;

        if (aspect_src > aspect_dst) {
            cam_draw_w = cam_viewport_w;
            cam_draw_h = (int)((float)cam_viewport_w / aspect_src);
            cam_offset_x = 0;
            cam_offset_y = (cam_viewport_h - cam_draw_h) / 2;
        } else {
            cam_draw_h = cam_viewport_h;
            cam_draw_w = (int)((float)cam_viewport_h * aspect_src);
            cam_offset_x = (cam_viewport_w - cam_draw_w) / 2;
            cam_offset_y = 0;
        }

        scale_x = (float)cam_draw_w / (float)frame->width;
        scale_y = (float)cam_draw_h / (float)frame->height;

        BITMAPINFO bmi;
        memset(&bmi, 0, sizeof(bmi));
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = frame->width;
        bmi.bmiHeader.biHeight = -frame->height; /* Top-down BGR */
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;

        SetStretchBltMode(hdc, COLORONCOLOR);
        StretchDIBits(
            hdc,
            cam_offset_x, cam_offset_y, cam_draw_w, cam_draw_h,
            0, 0, frame->width, frame->height,
            frame->data,
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY
        );
    }

    /* 2. Render ONLY Clean Bounding Box (No HUD, No yellow dots, No sidebar) */
    if (faces && num_faces > 0) {
        COLORREF box_color = RGB(0, 255, 128); /* Vibrant Clean Green */
        HPEN box_pen = CreatePen(PS_SOLID, 2, box_color);
        HPEN old_pen = (HPEN)SelectObject(hdc, box_pen);
        HBRUSH null_brush = (HBRUSH)GetStockObject(NULL_BRUSH);
        HBRUSH old_brush = (HBRUSH)SelectObject(hdc, null_brush);

        for (int i = 0; i < num_faces; i++) {
            const FaceResult *face = &faces[i];
            int bx1 = cam_offset_x + (int)(face->bbox.x1 * scale_x);
            int by1 = cam_offset_y + (int)(face->bbox.y1 * scale_y);
            int bx2 = cam_offset_x + (int)(face->bbox.x2 * scale_x);
            int by2 = cam_offset_y + (int)(face->bbox.y2 * scale_y);

            if (bx2 <= bx1 || by2 <= by1) continue;

            /* Draw clean green bounding box */
            Rectangle(hdc, bx1, by1, bx2, by2);
        }

        SelectObject(hdc, old_brush);
        SelectObject(hdc, old_pen);
        DeleteObject(box_pen);
    }

    /* 3. Export Fixed Canonical 1280x720 Frame to Shared Memory for Web & Mobile (Throttled) */
    static int shm_counter = 0;
    if (win->shm_ptr && (++shm_counter % 2 == 0)) {
        BITMAPINFO shm_bmi;
        memset(&shm_bmi, 0, sizeof(shm_bmi));
        shm_bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        shm_bmi.bmiHeader.biWidth = CANONICAL_CANVAS_W;
        shm_bmi.bmiHeader.biHeight = -CANONICAL_CANVAS_H; /* Top-down BGR */
        shm_bmi.bmiHeader.biPlanes = 1;
        shm_bmi.bmiHeader.biBitCount = 24;
        shm_bmi.bmiHeader.biCompression = BI_RGB;

        uint8_t *pixels = win->shm_ptr + 24; /* 24 bytes header offset */
        GetDIBits(win->hdc_mem, win->hbm_mem, 0, CANONICAL_CANVAS_H, pixels, &shm_bmi, DIB_RGB_COLORS);

        /* Write 24-byte header: uint32 magic (TMAS), uint32 width, uint32 height, uint32 channels, uint64 seq */
        uint32_t *hdr32 = (uint32_t *)win->shm_ptr;
        uint64_t *hdr64 = (uint64_t *)(win->shm_ptr + 16);
        hdr32[1] = (uint32_t)CANONICAL_CANVAS_W;
        hdr32[2] = (uint32_t)CANONICAL_CANVAS_H;
        hdr32[3] = 3;
        (*hdr64)++;
        hdr32[0] = 0x53414D54; /* 'TMAS' magic */
    }

    /* 4. Letterbox / Pillarbox Scale to Desktop Window Client Area */
    RECT client_rc;
    GetClientRect(win->hwnd, &client_rc);
    int win_w = client_rc.right - client_rc.left;
    int win_h = client_rc.bottom - client_rc.top;

    if (win_w > 0 && win_h > 0) {
        float scale_fit = ((float)win_w / (float)CANONICAL_CANVAS_W < (float)win_h / (float)CANONICAL_CANVAS_H) ?
                          ((float)win_w / (float)CANONICAL_CANVAS_W) :
                          ((float)win_h / (float)CANONICAL_CANVAS_H);

        int dest_w = (int)(CANONICAL_CANVAS_W * scale_fit);
        int dest_h = (int)(CANONICAL_CANVAS_H * scale_fit);
        int dest_x = (win_w - dest_w) / 2;
        int dest_y = (win_h - dest_h) / 2;

        HDC hdc_screen = GetDC(win->hwnd);

        /* Clear black letterbox bars if needed */
        if (dest_x > 0 || dest_y > 0) {
            HBRUSH black_br = (HBRUSH)GetStockObject(BLACK_BRUSH);
            if (dest_x > 0) {
                RECT r_left = {0, 0, dest_x, win_h};
                RECT r_right = {dest_x + dest_w, 0, win_w, win_h};
                FillRect(hdc_screen, &r_left, black_br);
                FillRect(hdc_screen, &r_right, black_br);
            }
            if (dest_y > 0) {
                RECT r_top = {0, 0, win_w, dest_y};
                RECT r_bot = {0, dest_y + dest_h, win_w, win_h};
                FillRect(hdc_screen, &r_top, black_br);
                FillRect(hdc_screen, &r_bot, black_br);
            }
        }

        SetStretchBltMode(hdc_screen, COLORONCOLOR);
        StretchBlt(
            hdc_screen,
            dest_x, dest_y, dest_w, dest_h,
            win->hdc_mem,
            0, 0, CANONICAL_CANVAS_W, CANONICAL_CANVAS_H,
            SRCCOPY
        );

        ReleaseDC(win->hwnd, hdc_screen);
    }
#endif
}
