#include "onebit/onebit_hotreload.h"
#include "onebit/onebit_error.h"
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif

typedef struct {
    char path[256];
    time_t last_modified;
} WatchedFile;

typedef struct {
    WatchedFile* files;
    size_t num_files;
    size_t capacity;
} FileRegistry;

static FileRegistry* create_file_registry(size_t initial_capacity) {
    FileRegistry* registry = malloc(sizeof(FileRegistry));
    if (!registry) return NULL;
    
    registry->files = malloc(initial_capacity * sizeof(WatchedFile));
    if (!registry->files) {
        free(registry);
        return NULL;
    }
    
    registry->num_files = 0;
    registry->capacity = initial_capacity;
    return registry;
}

static void scan_directory(const char* dir_path, FileRegistry* registry,
                          const char** patterns, int num_patterns) {
    #ifdef _WIN32
    WIN32_FIND_DATA find_data;
    HANDLE find_handle;
    char search_path[256];
    snprintf(search_path, sizeof(search_path), "%s\\*", dir_path);
    
    find_handle = FindFirstFile(search_path, &find_data);
    if (find_handle == INVALID_HANDLE_VALUE) return;
    
    do {
        if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            // Check if file matches any pattern
            for (int i = 0; i < num_patterns; i++) {
                if (strstr(find_data.cFileName, patterns[i])) {
                    WatchedFile* file = &registry->files[registry->num_files++];
                    snprintf(file->path, sizeof(file->path), "%s\\%s",
                            dir_path, find_data.cFileName);
                    file->last_modified = 0;  // Will be updated on first check
                    break;
                }
            }
        }
    } while (FindNextFile(find_handle, &find_data));
    
    FindClose(find_handle);
    #else
    DIR* dir = opendir(dir_path);
    if (!dir) return;
    
    struct dirent* entry;
    while ((entry = readdir(dir))) {
        if (entry->d_type == DT_REG) {
            for (int i = 0; i < num_patterns; i++) {
                if (strstr(entry->d_name, patterns[i])) {
                    WatchedFile* file = &registry->files[registry->num_files++];
                    snprintf(file->path, sizeof(file->path), "%s/%s",
                            dir_path, entry->d_name);
                    file->last_modified = 0;
                    break;
                }
            }
        }
    }
    
    closedir(dir);
    #endif
}

static void* monitor_thread(void* arg) {
    HotReloadContext* ctx = (HotReloadContext*)arg;
    FileRegistry* registry = ctx->file_monitor;
    
    while (ctx->is_running) {
        bool changes_detected = false;
        
        for (size_t i = 0; i < registry->num_files; i++) {
            struct stat st;
            if (stat(registry->files[i].path, &st) == 0) {
                if (st.st_mtime > registry->files[i].last_modified) {
                    registry->files[i].last_modified = st.st_mtime;
                    if (ctx->config.reload_callback) {
                        ctx->config.reload_callback(registry->files[i].path,
                                                 ctx->config.callback_ctx);
                    }
                    changes_detected = true;
                }
            }
        }
        
        if (!changes_detected) {
            usleep(ctx->config.poll_interval_ms * 1000);
        }
    }
    
    return NULL;
}

int hotreload_init(HotReloadContext* ctx, const HotReloadConfig* config) {
    if (!ctx || !config) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    memcpy(&ctx->config, config, sizeof(HotReloadConfig));
    ctx->file_monitor = create_file_registry(config->max_files);
    if (!ctx->file_monitor) {
        return ONEBIT_ERROR_OUT_OF_MEMORY;
    }
    
    scan_directory(config->root_dir, ctx->file_monitor, config->patterns, config->num_patterns);
    
    ctx->monitor_thread = create_thread(monitor_thread, ctx);
    if (!ctx->monitor_thread) {
        return ONEBIT_ERROR_OUT_OF_MEMORY;
    }
    
    return ONEBIT_SUCCESS;
} 