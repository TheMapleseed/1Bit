#include "../test_utils.h"
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <errno.h>

// Deadlock detection timeout (in seconds)
#define DEADLOCK_TIMEOUT 5

// Thread safety monitoring
typedef struct {
    pthread_mutex_t mutex;
    pthread_t owner;
    struct timespec lock_time;
    const char* lock_file;
    int lock_line;
    bool is_locked;
} SafeMutex;

// Global mutex tracking
#define MAX_MUTEXES 100
static SafeMutex g_mutex_tracking[MAX_MUTEXES];
static size_t g_num_mutexes = 0;
static pthread_mutex_t g_tracking_mutex = PTHREAD_MUTEX_INITIALIZER;

// Initialize safe mutex
void safe_mutex_init(SafeMutex* mutex) {
    pthread_mutex_init(&mutex->mutex, NULL);
    mutex->owner = 0;
    mutex->is_locked = false;
    
    pthread_mutex_lock(&g_tracking_mutex);
    if (g_num_mutexes < MAX_MUTEXES) {
        g_mutex_tracking[g_num_mutexes++] = *mutex;
    }
    pthread_mutex_unlock(&g_tracking_mutex);
}

// Safe mutex lock with deadlock detection
int safe_mutex_lock(SafeMutex* mutex, const char* file, int line) {
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    while (1) {
        int result = pthread_mutex_trylock(&mutex->mutex);
        if (result == 0) {
            mutex->owner = pthread_self();
            mutex->lock_time = start_time;
            mutex->lock_file = file;
            mutex->lock_line = line;
            mutex->is_locked = true;
            return 0;
        }
        
        // Check for timeout (potential deadlock)
        struct timespec current_time;
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        double elapsed = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
                        
        if (elapsed > DEADLOCK_TIMEOUT) {
            fprintf(stderr, "Potential deadlock detected at %s:%d\n", file, line);
            fprintf(stderr, "Mutex owned by thread %lu at %s:%d\n",
                    (unsigned long)mutex->owner,
                    mutex->lock_file,
                    mutex->lock_line);
            return EDEADLK;
        }
        
        usleep(1000);  // Small sleep to prevent CPU spinning
    }
}

// Safe mutex unlock
int safe_mutex_unlock(SafeMutex* mutex) {
    if (mutex->owner != pthread_self()) {
        fprintf(stderr, "Attempt to unlock mutex owned by different thread\n");
        return EPERM;
    }
    
    mutex->owner = 0;
    mutex->is_locked = false;
    return pthread_mutex_unlock(&mutex->mutex);
}

// Race condition detection
typedef struct {
    void* address;
    pthread_t thread;
    bool is_write;
    const char* file;
    int line;
} MemoryAccess;

#define MAX_MEMORY_ACCESSES 10000
static MemoryAccess g_memory_accesses[MAX_MEMORY_ACCESSES];
static size_t g_num_accesses = 0;
static SafeMutex g_access_mutex;

// Track memory access
void track_memory_access(void* addr, bool is_write, const char* file, int line) {
    safe_mutex_lock(&g_access_mutex, file, line);
    
    if (g_num_accesses < MAX_MEMORY_ACCESSES) {
        g_memory_accesses[g_num_accesses++] = (MemoryAccess){
            .address = addr,
            .thread = pthread_self(),
            .is_write = is_write,
            .file = file,
            .line = line
        };
        
        // Check for race conditions
        for (size_t i = 0; i < g_num_accesses - 1; i++) {
            if (g_memory_accesses[i].address == addr &&
                g_memory_accesses[i].thread != pthread_self() &&
                (is_write || g_memory_accesses[i].is_write)) {
                fprintf(stderr, "Potential race condition detected!\n");
                fprintf(stderr, "Current access: %s:%d\n", file, line);
                fprintf(stderr, "Conflicting access: %s:%d\n",
                        g_memory_accesses[i].file,
                        g_memory_accesses[i].line);
            }
        }
    }
    
    safe_mutex_unlock(&g_access_mutex);
}

// Segmentation fault handler
static void segfault_handler(int sig, siginfo_t* si, void* unused) {
    fprintf(stderr, "Segmentation fault at address: %p\n", si->si_addr);
    
    // Print recent memory accesses
    fprintf(stderr, "\nRecent memory accesses:\n");
    for (int i = (int)g_num_accesses - 1; i >= 0 && i >= (int)g_num_accesses - 10; i--) {
        fprintf(stderr, "%s at %s:%d by thread %lu\n",
                g_memory_accesses[i].is_write ? "Write" : "Read",
                g_memory_accesses[i].file,
                g_memory_accesses[i].line,
                (unsigned long)g_memory_accesses[i].thread);
    }
    
    // Print stack trace
    void* buffer[100];
    int nptrs = backtrace(buffer, 100);
    backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);
    
    exit(1);
}

// Initialize memory safety monitoring
void init_memory_safety(void) {
    // Initialize mutex tracking
    safe_mutex_init(&g_access_mutex);
    
    // Set up segfault handler
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
}

// Memory boundary checking
void* checked_malloc(size_t size, const char* file, int line) {
    // Add guard pages
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t total_size = size + 2 * page_size;
    
    void* base = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;
    
    // Protect guard pages
    mprotect(base, page_size, PROT_NONE);
    mprotect(base + page_size + size, page_size, PROT_NONE);
    
    void* ptr = base + page_size;
    track_memory_access(ptr, true, file, line);
    return ptr;
}

// Safe memory free
void checked_free(void* ptr, const char* file, int line) {
    if (!ptr) return;
    
    track_memory_access(ptr, true, file, line);
    
    size_t page_size = sysconf(_SC_PAGESIZE);
    void* base = ptr - page_size;
    
    // Get original allocation size from tracking
    size_t size = 0;  // You need to track allocation sizes
    munmap(base, size + 2 * page_size);
} 