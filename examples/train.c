#include <onebit/onebit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/**
 * @brief Example application for OneBit model training
 * 
 * This example loads a model and performs training on a
 * text dataset using the OneBit library.
 */

// Training hyperparameters
#define BATCH_SIZE 16
#define LEARNING_RATE 1e-4
#define MAX_EPOCHS 10
#define SEQUENCE_LENGTH 128
#define CHECKPOINT_INTERVAL 100

// Dataset struct
typedef struct {
    char** texts;
    size_t* lengths;
    size_t size;
} Dataset;

// Load dataset from file
Dataset* load_dataset(const char* path, size_t max_samples) {
    FILE* file = fopen(path, "r");
    if (!file) {
        printf("Error opening dataset file: %s\n", path);
        return NULL;
    }
    
    Dataset* dataset = malloc(sizeof(Dataset));
    if (!dataset) {
        fclose(file);
        return NULL;
    }
    
    // Allocate initial capacity
    size_t capacity = 1000;
    dataset->texts = malloc(capacity * sizeof(char*));
    dataset->lengths = malloc(capacity * sizeof(size_t));
    dataset->size = 0;
    
    if (!dataset->texts || !dataset->lengths) {
        free(dataset->texts);
        free(dataset->lengths);
        free(dataset);
        fclose(file);
        return NULL;
    }
    
    // Read lines
    char line[4096];
    while (dataset->size < max_samples && fgets(line, sizeof(line), file)) {
        // Expand capacity if needed
        if (dataset->size >= capacity) {
            capacity *= 2;
            char** new_texts = realloc(dataset->texts, capacity * sizeof(char*));
            size_t* new_lengths = realloc(dataset->lengths, capacity * sizeof(size_t));
            
            if (!new_texts || !new_lengths) {
                for (size_t i = 0; i < dataset->size; i++) {
                    free(dataset->texts[i]);
                }
                free(dataset->texts);
                free(dataset->lengths);
                free(dataset);
                fclose(file);
                return NULL;
            }
            
            dataset->texts = new_texts;
            dataset->lengths = new_lengths;
        }
        
        // Trim newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
            len--;
        }
        
        // Skip empty lines
        if (len == 0) continue;
        
        // Copy the line
        dataset->texts[dataset->size] = strdup(line);
        dataset->lengths[dataset->size] = len;
        dataset->size++;
    }
    
    fclose(file);
    return dataset;
}

// Free dataset
void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    
    for (size_t i = 0; i < dataset->size; i++) {
        free(dataset->texts[i]);
    }
    
    free(dataset->texts);
    free(dataset->lengths);
    free(dataset);
}

// Progress monitoring thread
void* monitor_progress(void* arg) {
    volatile size_t* progress = (volatile size_t*)arg;
    const char spinner[] = {'|', '/', '-', '\\'};
    size_t last_progress = 0;
    size_t spin_idx = 0;
    
    while (1) {
        if (*progress == (size_t)-1) break;  // Signal to exit
        
        if (*progress != last_progress) {
            printf("\rTraining progress: %zu steps %c", *progress, spinner[spin_idx]);
            fflush(stdout);
            last_progress = *progress;
        }
        
        spin_idx = (spin_idx + 1) % 4;
        usleep(100000);  // 100ms
    }
    
    printf("\rTraining completed: %zu steps    \n", last_progress);
    return NULL;
}

int main(int argc, char** argv) {
    // Check arguments
    if (argc < 3) {
        printf("Usage: %s <model_path> <dataset_path> [output_model_path]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* dataset_path = argv[2];
    const char* output_path = (argc > 3) ? argv[3] : "model_trained.bin";
    
    // Initialize OneBit context
    OneBitContext ctx;
    OneBitConfig config = onebit_default_config();
    
    // Configure hardware options
    config.use_cuda = onebit_get_device_count() > 0;
    config.device_id = 0;
    config.num_threads = 8;
    config.use_memory_pool = true;
    config.memory_pool_size = 2 * 1024 * 1024 * 1024;  // 2GB
    
    // Configure training options
    config.attention_dropout = 0.1f;
    config.hidden_dropout = 0.1f;
    config.embedding_dropout = 0.1f;
    
    // Initialize OneBit
    int result = onebit_init(&ctx, &config);
    if (result != ONEBIT_SUCCESS) {
        printf("Error initializing OneBit: %s\n", onebit_error_string(result));
        return 1;
    }
    
    printf("OneBit v%s initialized\n", onebit_version());
    
    // Print hardware capabilities
    printf("Hardware capabilities:\n");
    printf("  AVX-512: %s\n", onebit_has_avx512() ? "Yes" : "No");
    printf("  AVX2   : %s\n", onebit_has_avx2() ? "Yes" : "No");
    printf("  SSE4   : %s\n", onebit_has_sse4() ? "Yes" : "No");
    printf("  GPUs   : %d\n", onebit_get_device_count());
    
    // Load model
    printf("Loading model from %s...\n", model_path);
    result = onebit_load_model(&ctx, model_path);
    if (result != ONEBIT_SUCCESS) {
        printf("Error loading model: %s\n", onebit_error_string(result));
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Load dataset
    printf("Loading dataset from %s...\n", dataset_path);
    Dataset* dataset = load_dataset(dataset_path, 100000);
    if (!dataset) {
        printf("Failed to load dataset\n");
        onebit_cleanup(&ctx);
        return 1;
    }
    
    printf("Dataset loaded with %zu samples\n", dataset->size);
    
    // Calculate total training steps
    size_t steps_per_epoch = (dataset->size + BATCH_SIZE - 1) / BATCH_SIZE;
    size_t total_steps = steps_per_epoch * MAX_EPOCHS;
    printf("Training for %zu steps (%d epochs)\n", total_steps, MAX_EPOCHS);
    
    // Launch progress monitoring thread
    volatile size_t progress = 0;
    pthread_t monitor_thread;
    if (pthread_create(&monitor_thread, NULL, monitor_progress, (void*)&progress) != 0) {
        printf("Failed to create monitoring thread\n");
        free_dataset(dataset);
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Start timing
    clock_t start = clock();
    
    // Perform training (simplified simulation for example)
    size_t step = 0;
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        for (size_t batch = 0; batch < steps_per_epoch; batch++) {
            // In a real implementation, we would:
            // 1. Prepare batch data from dataset
            // 2. Tokenize input texts
            // 3. Run forward pass
            // 4. Compute loss
            // 5. Run backward pass
            // 6. Update weights
            
            // Simulate training step (sleep for a bit)
            usleep(100000);  // 100ms per step
            
            // Update progress
            progress = ++step;
            
            // Save checkpoint at regular intervals
            if (step % CHECKPOINT_INTERVAL == 0) {
                printf("\rSaving checkpoint at step %zu...\n", step);
                // In a real implementation: save model checkpoint
            }
        }
    }
    
    // End timing
    clock_t end = clock();
    float elapsed = (float)(end - start) / CLOCKS_PER_SEC;
    
    // Signal monitor thread to exit
    progress = (size_t)-1;
    pthread_join(monitor_thread, NULL);
    
    printf("Training completed in %.2f seconds\n", elapsed);
    printf("Saving model to %s...\n", output_path);
    
    // In a real implementation: save final model
    
    // Cleanup
    free_dataset(dataset);
    onebit_cleanup(&ctx);
    
    return 0;
} 