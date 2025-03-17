/**
 * @file online_learning_demo.c
 * @brief Example demonstrating standard training and online learning with the OneBit library
 */

#include <onebit/onebit.h>
#include <onebit/onebit_model.h>
#include <onebit/onebit_train.h>
#include <onebit/onebit_inference.h>
#include <onebit/onebit_data.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Demo data generation
void generate_training_data(float* inputs, float* outputs, size_t n_samples, 
                           size_t input_size, size_t output_size) {
    // Simple classification task: output[0] = 1 if sum(inputs) > 0.5 * input_size, else 0
    for (size_t i = 0; i < n_samples; i++) {
        float* input = inputs + i * input_size;
        float* output = outputs + i * output_size;
        
        // Generate random inputs between 0 and 1
        float sum = 0.0f;
        for (size_t j = 0; j < input_size; j++) {
            input[j] = (float)rand() / RAND_MAX;
            sum += input[j];
        }
        
        // Binary classification
        memset(output, 0, output_size * sizeof(float));
        output[0] = (sum > 0.5f * input_size) ? 1.0f : 0.0f;
    }
}

// Standard training example
void run_training_example() {
    printf("\n=== Standard Training Example ===\n");
    
    // Initialize OneBit
    OneBitContext ctx;
    OneBitConfig config = {0};
    config.device_type = DEVICE_CPU;
    config.num_threads = 4;
    config.seed = time(NULL);
    config.log_level = LOG_INFO;
    config.batch_size = 32;
    config.max_epochs = 10;
    
    int result = onebit_init(&ctx, &config);
    if (result != ONEBIT_SUCCESS) {
        printf("Failed to initialize OneBit: %d\n", result);
        return;
    }
    
    // Create a simple model (binary classifier)
    const size_t input_size = 10;
    const size_t hidden_size = 32;
    const size_t output_size = 1;
    
    // Create model
    result = model_create(&ctx, input_size, hidden_size, output_size);
    if (result != ONEBIT_SUCCESS) {
        printf("Failed to create model: %d\n", result);
        onebit_cleanup(&ctx);
        return;
    }
    
    // Generate synthetic training data
    const size_t train_samples = 1000;
    const size_t val_samples = 200;
    
    float* train_inputs = (float*)malloc(train_samples * input_size * sizeof(float));
    float* train_outputs = (float*)malloc(train_samples * output_size * sizeof(float));
    float* val_inputs = (float*)malloc(val_samples * input_size * sizeof(float));
    float* val_outputs = (float*)malloc(val_samples * output_size * sizeof(float));
    
    if (!train_inputs || !train_outputs || !val_inputs || !val_outputs) {
        printf("Memory allocation failed\n");
        free(train_inputs);
        free(train_outputs);
        free(val_inputs);
        free(val_outputs);
        onebit_cleanup(&ctx);
        return;
    }
    
    // Generate data
    generate_training_data(train_inputs, train_outputs, train_samples, input_size, output_size);
    generate_training_data(val_inputs, val_outputs, val_samples, input_size, output_size);
    
    // Create datasets
    Dataset* train_dataset = dataset_create_from_memory(&ctx, train_inputs, train_outputs, 
                                                      train_samples, input_size, output_size);
    Dataset* val_dataset = dataset_create_from_memory(&ctx, val_inputs, val_outputs, 
                                                    val_samples, input_size, output_size);
    
    if (!train_dataset || !val_dataset) {
        printf("Failed to create datasets\n");
        free(train_inputs);
        free(train_outputs);
        free(val_inputs);
        free(val_outputs);
        onebit_cleanup(&ctx);
        return;
    }
    
    // Configure training
    TrainConfig train_config = {0};
    train_config.learning_rate = 0.01f;
    train_config.weight_decay = 0.0001f;
    train_config.grad_clip = 1.0f;
    train_config.batch_size = 32;
    train_config.max_epochs = 10;
    train_config.print_frequency = 10;
    train_config.checkpoint_interval = 2;
    train_config.checkpoint_dir = "./checkpoints";
    
    // Initialize training state
    TrainState* train_state = NULL;
    result = onebit_train_init(&ctx, &train_config, &train_state);
    if (result != ONEBIT_SUCCESS || !train_state) {
        printf("Failed to initialize training: %d\n", result);
        dataset_destroy(train_dataset);
        dataset_destroy(val_dataset);
        free(train_inputs);
        free(train_outputs);
        free(val_inputs);
        free(val_outputs);
        onebit_cleanup(&ctx);
        return;
    }
    
    // Run training
    printf("Starting training...\n");
    result = onebit_train(&ctx, train_state, train_dataset, val_dataset);
    if (result != ONEBIT_SUCCESS) {
        printf("Training failed: %d\n", result);
    } else {
        printf("Training completed successfully.\n");
        printf("Best validation loss: %f\n", train_state->best_loss);
    }
    
    // Clean up training resources
    onebit_train_cleanup(train_state);
    dataset_destroy(train_dataset);
    dataset_destroy(val_dataset);
    free(train_inputs);
    free(train_outputs);
    free(val_inputs);
    free(val_outputs);
    
    // Verify the model
    printf("\nModel verification:\n");
    float test_input[input_size];
    float test_output[output_size];
    
    // Generate a test input with known output (sum > 0.5 * input_size)
    float sum = 0.0f;
    for (size_t i = 0; i < input_size; i++) {
        test_input[i] = 0.8f;  // All high values should give output 1
        sum += test_input[i];
    }
    
    // Run inference
    result = model_forward(ctx.model, test_input, test_output, 1, 1);
    if (result != ONEBIT_SUCCESS) {
        printf("Inference failed: %d\n", result);
    } else {
        printf("Test input (sum=%f) -> output: %f (expected: 1.0)\n", 
               sum, test_output[0]);
    }
    
    // Don't clean up ctx yet - we'll use it for the online learning example
}

// Online learning example
void run_online_learning_example(OneBitContext* ctx) {
    printf("\n=== Online Learning Example ===\n");
    
    if (!ctx || !ctx->model) {
        printf("Invalid context or model\n");
        return;
    }
    
    // Configure inference with online learning
    InferenceConfig inf_config = {0};
    inf_config.temperature = 1.0f;
    inf_config.top_p = 1.0f;
    inf_config.top_k = 0;
    inf_config.max_seq_len = 100;
    inf_config.batch_size = 1;
    inf_config.use_kv_cache = true;
    inf_config.num_threads = 4;
    
    // Enable online learning
    inf_config.enable_online_learning = true;
    inf_config.online_learning_config.enabled = true;
    inf_config.online_learning_config.learning_rate = 0.001f;  // Smaller than training
    inf_config.online_learning_config.update_frequency = 1;    // Update every inference
    inf_config.online_learning_config.forgetting_factor = 0.9f;
    inf_config.online_learning_config.cache_examples = true;
    inf_config.online_learning_config.cache_size = 10;
    inf_config.online_learning_config.update_during_inference = true;
    
    // Create inference context
    InferenceContext* inf_ctx = NULL;
    int result = inference_init(&inf_ctx, ctx->model, NULL, &inf_config);
    if (result != ONEBIT_SUCCESS || !inf_ctx) {
        printf("Failed to initialize inference: %d\n", result);
        return;
    }
    
    printf("Online learning enabled with learning rate %.4f\n", 
           inf_config.online_learning_config.learning_rate);
    
    // Model parameters from training example
    const size_t input_size = 10;
    const size_t output_size = 1;
    
    // Run inference with online learning
    printf("\nRunning inference with real-time learning...\n");
    
    float correct_predictions = 0.0f;
    float total_predictions = 0.0f;
    float current_loss = 0.0f;
    
    // Generate test inputs with known outputs
    const int num_test_examples = 20;
    
    for (int i = 0; i < num_test_examples; i++) {
        // Prepare input
        float input[input_size];
        float expected_output[output_size];
        float model_output[output_size];
        
        // Alternating examples of each class
        float value = (i % 2 == 0) ? 0.2f : 0.8f;
        float sum = 0.0f;
        
        for (size_t j = 0; j < input_size; j++) {
            input[j] = value;
            sum += input[j];
        }
        
        expected_output[0] = (sum > 0.5f * input_size) ? 1.0f : 0.0f;
        
        // Run inference
        uint32_t input_tokens[1] = {0};  // Simplified token representation
        uint32_t next_token = 0;
        
        result = inference_next_token(inf_ctx, input_tokens, 1, &next_token);
        if (result != ONEBIT_SUCCESS) {
            printf("Inference failed: %d\n", result);
            continue;
        }
        
        // Get model output
        model_forward(ctx->model, input, model_output, 1, 1);
        
        // Check if prediction is correct (threshold at 0.5)
        bool prediction_correct = (model_output[0] > 0.5f) == (expected_output[0] > 0.5f);
        if (prediction_correct) {
            correct_predictions += 1.0f;
        }
        total_predictions += 1.0f;
        
        // Provide feedback for online learning
        result = inference_provide_feedback(inf_ctx, expected_output, &current_loss);
        if (result != ONEBIT_SUCCESS) {
            printf("Failed to provide feedback: %d\n", result);
            continue;
        }
        
        printf("Example %2d: input=%0.1f, output=%0.4f, expected=%0.1f, correct=%s, loss=%0.4f\n",
               i, value, model_output[0], expected_output[0], 
               prediction_correct ? "yes" : "no", current_loss);
    }
    
    // Get online learning stats
    int num_updates = 0;
    float avg_loss = 0.0f;
    inference_get_online_learning_stats(inf_ctx, &num_updates, &avg_loss);
    
    printf("\nOnline learning summary:\n");
    printf("  Total predictions: %.0f\n", total_predictions);
    printf("  Correct predictions: %.0f (%.1f%%)\n", 
           correct_predictions, (correct_predictions / total_predictions) * 100.0f);
    printf("  Online updates: %d\n", num_updates);
    printf("  Average loss: %f\n", avg_loss);
    
    // Clean up
    inference_cleanup(inf_ctx);
}

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Run standard training example
    OneBitContext ctx;
    OneBitConfig config = {0};
    config.device_type = DEVICE_CPU;
    config.num_threads = 4;
    config.seed = time(NULL);
    config.log_level = LOG_INFO;
    config.batch_size = 32;
    config.max_epochs = 10;
    
    int result = onebit_init(&ctx, &config);
    if (result != ONEBIT_SUCCESS) {
        printf("Failed to initialize OneBit: %d\n", result);
        return 1;
    }
    
    // Create a simple model (binary classifier)
    const size_t input_size = 10;
    const size_t hidden_size = 32;
    const size_t output_size = 1;
    
    // Create model
    result = model_create(&ctx, input_size, hidden_size, output_size);
    if (result != ONEBIT_SUCCESS) {
        printf("Failed to create model: %d\n", result);
        onebit_cleanup(&ctx);
        return 1;
    }
    
    // Run examples
    run_training_example();
    run_online_learning_example(&ctx);
    
    // Clean up
    onebit_cleanup(&ctx);
    printf("\nDone.\n");
    
    return 0;
} 