#include "onebit.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_model_initialization(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    config.use_cuda = false;  // Use CPU for testing
    
    OneBitContext ctx;
    assert(onebit_init(&ctx, &config) == ONEBIT_SUCCESS);
    assert(ctx.is_initialized);
    assert(ctx.model != NULL);
    assert(ctx.tokenizer != NULL);
    
    onebit_cleanup(&ctx);
    printf("Model initialization test passed\n");
}

void test_text_generation(void) {
    OneBitConfig config = onebit_default_config();
    config.vocab_file = "test_vocab.txt";
    
    OneBitContext ctx;
    assert(onebit_init(&ctx, &config) == ONEBIT_SUCCESS);
    
    const char* prompt = "Hello, world!";
    char output[1024];
    
    assert(onebit_generate(&ctx, prompt, output, sizeof(output),
                         0.7f, 0.9f) == ONEBIT_SUCCESS);
    assert(strlen(output) > strlen(prompt));
    
    onebit_cleanup(&ctx);
    printf("Text generation test passed\n");
}

int main(void) {
    printf("Running OneBit tests...\n");
    
    test_model_initialization();
    test_text_generation();
    
    printf("All tests passed!\n");
    return 0;
} 