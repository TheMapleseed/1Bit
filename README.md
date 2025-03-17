# 1bit - Efficient 1-bit Transformer Library

1bit is a high-performance, self-contained library for deploying transformer-based language models with 1-bit weight quantization. It provides significant memory reduction while maintaining model quality and enabling faster inference.

## Features

- **1-bit Weight Quantization**: Compress transformer weights by using binary (1-bit) representation
- **Self-contained**: All components in a single build without external dependencies
- **Multi-instance Support**: Parallel inference across multiple models
- **Production Ready**: Enterprise-grade code for deployment scenarios
- **Hardware Acceleration**: SIMD, multi-threading, and platform-specific optimizations
- **C API**: Simple and efficient C interface for integration with any language

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Quick Start

```c
#include "onebit/onebit.h"

int main() {
    // Initialize context
    OneBitContext* ctx = onebit_create_context();
    
    // Load model
    OneBitModel* model = onebit_load_model(ctx, "path/to/model");
    
    // Generate text
    const char* prompt = "Once upon a time";
    char* output = onebit_generate_text(ctx, model, prompt, 100);
    
    printf("Generated: %s\n", output);
    
    // Cleanup
    onebit_free_string(ctx, output);
    onebit_free_model(ctx, model);
    onebit_free_context(ctx);
    
    return 0;
}
```

## Architecture

The library consists of several key components:

- **Transformer Model**: Core architecture with self-attention and feed-forward layers
- **Quantization**: 1-bit weight representation with scaling factors
- **Memory Management**: Optimized allocation and deallocation system
- **Inference Engine**: Efficient forward-pass computation
- **Tokenizer**: Text to token conversion and vice versa

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
