/**
 * @file onebit_data.h
 * @brief Production-grade data handling system for OneBit
 *
 * This module provides thread-safe, high-performance data loading and iteration
 * capabilities for large-scale machine learning workloads. It supports multiple
 * data formats, memory mapping for large datasets, and efficient batch processing.
 *
 * @author OneBit Team
 * @version 1.0
 * @date 2024-02-20
 */

#ifndef ONEBIT_DATA_H
#define ONEBIT_DATA_H

#include <stdint.h>
#include <stdbool.h>

// Error codes need to be updated
#define ONEBIT_SUCCESS        0
#define ONEBIT_ERROR_INVALID -1
#define ONEBIT_ERROR_MEMORY  -2
#define ONEBIT_ERROR_IO      -3
#define ONEBIT_ERROR_FORMAT  -4

// Rest of the header content... 