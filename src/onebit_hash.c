#include "onebit/onebit_hash.h"
#include "onebit/onebit_error.h"
#include <string.h>

// FNV-1a hash constants
#define FNV_PRIME 0x01000193
#define FNV_OFFSET 0x811C9DC5

uint32_t hash_string(const char* str) {
    uint32_t hash = FNV_OFFSET;
    
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= FNV_PRIME;
    }
    
    return hash;
}

uint32_t hash_bytes(const void* data, size_t size) {
    uint32_t hash = FNV_OFFSET;
    const uint8_t* bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    
    return hash;
}

uint32_t hash_combine(uint32_t h1, uint32_t h2) {
    h1 ^= h2 + 0x9E3779B9 + (h1 << 6) + (h1 >> 2);
    return h1;
}

int hash_table_init(HashTable* table, const HashConfig* config) {
    if (!table || !config || config->num_buckets == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    table->buckets = calloc(config->num_buckets,
                           sizeof(HashEntry*));
    if (!table->buckets) {
        return ONEBIT_ERROR_MEMORY;
    }
    
    table->num_buckets = config->num_buckets;
    table->size = 0;
    
    if (pthread_mutex_init(&table->mutex, NULL) != 0) {
        free(table->buckets);
        return ONEBIT_ERROR_THREAD;
    }
    
    return ONEBIT_SUCCESS;
}

void hash_table_cleanup(HashTable* table) {
    if (!table) return;
    
    pthread_mutex_lock(&table->mutex);
    
    for (size_t i = 0; i < table->num_buckets; i++) {
        HashEntry* entry = table->buckets[i];
        while (entry) {
            HashEntry* next = entry->next;
            free(entry->key);
            free(entry->value);
            free(entry);
            entry = next;
        }
    }
    
    free(table->buckets);
    
    pthread_mutex_unlock(&table->mutex);
    pthread_mutex_destroy(&table->mutex);
}

int hash_table_set(HashTable* table, const char* key,
                   const void* value, size_t size) {
    if (!table || !key || !value || size == 0) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&table->mutex);
    
    uint32_t hash = hash_string(key);
    size_t bucket = hash % table->num_buckets;
    
    // Check if key exists
    HashEntry* entry = table->buckets[bucket];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            // Update existing entry
            void* new_value = malloc(size);
            if (!new_value) {
                pthread_mutex_unlock(&table->mutex);
                return ONEBIT_ERROR_MEMORY;
            }
            
            memcpy(new_value, value, size);
            free(entry->value);
            entry->value = new_value;
            entry->size = size;
            
            pthread_mutex_unlock(&table->mutex);
            return ONEBIT_SUCCESS;
        }
        entry = entry->next;
    }
    
    // Create new entry
    entry = malloc(sizeof(HashEntry));
    if (!entry) {
        pthread_mutex_unlock(&table->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    entry->key = strdup(key);
    if (!entry->key) {
        free(entry);
        pthread_mutex_unlock(&table->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    entry->value = malloc(size);
    if (!entry->value) {
        free(entry->key);
        free(entry);
        pthread_mutex_unlock(&table->mutex);
        return ONEBIT_ERROR_MEMORY;
    }
    
    memcpy(entry->value, value, size);
    entry->size = size;
    entry->hash = hash;
    
    // Add to bucket
    entry->next = table->buckets[bucket];
    table->buckets[bucket] = entry;
    table->size++;
    
    pthread_mutex_unlock(&table->mutex);
    return ONEBIT_SUCCESS;
}

int hash_table_get(const HashTable* table, const char* key,
                   void* value, size_t* size) {
    if (!table || !key || !value || !size) {
        return ONEBIT_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&table->mutex);
    
    uint32_t hash = hash_string(key);
    size_t bucket = hash % table->num_buckets;
    
    HashEntry* entry = table->buckets[bucket];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            memcpy(value, entry->value, entry->size);
            *size = entry->size;
            
            pthread_mutex_unlock((pthread_mutex_t*)&table->mutex);
            return ONEBIT_SUCCESS;
        }
        entry = entry->next;
    }
    
    pthread_mutex_unlock((pthread_mutex_t*)&table->mutex);
    return ONEBIT_ERROR_NOT_FOUND;
}

bool hash_table_contains(const HashTable* table,
                        const char* key) {
    if (!table || !key) return false;
    
    pthread_mutex_lock((pthread_mutex_t*)&table->mutex);
    
    uint32_t hash = hash_string(key);
    size_t bucket = hash % table->num_buckets;
    
    HashEntry* entry = table->buckets[bucket];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            pthread_mutex_unlock((pthread_mutex_t*)&table->mutex);
            return true;
        }
        entry = entry->next;
    }
    
    pthread_mutex_unlock((pthread_mutex_t*)&table->mutex);
    return false;
}

void hash_table_remove(HashTable* table, const char* key) {
    if (!table || !key) return;
    
    pthread_mutex_lock(&table->mutex);
    
    uint32_t hash = hash_string(key);
    size_t bucket = hash % table->num_buckets;
    
    HashEntry** curr = &table->buckets[bucket];
    while (*curr) {
        HashEntry* entry = *curr;
        
        if (strcmp(entry->key, key) == 0) {
            *curr = entry->next;
            free(entry->key);
            free(entry->value);
            free(entry);
            table->size--;
            break;
        }
        
        curr = &entry->next;
    }
    
    pthread_mutex_unlock(&table->mutex);
}

void hash_table_clear(HashTable* table) {
    if (!table) return;
    
    pthread_mutex_lock(&table->mutex);
    
    for (size_t i = 0; i < table->num_buckets; i++) {
        HashEntry* entry = table->buckets[i];
        while (entry) {
            HashEntry* next = entry->next;
            free(entry->key);
            free(entry->value);
            free(entry);
            entry = next;
        }
        table->buckets[i] = NULL;
    }
    
    table->size = 0;
    
    pthread_mutex_unlock(&table->mutex);
}

size_t hash_table_size(const HashTable* table) {
    return table ? table->size : 0;
} 