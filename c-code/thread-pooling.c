#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

// Structure to represent a task
typedef struct {
    void (*function)(void* arg);
    void* arg;
} task_t;

// Structure to represent the thread pool
typedef struct {
    pthread_t* threads;
    int num_threads;
    task_t* queue;
    int queue_size;
    int queue_head;
    int queue_tail;
    pthread_mutex_t queue_lock;
    pthread_cond_t queue_cond;
} thread_pool_t;

// Function to execute a task
void* worker_thread(void* arg) {
    thread_pool_t* pool = (thread_pool_t*)arg;
    while (1) {
        pthread_mutex_lock(&pool->queue_lock);
        while (pool->queue_head == pool->queue_tail) {
            pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
        }
        task_t task = pool->queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % pool->queue_size;
        pthread_mutex_unlock(&pool->queue_lock);
        task.function(task.arg);
    }
    return NULL;
}

// Function to create a thread pool
thread_pool_t* thread_pool_create(int num_threads, int queue_size) {
    thread_pool_t* pool = malloc(sizeof(thread_pool_t));
    pool->num_threads = num_threads;
    pool->queue_size = queue_size;
    pool->queue = malloc(sizeof(task_t) * queue_size);
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pthread_mutex_init(&pool->queue_lock, NULL);
    pthread_cond_init(&pool->queue_cond, NULL);
    pool->threads = malloc(sizeof(pthread_t) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
    }
    return pool;
}

// Function to add a task to the thread pool
void thread_pool_add_task(thread_pool_t* pool, void (*function)(void* arg), void* arg) {
    pthread_mutex_lock(&pool->queue_lock);
    while ((pool->queue_tail + 1) % pool->queue_size == pool->queue_head) {
        pthread_cond_wait(&pool->queue_cond, &pool->queue_lock);
    }
    pool->queue[pool->queue_tail].function = function;
    pool->queue[pool->queue_tail].arg = arg;
    pool->queue_tail = (pool->queue_tail + 1) % pool->queue_size;
    pthread_cond_signal(&pool->queue_cond);
    pthread_mutex_unlock(&pool->queue_lock);
}

// Function to wait for all tasks to complete
void thread_pool_wait(thread_pool_t* pool) {
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
}

// Example usage
void example_task(void* arg) {
    printf("Task executed with arg %d\n", *(int*)arg);
}

int main() {
    thread_pool_t* pool = thread_pool_create(4, 10);
    for (int i = 0; i < 10; i++) {
        int* arg = malloc(sizeof(int));
        *arg = i;
        thread_pool_add_task(pool, example_task, arg);
    }
    thread_pool_wait(pool);
    return 0;
}