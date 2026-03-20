// I just want to make my own implementation of a ring buffer in C and C++
#include <cstring>
#include <stdint.h>

// Sort of like in cs140e
#define BUF_SIZE 1024 

// ####################################################################################################
//                                      C Implementation
// ####################################################################################################

typedef struct {
    uint8_t data[BUF_SIZE]; 
    uint32_t head;
    uint32_t tail;
    uint32_t size; // Kinda trivial/redundant but could save time
} cq_t;

void cq_init(cq_t* cq, uint8_t default_val) {
    cq->head = 0;
    cq->tail = 0;
    cq->size = 0;
    memset(cq->data, default_val, BUF_SIZE);    
}

void cq_push(cq_t* cq, uint8_t val) {

    // Case 1: More space, no overflow
}


// ####################################################################################################
//                                      C++ Implementation
// ####################################################################################################


#include <stdio.h>

using namespace std;


int main() {
    printf("Hello\n");
}