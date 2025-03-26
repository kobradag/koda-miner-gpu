#include<stdint.h>
#include <assert.h>
#include "keccak-tiny.c"
#include "xoshiro256starstar.c"
#include "blake3_compact.h"
#include "blake2b.cu"
#include "sha3.cu"
#include "sha3.cuh"
#include "skein.cuh"

typedef uint8_t Hash[32];

typedef union _uint256_t {
    uint64_t number[4];
    uint8_t hash[32];
} uint256_t;

#define BLOCKDIM 1024
#define MATRIX_SIZE 64
#define HALF_MATRIX_SIZE 32
#define QUARTER_MATRIX_SIZE 16
#define HASH_HEADER_SIZE 72

#define RANDOM_LEAN 0
#define RANDOM_XOSHIRO 1
#define HASH_SIZE 32

#define LT_U256(X,Y) (X.number[3] != Y.number[3] ? X.number[3] < Y.number[3] : X.number[2] != Y.number[2] ? X.number[2] < Y.number[2] : X.number[1] != Y.number[1] ? X.number[1] < Y.number[1] : X.number[0] < Y.number[0])

__constant__ uint8_t matrix[MATRIX_SIZE][MATRIX_SIZE];
__constant__ uint8_t hash_header[HASH_HEADER_SIZE];
__constant__ uint256_t target;
__constant__ uint16_t sinusoidal_table[256] = {
    0, 841, 909, 141, 64780, 64578, 65257, 656,
    989, 412, 64992, 64537, 65000, 420, 990, 650,
    65249, 64575, 64786, 149, 912, 836, 65528, 64690,
    64631, 65404, 762, 956, 270, 64873, 64548, 65132,
    551, 999, 529, 65108, 64545, 64893, 296, 963,
    745, 65378, 64620, 64705, 17, 850, 901, 123,
    64768, 64583, 65274, 670, 986, 395, 64978, 64537,
    65015, 436, 992, 636, 65232, 64570, 64797, 167,
    920, 826, 65510, 64681, 64639, 65422, 773, 951,
    253, 64860, 64551, 65149, 566, 999, 513, 65092,
    64543, 64907, 313, 968, 733, 65360, 64613, 64715,
    35, 860, 893, 105, 64757, 64588, 65291, 683,
    983, 379, 64963, 64537, 65030, 452, 994, 622,
    65215, 64566, 64809, 184, 926, 816, 65492, 64672,
    64647, 65439, 784, 945, 236, 64847, 64555, 65165,
    580, 998, 498, 65077, 64541, 64920, 329, 972,
    721, 65343, 64606, 64725, 53, 868, 885, 88,
    64746, 64594, 65308, 696, 980, 363, 64949, 64538,
    65045, 467, 996, 609, 65198, 64562, 64822, 202,
    933, 806, 65475, 64663, 64655, 65457, 795, 939,
    219, 64834, 64558, 65182, 594, 997, 483, 65061,
    64539, 64935, 346, 976, 708, 65326, 64600, 64735,
    70, 877, 877, 70, 64735, 64600, 65326, 708,
    976, 346, 64934, 64539, 65061, 483, 997, 594,
    65182, 64558, 64834, 219, 939, 795, 65457, 64655,
    64663, 65475, 806, 933, 202, 64822, 64562, 65198,
    609, 996, 467, 65045, 64538, 64949, 363, 980,
    696, 65308, 64594, 64746, 88, 885, 868, 53,
    64725, 64606, 65343, 721, 972, 329, 64920, 64541,
    65077, 498, 998, 580, 65165, 64555, 64847, 236,
    945, 784, 65439, 64646, 64672, 65492, 816, 926,
    184, 64809, 64566, 65215, 623, 994, 451, 65030,
};
__constant__ uint8_t exp2_lookup[16] = {
    1, 2, 4, 8, 16, 32, 64, 128, 255, 255, 255, 255, 255, 255, 255, 255
};

__device__ void blake2b_hash(uint8_t* output, const uint8_t* input, uint32_t input_len);
__device__ void cn_skein(const uint8_t* input, size_t input_len, uint8_t* output);

__device__ void convert_endian(uint32_t* input, uint8_t* output) {
    for (int i = 0; i < 8; i++) {
        uint32_t value = input[i];
        output[i * 4 + 0] = (value) & 0xFF;
        output[i * 4 + 1] = (value >> 8) & 0xFF;
        output[i * 4 + 2] = (value >> 16) & 0xFF;
        output[i * 4 + 3] = (value >> 24) & 0xFF;
    }
}

__device__ __inline__ uint8_t apply_xor(uint32_t sum) {
    uint8_t result1 = (sum & 0xF) ^ ((sum >> 4) & 0xF) ^ ((sum >> 8) & 0xF);
    return result1;
}
__device__ __inline__ uint16_t sinusoidal_multiply(uint8_t matrix_value, uint8_t vec_value) {
    uint16_t sin_value = sinusoidal_table[vec_value];
    uint16_t product = (uint16_t)(matrix_value) * (uint16_t)(sin_value);
    return product & 0xFF;
}
__device__ __inline__ uint8_t* heavy_hash_step(uint8_t *input_hash, uint8_t *output_hash) {
    uint8_t vec[64];

    for (int i = 0; i < 32; i++) {
        vec[2 * i] = input_hash[i] >> 4;
        vec[2 * i + 1] = input_hash[i] & 0x0F;
    }

    uint16_t sinusoidal_values[64];
    for (int j = 0; j < 64; j++) {
        sinusoidal_values[j] = sinusoidal_table[vec[j] & 0xFF]; 
    }

    uint8_t product[64] = {0};
    #pragma unroll 4 
    for (int i = 0; i < 64; i++) {
        uint16_t sum = 0;
        for (int j = 0; j < 64; j++) {
            sum = (sum + ((uint16_t)matrix[i][j] * sinusoidal_values[j])) & 0xFFFF;  // Prevent overflow
        }
        product[i] = (sum & 0xF) ^ ((sum >> 4) & 0xF) ^ ((sum >> 8) & 0xF);
    }

    for (int i = 0; i < 32; i++) {
        uint8_t shift_value = product[2 * i] << 4;
        
       /* double exp_value = exp2((double)product[2 * i + 1]);  
        uint8_t exponent_value = (uint8_t)exp_value;*/
        uint8_t exponent_value = exp2_lookup[product[2 * i + 1]];


        if (exponent_value == 0xFF) {
            exponent_value = 0;
        }

        output_hash[i] = input_hash[i] ^ (shift_value | exponent_value);
    }

    return output_hash;
}

extern "C" {


    __global__ void heavy_hash(const uint64_t nonce_mask, const uint64_t nonce_fixed, const uint64_t nonces_len, uint8_t random_type, void* states, uint64_t *final_nonce) {
        // assuming header_len is 72
        int nonceId = threadIdx.x + blockIdx.x*blockDim.x;
        if (nonceId < nonces_len) {
            if (nonceId == 0) *final_nonce = 0;
            uint64_t nonce;
            switch (random_type) {
                case RANDOM_LEAN:
                    nonce = ((uint64_t *)states)[0] ^ nonceId;
                    break;
                case RANDOM_XOSHIRO:
                default:
                    nonce = xoshiro256_next(((ulonglong4 *)states) + nonceId);
                    break;
            }
            nonce = (nonce & nonce_mask) | nonce_fixed;
            // header
            uint8_t input[80];
            memcpy(input, hash_header, HASH_HEADER_SIZE);
            
            uint256_t hash_;
            uint256_t blake3_;
            memcpy(input +  HASH_HEADER_SIZE, (uint8_t *)(&nonce), 8);
            
            blake3_hasher pow_hasher;
            blake3_hasher_init(&pow_hasher);
            blake3_hasher_update(&pow_hasher, input, 80);
            blake3_hasher_finalize(&pow_hasher, blake3_.hash, BLAKE3_KEY_LEN);

            uint8_t input_bytes[HASH_SIZE];
            uint32_t skein_hash[HASH_SIZE / 4];
            uint8_t blake2_hash[HASH_SIZE];
            
            convert_endian((uint32_t*)blake3_.hash, input_bytes);
            
            // Step 1: BLAKE2b hashing
            blake2b_hash(blake2_hash, input_bytes, HASH_SIZE);
            
            // Step 2: Skein hashing
            cn_skein(blake2_hash, (DataLength)HASH_SIZE, (uint32_t*)skein_hash);
            
            // Step 3: SHA3-256 hashing
            sha3_context ctx;
            sha3_Init256(&ctx);
            sha3_Update(&ctx, skein_hash, HASH_SIZE);
            const uint8_t* result = (const uint8_t*) sha3_Finalize(&ctx);
        
            uint8_t output_hash[32];
            uint8_t temp_result[32];  
            memcpy(temp_result, result, 32);  

            heavy_hash_step(temp_result, output_hash);  

            uint256_t temp_hash;
            memcpy(temp_hash.hash, output_hash, HASH_SIZE); 

            convert_endian((uint32_t*)temp_hash.hash, hash_.hash); 
            hash_ = temp_hash;  
           
            memset(input, 0, 80);
            memcpy(input, hash_.hash, 32);
            blake3_hasher heavy_hasher;
            blake3_hasher_init(&heavy_hasher);
            blake3_hasher_update(&heavy_hasher, input, 32);
            blake3_hasher_finalize(&heavy_hasher, hash_.hash, BLAKE3_KEY_LEN);

            
            if (LT_U256(hash_, target)){
                atomicCAS((unsigned long long int*) final_nonce, 0, (unsigned long long int) nonce);
            }
          
        }
    }

}
