#include <immintrin.h>
#include <stdint.h>

#include "cbd.h"
// The macro AKCN_N is included in this file
#include "params.h"

/**
 * @description: cbd: center binomial distribution using AVX2
 * @param: poly *r is output, uint8_t *buf is input, which includes random
 * bits， and the length of buf is 32 * AKCN_N / 64 = 512.
 * @return:
 */
void cbd(poly *r, const uint8_t *buf) {
  __m256i vec0, vec1, vec2, vec3, tmp;
  // _m256_set1_epi32 : Broadcast 32-bit integer a to all elements of dst
  // epi: extended packed integer, extended: SSE not MMX
  // 0x5 == 0101B, so mask55 == 0101 0101 ... 0101B
  const __m256i mask55 = _mm256_set1_epi32(0x55555555);
  // mask33 == 0011 0011 ... 0011B
  const __m256i mask33 = _mm256_set1_epi32(0x33333333);
  // mask03 == 0000 0011 0000 0011 ... 0000 0011B
  const __m256i mask03 = _mm256_set1_epi32(0x03030303);

  // one loop can generate 64 coefficients
  for (size_t i = 0; i < AKCN_N / 64; i++) {
    /*-----------------------------------------------------------------------------*/
    // _mm256_loadu_si256: Load 256-bits of integer data from memory into dst
    // si256: signed 256-bit integer
    // so, vec0 includes 32bytes == 32 * 8 == 256-bit coming from buf
    vec0 = _mm256_loadu_si256((__m256i *)&buf[32 * i]);
    // _mm256_srli_epi32: Shift packed 32-bit integers in a right by imm8
    // shift every 32-bit in a right by 1
    vec1 = _mm256_srli_epi32(vec0, 1);
    // _mm256_and_si256: Compute the bitwise AND of 256 bits
    vec0 = _mm256_and_si256(mask55, vec0);
    vec1 = _mm256_and_si256(mask55, vec1);
    // _mm256_add_epi32: Add packed 32-bit integers in a and b
    vec0 = _mm256_add_epi32(vec0, vec1);
    // simulation：
    // vec0 = a7 a6 a5 a4 a3 a2 a1 a0
    // vec1 = 0  a7 a6 a5 a4 a3 a2 a1
    // vec0 = 0  a6 0  a4 0  a2 0  a0
    // vec1 = 0  a7 0  a5 0  a3 0  a1
    // vec0 = 0  a6 0  a4 0  a2 0  a0 + 0  a7 0  a5 0  a3 0  a1
    // assuming vec0 = b7 b6 b5 b4 b3 b2 b1 b0

    /*-----------------------------------------------------------------------------*/
    //vec0 = b7 b6 b5 b4 b3 b2 b1 b0
    //vec1 = 0  0  b7 b6 b5 b4 b3 b2
    vec1 = _mm256_srli_epi32(vec0, 2);
    //vec0 = 0  0  b5 b4 0  0  b1 b0
    vec0 = _mm256_and_si256(mask33, vec0);
    //vec1 = 0  0  b7 b6 0  0  b3 b2 
    vec1 = _mm256_and_si256(mask33, vec1);

    //vec2 = 0  0  0  0  0  0  b1 b0
    vec2 = _mm256_srli_epi32(vec0, 4);
    //vec3 = 0  0  0  0  0  0  b7 b6
    vec3 = _mm256_srli_epi32(vec1, 4);
    //vec0 = 0  0  0  0  0  0  b1 b0
    vec0 = _mm256_and_si256(mask03, vec0);
    //vec1 = 0  0  0  0  0  0  b3 b2
    vec1 = _mm256_and_si256(mask03, vec1);
    //vec2 = 0  0  0  0  0  0  b1 b0 
    vec2 = _mm256_and_si256(mask03, vec2);
    //vec3 = 0  0  0  0  0  0  b7 b6
    vec3 = _mm256_and_si256(mask03, vec3);

    //vec1 = 0  0  0  0  0  0  b1 b0 - 0  0  0  0  0  0  b3 b2
    //assuming vec1 = c7 c6 c5 c4 c3 c2 c1 c0
    vec1 = _mm256_sub_epi8(vec0, vec1);
    //vec3 = 0  0  0  0  0  0  b1 b0 - 0  0  0  0  0  0  b7 b6
    //assuming vec3 = d7 d6 d5 d4 d3 d2 d1 d0
    vec3 = _mm256_sub_epi8(vec2, vec3);
    //_mm256_castsi256_si128: cast type _m256i to _m128i
    //_mm256_cvtepi8_epi16: using zero-extend to convert 8-bit word to 16-bit
    //vec0 = 0...0 0 0 0 0 0 0 b1 b0
    vec0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec1));
    //_mm256_extracti128_si256: if 1, extract vec1[255:128]
    //vec1 = 0...0 0...0
    vec1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec1, 1));
    //vec2 = 0...0 d7 d6 d5 d4 d3 d2 d1 d0 
    vec2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec3));
    //vec3 = 0...0 0...0
    vec3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec3, 1));
    //tmp = 0...0 d7 d6 d5 d4 d3 d2 d1 d0 0...0 0 0 0 0 0 0 b1 b0
    tmp = _mm256_unpacklo_epi16(vec0, vec2);
    //vec2 = 0...0 0..0
    vec2 = _mm256_unpackhi_epi16(vec0, vec2);
    //vec0 = 0...0 d7 d6 d5 d4 d3 d2 d1 d0 0...0 0 0 0 0 0 0 b1 b0
    vec0 = tmp;
    //tmp = 0...0 0...0 0...0 0...0
    tmp = _mm256_unpacklo_epi16(vec1, vec3);
    //vec3 =  0...0 0...0 0...0 0...0
    vec3 = _mm256_unpackhi_epi16(vec1, vec3);
    //vec1 = 0...0  0...0 0...0 0...0 
    vec1 = tmp;
    
    tmp = _mm256_permute2x128_si256(vec0, vec2, 0x20);
    vec2 = _mm256_permute2x128_si256(vec0, vec2, 0x31);
    vec0 = tmp;
    tmp = _mm256_permute2x128_si256(vec1, vec3, 0x20);
    vec3 = _mm256_permute2x128_si256(vec1, vec3, 0x31);
    vec1 = tmp;

    _mm256_store_si256((__m256i *)&r->coeffs[64 * i + 0], vec0);
    _mm256_store_si256((__m256i *)&r->coeffs[64 * i + 16], vec2);
    _mm256_store_si256((__m256i *)&r->coeffs[64 * i + 32], vec1);
    _mm256_store_si256((__m256i *)&r->coeffs[64 * i + 48], vec3);
  }
  for (size_t i = 0; i < AKCN_N; i++) {
    r->coeffs[i] += AKCN_Q;
  }
}
