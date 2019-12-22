#ifndef _SM3_KDF_H_
#define _SM3_KDF_H_

#include <stdint.h>

#define SM3_KDF_RATE	32

typedef struct {
	//save the seed or message
	unsigned char buf[512];
	//the real length of buf
	unsigned int pos;
	//control word
	unsigned int cnt;
} sm3kdf_ctx;

//x[3-0] --> x[0-3]
#define BSWAP32(x)	(((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))

void sm3_kdf_init(sm3kdf_ctx* state, const unsigned char* key, unsigned long long klen, uint16_t nonce);

void sm3_kdf_absorb(sm3kdf_ctx* state, const unsigned char* input, unsigned long long inlen);

void sm3_kdf_squeezeblocks(unsigned char* out, unsigned long long nblocks, sm3kdf_ctx* state);

void sm3_kdf(unsigned char* output, unsigned long long outlen, const unsigned char* input, unsigned long long inlen);

void sm3_256(unsigned char *output, const unsigned char *input,	unsigned long long inlen);

void sm3_512(unsigned char *output, const unsigned char *input,	unsigned long long inlen);

void sm3_kdf_select(sm3kdf_ctx *state, uint32_t nonce);

#endif // !_SM3_KDF_H_
