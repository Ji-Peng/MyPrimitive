#include "sm3kdf.h"
#include "./ippcpsm3/ippcp.h"
#include <stdlib.h>
#include <memory.h>
#include <emmintrin.h>
#include <stdint.h>

void sm3_kdf(unsigned char* output, unsigned long long outlen, const unsigned char* input, unsigned long long inlen)
{
	unsigned char* pData = NULL;
	unsigned char cdgst[32];
	unsigned char *cCnt;
	int nCnt = 1;
	int nDgst = 32;
	int nTimes;
	int i = 0;

	if (NULL == (pData = (unsigned char*)malloc(inlen + 4)))
		return;
	memcpy(pData, input, inlen);
	cCnt = pData + inlen;

	nTimes = ((int)outlen + 31) >> 5;
	nDgst = 32; nCnt = 1;
	for (i = 0; i < nTimes; i++)
	{
		//cCnt
		{
			cCnt[0] = (nCnt >> 24) & 0xFF;
			cCnt[1] = (nCnt >> 16) & 0xFF;
			cCnt[2] = (nCnt >> 8) & 0xFF;
			cCnt[3] = (nCnt) & 0xFF;
		}
		ippsSM3MessageDigest(pData, (int)inlen + 4, cdgst);

		if (i == nTimes - 1)
		{
			//if (outlen & 0x3f != 0) nDgst = outlen & 0x3f;
			if (outlen % 32 != 0)
			{
				nDgst = outlen % 32;
			}
		}
		memcpy(output, cdgst, nDgst);
		output += nDgst;
		nCnt++;
	}
	free(pData);
}

void sm3_256(unsigned char * output, const unsigned char * input, unsigned long long inlen)
{
	sm3_kdf(output, 32, input, inlen);
}

void sm3_512(unsigned char * output, const unsigned char * input, unsigned long long inlen)
{
	sm3_kdf(output, 64, input, inlen);
}

/**
 * @description: use BSWAP32(nonce) to change the state->cnt
 * @param {state: old state} 
 * @return: 
 */
void sm3_kdf_select(sm3kdf_ctx * state, uint32_t nonce)
{
	state->cnt = (unsigned int)BSWAP32(nonce);
}

/**
 * @description: using key to init the state
 * @param {type} 
 * @return: 
 */
void sm3_kdf_init(sm3kdf_ctx* state, const unsigned char* key, unsigned long long klen, uint16_t nonce)
{
	//state->buf = key, copy the key to state->buf
	memcpy(state->buf, key, (unsigned int)klen);
	//state->pos is the length of state->buf
	state->pos = (unsigned int)klen;
	//nonce = nonce[1]||nonce[0], state->buf == key||nonce[0]
	state->buf[state->pos++] = (unsigned char)(nonce & 0xFF);
	//state->buf = key||nonce[0]||nonce[1]
	//nonce[0]||nonce[1] used to distinguish between message digest and key derivation
	state->buf[state->pos++] = (unsigned char)(nonce >> 8);
	//init the control word
	state->cnt = 1;
}

/**
 * @description: Absorb the input to state->buf
 * This function is derived from the keccak algorithm, whose structure is sponge.
 * The meaning of "absorb" is seem with "a sponge absorbs water"
 * @param {type} 
 * @return: 
 */
void sm3_kdf_absorb(sm3kdf_ctx* state, const unsigned char* input, unsigned long long inlen)
{
	//state->buf = input
	memcpy(state->buf, input, (unsigned int)inlen);
	//state->pos = inlen
	state->pos = (unsigned int)inlen;
	//init the control word
	state->cnt = 1;
}

/**
 * @description: 
 * This function is also derived from the keccak algorithm, whose structure is sponge.
 * The meaning of "squeezeblocks" is seem with "squeeze n blocks of water out of the sponge"
 * @param {type} 
 * @return: 
 */
void sm3_kdf_squeezeblocks(unsigned char* out, unsigned long long nblocks, sm3kdf_ctx* state)
{
	//control word pointer, tips: the avalanche effect of the hash function
	unsigned char *pCnt;
	//number control word
	unsigned int nCnt;
	//loop
	unsigned int i;

	nCnt = state->cnt;
	//point to the end of state->buf
	pCnt = state->buf + state->pos;
	//one loop generate 128-bytes which is also one block
	for (i = 0; i < nblocks; i++)
	{
		//exectuing the sm3 function once generates 32bytes, 4 * 32 = 128-bytes
		for (size_t j = 0; j < 4; j++)
		{
			//pCnt[0]=nCnt[3]
			pCnt[0] = (nCnt >> 24) & 0xFF;
			//pCnt[1]=nCnt[2]
			pCnt[1] = (nCnt >> 16) & 0xFF;
			//pCnt[2]=nCnt[1]
			pCnt[2] = (nCnt >> 8) & 0xFF;
			//pCnt[3]=nCnt[0]
			pCnt[3] = (nCnt) & 0xFF;
			//so pCnt[0-3] == nCnt[3-0]

			//out = hash(state->buf), one execute --> 32-bytes out
			ippsSM3MessageDigest(state->buf, state->pos + 4, out);
			//out+=32
			out += SM3_KDF_RATE;
			//using the avalanche effect of hash functions
			nCnt++;
		}
	}
	//saving the value of nCnt is necessary
	state->cnt = nCnt;
}
