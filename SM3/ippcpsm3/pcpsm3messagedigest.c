/*******************************************************************************
 * Copyright 2013-2019 Intel Corporation
 * All Rights Reserved.
 *
 * If this  software was obtained  under the  Intel Simplified  Software
 *License, the following terms apply:
 *
 * The source code,  information  and material  ("Material") contained  herein
 *is owned by Intel Corporation or its  suppliers or licensors,  and  title to
 *such Material remains with Intel  Corporation or its  suppliers or  licensors.
 *The Material  contains  proprietary  information  of  Intel or  its suppliers
 *and licensors.  The Material is protected by  worldwide copyright  laws and
 *treaty provisions.  No part  of  the  Material   may  be  used,  copied,
 *reproduced, modified, published,  uploaded, posted, transmitted,  distributed
 *or disclosed in any way without Intel's prior express written permission.  No
 *license under any patent,  copyright or other  intellectual property rights in
 *the Material is granted to  or  conferred  upon  you,  either   expressly,  by
 *implication, inducement,  estoppel  or  otherwise.  Any  license   under such
 *intellectual property rights must be express and approved by Intel in writing.
 *
 * Unless otherwise agreed by Intel in writing,  you may not remove or alter
 *this notice or  any  other  notice   embedded  in  Materials  by  Intel  or
 *Intel's suppliers or licensors in any way.
 *
 *
 * If this  software  was obtained  under the  Apache License,  Version  2.0
 *(the "License"), the following terms apply:
 *
 * You may  not use this  file except  in compliance  with  the License.  You
 *may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 *
 * Unless  required  by   applicable  law  or  agreed  to  in  writing, software
 * distributed under the License  is distributed  on an  "AS IS"  BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * See the   License  for the   specific  language   governing   permissions and
 * limitations under the License.
 *******************************************************************************/

/*
//
//  Purpose:
//     Cryptography Primitive.
//     Digesting message according to SM3
//
//  Contents:
//        ippsSM3MessageDigest()
//
*/

#include "owncp.h"
#include "owndefs.h"
#include "pcphash.h"
//#include "pcphash_rmf.h"
#include "pcpsm3stuff.h"
#include "pcptool.h"

/*F*
//    Name: ippsSM3MessageDigest
//
// Purpose: Digest of the whole message.
//
// Returns:                Reason:
//    ippStsNullPtrErr        pMsg == NULL
//                            pMD == NULL
//    ippStsLengthErr         len <0
//    ippStsNoErr             no errors
//
// Parameters:
//    pMsg        pointer to the input message
//    len         input message length
//    pMD         address of the output digest
//
*F*/
IPPFUN(IppStatus, ippsSM3MessageDigest,
       (const Ipp8u* pMsg, int len, Ipp8u* pMD)) {
  /* test digest pointer */
  IPP_BAD_PTR1_RET(pMD);
  /* test message length */
  IPP_BADARG_RET((len < 0), ippStsLengthErr);
  /* test message pointer */
  IPP_BADARG_RET((len && !pMsg), ippStsNullPtrErr);

  {
    /* message length in the multiple MBS and the rest */
    int msgLenBlks = len & (-MBS_SM3);
    int msgLenRest = len - msgLenBlks;

    /* init hash */
    ((Ipp32u*)(pMD))[0] = sm3_iv[0];
    ((Ipp32u*)(pMD))[1] = sm3_iv[1];
    ((Ipp32u*)(pMD))[2] = sm3_iv[2];
    ((Ipp32u*)(pMD))[3] = sm3_iv[3];
    ((Ipp32u*)(pMD))[4] = sm3_iv[4];
    ((Ipp32u*)(pMD))[5] = sm3_iv[5];
    ((Ipp32u*)(pMD))[6] = sm3_iv[6];
    ((Ipp32u*)(pMD))[7] = sm3_iv[7];

    /* process main part of the message */
    if (msgLenBlks) {
      UpdateSM3((Ipp32u*)pMD, pMsg, msgLenBlks, sm3_cnt);
      pMsg += msgLenBlks;
    }

    cpFinalizeSM3((Ipp32u*)pMD, pMsg, msgLenRest, len);
    ((Ipp32u*)pMD)[0] = ENDIANNESS32(((Ipp32u*)pMD)[0]);
    ((Ipp32u*)pMD)[1] = ENDIANNESS32(((Ipp32u*)pMD)[1]);
    ((Ipp32u*)pMD)[2] = ENDIANNESS32(((Ipp32u*)pMD)[2]);
    ((Ipp32u*)pMD)[3] = ENDIANNESS32(((Ipp32u*)pMD)[3]);
    ((Ipp32u*)pMD)[4] = ENDIANNESS32(((Ipp32u*)pMD)[4]);
    ((Ipp32u*)pMD)[5] = ENDIANNESS32(((Ipp32u*)pMD)[5]);
    ((Ipp32u*)pMD)[6] = ENDIANNESS32(((Ipp32u*)pMD)[6]);
    ((Ipp32u*)pMD)[7] = ENDIANNESS32(((Ipp32u*)pMD)[7]);

    return ippStsNoErr;
  }
}

