/*
Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
This file is part of ByteQC.

ByteQC is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ByteQC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <cuComplex.h>
#include <assert.h>
#include <mutex>
#include <unordered_map>
#include "cutensor.h"

#define checkcuTensorError(x)                                                  \
    {                                                                          \
        const auto err = x;                                                    \
        if (err != CUTENSOR_STATUS_SUCCESS) {                                  \
            printf("Error: %s\n in %s:%d", cutensorGetErrorString(err),        \
                __FILE__, __LINE__);                                           \
            exit(-1);                                                          \
        }                                                                      \
    };

static const char *_cudaCublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

#define checkcuBlasError(F)                                                    \
    if ((F) != CUBLAS_STATUS_SUCCESS) {                                        \
        printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__,        \
            _cudaCublasGetErrorEnum(F));                                       \
        exit(-1);                                                              \
    }

std::mutex mtx;

struct _Handle {
    std::unordered_map<int, cutensorHandle_t> dcutensor;
    std::unordered_map<int, cublasHandle_t> dcublas;
    cutensorHandle_t &get_cutensor_handle() {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        mtx.lock();
        if (dcutensor.find(device) == dcutensor.end()) {
            cutensorHandle_t handle;
            checkcuTensorError(cutensorCreate(&handle));
            dcutensor[device] = handle;
        }
        mtx.unlock();
        return dcutensor[device];
    }
    cublasHandle_t &get_cublas_handle() {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        mtx.lock();
        if (dcublas.find(device) == dcublas.end()) {
            cublasHandle_t handle;
            checkcuBlasError(cublasCreate(&handle));
            dcublas[device] = handle;
        }
        mtx.unlock();
        return dcublas[device];
    }
    ~_Handle() {
        for (auto &p : dcutensor) {
            checkcuTensorError(cutensorDestroy(p.second));
        }
        for (auto &p : dcublas) {
            checkcuBlasError(cublasDestroy(p.second));
        }
    }
};

_Handle _handle;

template <typename T> inline cutensorDataType_t getDataType();
template <> inline cutensorDataType_t getDataType<double>() {
    return CUTENSOR_R_64F;
}
template <> inline cutensorDataType_t getDataType<float>() {
    return CUTENSOR_R_32F;
}
template <> inline cutensorDataType_t getDataType<cuDoubleComplex>() {
    return CUTENSOR_C_64F;
}
template <> inline cutensorDataType_t getDataType<cuComplex>() {
    return CUTENSOR_C_32F;
}

template <typename T> inline cutensorComputeDescriptor_t getComputeType();
template <> inline cutensorComputeDescriptor_t getComputeType<double>() {
    return CUTENSOR_COMPUTE_DESC_64F;
}
template <> inline cutensorComputeDescriptor_t getComputeType<float>() {
    return CUTENSOR_COMPUTE_DESC_32F;
}
template <>
inline cutensorComputeDescriptor_t getComputeType<cuDoubleComplex>() {
    return CUTENSOR_COMPUTE_DESC_64F;
}
template <> inline cutensorComputeDescriptor_t getComputeType<cuComplex>() {
    return CUTENSOR_COMPUTE_DESC_32F;
}

template <typename T, typename R>
int contraction(const char *indA, const T *A, const int64_t *stridesA,
    const int64_t *extentA, const char *indB, const T *B,
    const int64_t *stridesB, const int64_t *extentB, const char *indC, T *C,
    const int64_t *stridesC, void *buf = NULL, size_t buf_size = 0,
    const R alpha = 1.0, const R beta = 0.0,
    const cutensorOperator_t opA = CUTENSOR_OP_IDENTITY,
    const cutensorOperator_t opB = CUTENSOR_OP_IDENTITY,
    const cutensorOperator_t opC = CUTENSOR_OP_IDENTITY,
    const cutensorAlgo_t alg = CUTENSOR_ALGO_DEFAULT,
    const cutensorWorksizePreference_t wspref = CUTENSOR_WORKSPACE_DEFAULT,
    const cudaStream_t stream = 0) {
    cutensorDataType_t type = getDataType<T>();
    cutensorComputeDescriptor_t typeCompute = getComputeType<R>();

    int nA = strlen(indA);
    int32_t modeA[nA];
    for (int i = 0; i < nA; i++)
        modeA[i] = indA[i];
    int nB = strlen(indB);
    int32_t modeB[nB];
    for (int i = 0; i < nB; i++)
        modeB[i] = indB[i];
    int nC = strlen(indC);
    int32_t modeC[nC];
    for (int i = 0; i < nC; i++)
        modeC[i] = indC[i];

    int64_t extentC[nC];
    for (int i = 0; i < nA; i++) {
        for (int j = 0; j < nC; j++) {
            if (modeA[i] == modeC[j]) {
                extentC[j] = extentA[i];
            }
        }
    }
    for (int i = 0; i < nB; i++) {
        for (int j = 0; j < nC; j++) {
            if (modeB[i] == modeC[j]) {
                extentC[j] = extentB[i];
            }
        }
    }

    cutensorHandle_t handle = _handle.get_cutensor_handle();

    const uint32_t kAlignment =
        128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A) % kAlignment == 0);
    assert(uintptr_t(B) % kAlignment == 0);
    assert(uintptr_t(C) % kAlignment == 0);

    cutensorTensorDescriptor_t descA;
    checkcuTensorError(cutensorCreateTensorDescriptor(
        handle, &descA, nA, extentA, stridesA, type, kAlignment));
    cutensorTensorDescriptor_t descB;
    checkcuTensorError(cutensorCreateTensorDescriptor(
        handle, &descB, nB, extentB, stridesB, type, kAlignment));
    cutensorTensorDescriptor_t descC;
    checkcuTensorError(cutensorCreateTensorDescriptor(
        handle, &descC, nC, extentC, stridesC, type, kAlignment));

    cutensorOperationDescriptor_t desc;
    checkcuTensorError(cutensorCreateContraction(handle, &desc, descA, modeA,
        opA, descB, modeB, opB, descC, modeC, opC, descC, modeC, typeCompute));

    cutensorPlanPreference_t planPref;
    checkcuTensorError(cutensorCreatePlanPreference(
        handle, &planPref, alg, CUTENSOR_JIT_MODE_NONE));

    cutensorPlan_t plan;
    checkcuTensorError(
        cutensorCreatePlan(handle, &plan, desc, planPref, buf_size));

    checkcuTensorError(cutensorContract(handle, plan, (void *)&alpha, A, B,
        (void *)&beta, C, C, buf, buf_size, stream));

    checkcuTensorError(cutensorDestroyPlan(plan));
    checkcuTensorError(cutensorDestroyOperationDescriptor(desc));
    checkcuTensorError(cutensorDestroyTensorDescriptor(descA));
    checkcuTensorError(cutensorDestroyTensorDescriptor(descB));
    checkcuTensorError(cutensorDestroyTensorDescriptor(descC));

    return 0;
}

template int contraction<double, double>(const char *indA, const double *A,
    const int64_t *stridesA, const int64_t *extentA, const char *indB,
    const double *B, const int64_t *stridesB, const int64_t *extentB,
    const char *indC, double *C, const int64_t *stridesC, void *buf,
    size_t buf_size, const double alpha, const double beta,
    const cutensorOperator_t opA, const cutensorOperator_t opB,
    const cutensorOperator_t opC, const cutensorAlgo_t alg,
    const cutensorWorksizePreference_t wspref, const cudaStream_t stream);
