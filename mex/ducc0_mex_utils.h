/*
 * DUCC0 MATLAB MEX Utilities
 * 
 * This file provides utilities for converting between MATLAB mxArray
 * and DUCC's multi-dimensional array views.
 * 
 * Key features:
 * - Handles MATLAB's column-major storage vs DUCC's row-major expectations
 * - Converts MATLAB's separate real/imaginary arrays to interleaved complex
 * - Supports automatic dimension reordering
 * - Provides error handling for MATLAB
 */

#ifndef DUCC0_MEX_UTILS_H
#define DUCC0_MEX_UTILS_H

#include "mex.h"
#include "ducc0/infra/mav.h"
#include "ducc0/infra/error_handling.h"
#include "ducc0/bindings/typecode.h"
#include <vector>
#include <complex>
#include <memory>
#include <algorithm>
#include <type_traits>

namespace ducc0_mex {

using namespace ducc0;
using namespace std;

// Helper trait to check if T is a complex type
template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template<typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// Error handler that throws MATLAB error
inline void handleDuccError(const exception &e)
{
    mexErrMsgIdAndTxt("DUCC0:MEXError", "%s", e.what());
}

// Convert MATLAB dimensions (column-major) to vector (row-major for DUCC)
inline vector<size_t> matlabDimsToVector(const mwSize *dims, mwSize ndim)
{
    vector<size_t> result(ndim);
    // MATLAB stores dimensions in column-major order [rows, cols, ...]
    // DUCC expects row-major order, so we reverse
    for (mwSize i = 0; i < ndim; ++i) {
        result[ndim - 1 - i] = dims[i];
    }
    return result;
}

// Convert vector dimensions (row-major) to MATLAB dimensions (column-major)
inline void vectorDimsToMatlab(const vector<size_t> &dims, mwSize *out_dims)
{
    mwSize ndim = dims.size();
    // Reverse to convert from row-major (DUCC) to column-major (MATLAB)
    for (size_t i = 0; i < dims.size(); ++i) {
        out_dims[dims.size() - 1 - i] = dims[i];
    }
}

// Convert axes from MATLAB 1-based column-major to DUCC 0-based row-major
inline vector<size_t> convertAxes(const mxArray *axes_arr, size_t ndim)
{
    vector<size_t> axes;
    
    if (axes_arr == nullptr || mxIsEmpty(axes_arr)) {
        // Default: all axes in row-major order
        for (size_t i = 0; i < ndim; ++i) {
            axes.push_back(i);
        }
    } else {
        size_t naxes = mxGetNumberOfElements(axes_arr);
        double *axes_data = mxGetPr(axes_arr);
        
        for (size_t i = 0; i < naxes; ++i) {
            // MATLAB uses 1-based indexing, column-major
            size_t ax_matlab = (size_t)axes_data[i] - 1;
            // Convert to 0-based row-major
            size_t ax_ducc = ndim - 1 - ax_matlab;
            if (ax_ducc >= ndim) {
                MR_fail("Axis index out of range");
            }
            axes.push_back(ax_ducc);
        }
    }
    
    return axes;
}

// Helper to compute MATLAB linear index from row-major indices
// Optimized version that pre-computes strides
inline size_t matlabLinearIndex(const size_t *indices_rowmajor,
                                 const size_t *strides_matlab, size_t ndim)
{
    size_t idx = 0;
    // MATLAB uses column-major, so reverse the indices
    for (size_t i = 0; i < ndim; ++i) {
        idx += indices_rowmajor[ndim - 1 - i] * strides_matlab[i];
    }
    return idx;
}

// Helper to increment row-major indices (rightmost dimension first)
// Optimized inline version
inline bool incrementIndices(size_t *indices, const size_t *shape, size_t ndim)
{
    for (size_t i = 0; i < ndim; ++i) {
        indices[i]++;
        if (indices[i] < shape[i]) {
            return true;  // Still valid, done
        }
        indices[i] = 0;  // Overflow, carry to next dimension
    }
    return false;  // All indices overflowed
}

// Copy data from MATLAB array to temporary buffer with proper layout
// This handles column-major to row-major conversion and complex interleaving
// Optimized version with pre-computed strides

// Overload for real types
template<typename T>
typename std::enable_if<!is_complex_v<T>>::type
copyMatlabToBuffer(const mxArray *arr, T *buffer, const vector<size_t> &shape_ducc)
{
    const void *real_data = mxGetData(arr);
    
    mwSize ndim_matlab = mxGetNumberOfDimensions(arr);
    const mwSize *dims_matlab = mxGetDimensions(arr);
    
    // Calculate total elements
    size_t nelem = 1;
    for (size_t s : shape_ducc) {
        nelem *= s;
    }
    
    // Real data - copy with dimension reordering
    const T *real = static_cast<const T *>(real_data);
    
    if (ndim_matlab == 1) {
        // 1D - just copy (no reordering needed)
        memcpy(buffer, real, nelem * sizeof(T));
    } else {
        // Multi-dimensional - need to reorder
        // Pre-compute MATLAB strides for faster indexing
        vector<size_t> strides_matlab(ndim_matlab);
        strides_matlab[0] = 1;
        for (mwSize i = 1; i < ndim_matlab; ++i) {
            strides_matlab[i] = strides_matlab[i-1] * dims_matlab[i-1];
        }
        
        vector<size_t> indices(shape_ducc.size(), 0);
        size_t ndim = shape_ducc.size();
        
        // Unroll the loop for small dimensions
        if (ndim == 2) {
            size_t idx0_max = shape_ducc[0];
            size_t idx1_max = shape_ducc[1];
            for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                    indices[0] = idx0;
                    indices[1] = idx1;
                    size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                    *buffer++ = real[idx_matlab];
                }
            }
        } else if (ndim == 3) {
            size_t idx0_max = shape_ducc[0];
            size_t idx1_max = shape_ducc[1];
            size_t idx2_max = shape_ducc[2];
            for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                    for (size_t idx2 = 0; idx2 < idx2_max; ++idx2) {
                        indices[0] = idx0;
                        indices[1] = idx1;
                        indices[2] = idx2;
                        size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                        *buffer++ = real[idx_matlab];
                    }
                }
            }
        } else {
            // General case
            for (size_t i = 0; i < nelem; ++i) {
                size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                buffer[i] = real[idx_matlab];
                if (i < nelem - 1) {
                    incrementIndices(indices.data(), shape_ducc.data(), ndim);
                }
            }
        }
    }
}

// Overload for complex types
template<typename T>
typename std::enable_if<is_complex_v<T>>::type
copyMatlabToBuffer(const mxArray *arr, T *buffer, const vector<size_t> &shape_ducc)
{
    const void *real_data = mxGetData(arr);
    const void *imag_data = mxIsComplex(arr) ? mxGetImagData(arr) : nullptr;
    
    mwSize ndim_matlab = mxGetNumberOfDimensions(arr);
    const mwSize *dims_matlab = mxGetDimensions(arr);
    
    // Calculate total elements
    size_t nelem = 1;
    for (size_t s : shape_ducc) {
        nelem *= s;
    }
    
    if (imag_data == nullptr) {
        // Real data stored in complex array - just copy real part
        using real_t = typename T::value_type;
        const real_t *real = static_cast<const real_t *>(real_data);
        
        if (ndim_matlab == 1) {
            // 1D - just copy
            for (size_t i = 0; i < nelem; ++i) {
                buffer[i] = T(real[i], 0);
            }
        } else {
            // Multi-dimensional - need to reorder
            vector<size_t> strides_matlab(ndim_matlab);
            strides_matlab[0] = 1;
            for (mwSize i = 1; i < ndim_matlab; ++i) {
                strides_matlab[i] = strides_matlab[i-1] * dims_matlab[i-1];
            }
            
            vector<size_t> indices(shape_ducc.size(), 0);
            size_t ndim = shape_ducc.size();
            
            if (ndim == 2) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        indices[0] = idx0;
                        indices[1] = idx1;
                        size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                        *buffer++ = T(real[idx_matlab], 0);
                    }
                }
            } else if (ndim == 3) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                size_t idx2_max = shape_ducc[2];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        for (size_t idx2 = 0; idx2 < idx2_max; ++idx2) {
                            indices[0] = idx0;
                            indices[1] = idx1;
                            indices[2] = idx2;
                            size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                            *buffer++ = T(real[idx_matlab], 0);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < nelem; ++i) {
                    size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                    buffer[i] = T(real[idx_matlab], 0);
                    if (i < nelem - 1) {
                        incrementIndices(indices.data(), shape_ducc.data(), ndim);
                    }
                }
            }
        }
    } else {
        // Complex data - interleave real and imaginary parts
        using real_t = typename T::value_type;
        const real_t *real = static_cast<const real_t *>(real_data);
        const real_t *imag = static_cast<const real_t *>(imag_data);
        
        if (ndim_matlab == 1) {
            // 1D - just interleave (no reordering needed)
            // Optimized: use pointer arithmetic
            const real_t *r = real;
            const real_t *i = imag;
            T *b = buffer;
            for (size_t j = 0; j < nelem; ++j) {
                *b++ = T(*r++, *i++);
            }
        } else {
            // Multi-dimensional - need to reorder and interleave
            // Pre-compute MATLAB strides for faster indexing
            vector<size_t> strides_matlab(ndim_matlab);
            strides_matlab[0] = 1;
            for (mwSize i = 1; i < ndim_matlab; ++i) {
                strides_matlab[i] = strides_matlab[i-1] * dims_matlab[i-1];
            }
            
            vector<size_t> indices(shape_ducc.size(), 0);
            size_t ndim = shape_ducc.size();
            
            // Unroll the loop for small dimensions
            if (ndim == 2) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        indices[0] = idx0;
                        indices[1] = idx1;
                        size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                        *buffer++ = T(real[idx_matlab], imag[idx_matlab]);
                    }
                }
            } else if (ndim == 3) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                size_t idx2_max = shape_ducc[2];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        for (size_t idx2 = 0; idx2 < idx2_max; ++idx2) {
                            indices[0] = idx0;
                            indices[1] = idx1;
                            indices[2] = idx2;
                            size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                            *buffer++ = T(real[idx_matlab], imag[idx_matlab]);
                        }
                    }
                }
            } else {
                // General case
                for (size_t i = 0; i < nelem; ++i) {
                    size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                    buffer[i] = T(real[idx_matlab], imag[idx_matlab]);
                    if (i < nelem - 1) {
                        incrementIndices(indices.data(), shape_ducc.data(), ndim);
                    }
                }
            }
        }
    }
}

// Copy data from buffer to MATLAB array with proper layout
// Optimized version with pre-computed strides
template<typename T>
void copyBufferToMatlab(const T *buffer, mxArray *arr, const vector<size_t> &shape_ducc)
{
    void *real_data = mxGetData(arr);
    void *imag_data = mxIsComplex(arr) ? mxGetImagData(arr) : nullptr;
    
    mwSize ndim_matlab = mxGetNumberOfDimensions(arr);
    const mwSize *dims_matlab = mxGetDimensions(arr);
    
    // Calculate total elements
    size_t nelem = 1;
    for (size_t s : shape_ducc) {
        nelem *= s;
    }
    
    if (imag_data == nullptr) {
        // Real data - copy with dimension reordering
        T *real = static_cast<T *>(real_data);
        
        if (ndim_matlab == 1) {
            // 1D - just copy (no reordering needed)
            memcpy(real, buffer, nelem * sizeof(T));
        } else {
            // Multi-dimensional - need to reorder
            // Pre-compute MATLAB strides for faster indexing
            vector<size_t> strides_matlab(ndim_matlab);
            strides_matlab[0] = 1;
            for (mwSize i = 1; i < ndim_matlab; ++i) {
                strides_matlab[i] = strides_matlab[i-1] * dims_matlab[i-1];
            }
            
            vector<size_t> indices(shape_ducc.size(), 0);
            size_t ndim = shape_ducc.size();
            const T *buf = buffer;
            
            // Unroll the loop for small dimensions
            if (ndim == 2) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        indices[0] = idx0;
                        indices[1] = idx1;
                        size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                        real[idx_matlab] = *buf++;
                    }
                }
            } else if (ndim == 3) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                size_t idx2_max = shape_ducc[2];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        for (size_t idx2 = 0; idx2 < idx2_max; ++idx2) {
                            indices[0] = idx0;
                            indices[1] = idx1;
                            indices[2] = idx2;
                            size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                            real[idx_matlab] = *buf++;
                        }
                    }
                }
            } else {
                // General case
                for (size_t i = 0; i < nelem; ++i) {
                    size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                    real[idx_matlab] = buffer[i];
                    if (i < nelem - 1) {
                        incrementIndices(indices.data(), shape_ducc.data(), ndim);
                    }
                }
            }
        }
    } else {
        // Complex data - deinterleave real and imaginary parts
        using real_t = typename T::value_type;
        real_t *real = static_cast<real_t *>(real_data);
        real_t *imag = static_cast<real_t *>(imag_data);
        
        if (ndim_matlab == 1) {
            // 1D - just deinterleave (no reordering needed)
            // Optimized: use pointer arithmetic
            const T *b = buffer;
            real_t *r = real;
            real_t *i = imag;
            for (size_t j = 0; j < nelem; ++j) {
                *r++ = b->real();
                *i++ = b++->imag();
            }
        } else {
            // Multi-dimensional - need to reorder and deinterleave
            // Pre-compute MATLAB strides for faster indexing
            vector<size_t> strides_matlab(ndim_matlab);
            strides_matlab[0] = 1;
            for (mwSize i = 1; i < ndim_matlab; ++i) {
                strides_matlab[i] = strides_matlab[i-1] * dims_matlab[i-1];
            }
            
            vector<size_t> indices(shape_ducc.size(), 0);
            size_t ndim = shape_ducc.size();
            const T *buf = buffer;
            
            // Unroll the loop for small dimensions
            if (ndim == 2) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        indices[0] = idx0;
                        indices[1] = idx1;
                        size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                        real[idx_matlab] = buf->real();
                        imag[idx_matlab] = buf++->imag();
                    }
                }
            } else if (ndim == 3) {
                size_t idx0_max = shape_ducc[0];
                size_t idx1_max = shape_ducc[1];
                size_t idx2_max = shape_ducc[2];
                for (size_t idx0 = 0; idx0 < idx0_max; ++idx0) {
                    for (size_t idx1 = 0; idx1 < idx1_max; ++idx1) {
                        for (size_t idx2 = 0; idx2 < idx2_max; ++idx2) {
                            indices[0] = idx0;
                            indices[1] = idx1;
                            indices[2] = idx2;
                            size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                            real[idx_matlab] = buf->real();
                            imag[idx_matlab] = buf++->imag();
                        }
                    }
                }
            } else {
                // General case
                for (size_t i = 0; i < nelem; ++i) {
                    size_t idx_matlab = matlabLinearIndex(indices.data(), strides_matlab.data(), ndim_matlab);
                    real[idx_matlab] = buffer[i].real();
                    imag[idx_matlab] = buffer[i].imag();
                    if (i < nelem - 1) {
                        incrementIndices(indices.data(), shape_ducc.data(), ndim);
                    }
                }
            }
        }
    }
}

// Get optional parameter with default value
template<typename T>
T getOptionalParam(const mxArray *param, T default_value)
{
    if (param == nullptr || mxIsEmpty(param)) {
        return default_value;
    }
    return static_cast<T>(mxGetScalar(param));
}

// Get string parameter
inline string getStringParam(const mxArray *param, const string &default_value = "")
{
    if (param == nullptr || mxIsEmpty(param)) {
        return default_value;
    }
    
    if (!mxIsChar(param)) {
        MR_fail("Parameter must be a string");
    }
    
    char *str = mxArrayToString(param);
    if (str == nullptr) {
        MR_fail("Failed to convert parameter to string");
    }
    
    string result(str);
    mxFree(str);
    return result;
}

} // namespace ducc0_mex

#endif // DUCC0_MEX_UTILS_H
