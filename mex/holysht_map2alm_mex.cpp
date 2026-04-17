/*
 * holysht_map2alm MEX gateway.
 *
 * Iterative map2alm (adjoint_synthesis + Jacobi refinement) on a set of
 * HEALPix-style rings (ring-based geometry).  Supports single-map and
 * batch (3-D) inputs in both double and single precision.  Spin must be
 * 0 or 2 (T or Q/U).
 *
 * Batch transforms use DUCC's native synthesis_batch / adjoint_synthesis_batch
 * which parallelise over rings and batch items internally, reusing FFT plans
 * across the batch dimension.
 *
 * Usage:
 *   alm = holysht_map2alm_mex(map, lmax, spin,
 *             theta, nphi, phi0, ringstart,
 *             weight, n_iter, nthreads)
 *
 * Inputs:
 *   map:       real array [ncomp, npix] or [N, ncomp, npix] (batch)
 *   lmax:      maximum multipole order (mmax = lmax)
 *   spin:      0 or 2
 *   theta:     colatitudes of rings              [nrings] (double)
 *   nphi:      pixels per ring                   [nrings] (double, integer-valued)
 *   phi0:      azimuth of first pixel per ring   [nrings] (double)
 *   ringstart: 0-based start index per ring      [nrings] (double, integer-valued)
 *   weight:    scalar quadrature weight (HEALPix: 4*pi/npix_full)
 *   n_iter:    number of Jacobi iterations (0 = adjoint only)
 *   nthreads:  number of threads (0 = std::thread::hardware_concurrency)
 *
 * Output:
 *   alm: complex array [ncomp, nalm] or [N, ncomp, nalm]
 *        where nalm = (lmax+1)*(lmax+2)/2
 */

#include "mex.h"
#include "ducc0_mex_utils.h"
#include "ducc0/sht/sht.h"
#include "ducc0/infra/error_handling.h"
#include <vector>
#include <complex>
#include <array>

using namespace ducc0;
using namespace ducc0_mex;
using namespace std;

static vector<size_t> build_mstart(size_t lmax)
{
    vector<size_t> ms(lmax + 1);
    for (size_t m = 0; m <= lmax; ++m)
        ms[m] = m * (2 * lmax + 1 - m) / 2;
    return ms;
}

static size_t min_mapdim(const vector<size_t> &nphi,
                         const vector<size_t> &ringstart)
{
    size_t mx = 0;
    for (size_t i = 0; i < nphi.size(); ++i) {
        size_t ring_end = ringstart[i] + nphi[i] - 1;
        if (ring_end > mx) mx = ring_end;
    }
    return mx + 1;
}

template<typename T>
static void map2alm_impl(
    const mxArray *map_arr, mxArray *&alm_out,
    size_t lmax, size_t spin,
    const vector<size_t> &mstart_vec,
    const cmav<double,1> &theta_v, const cmav<size_t,1> &nphi_v,
    const cmav<double,1> &phi0_v, const cmav<size_t,1> &ringstart_v,
    const cmav<double,1> &ringfactor_v,
    T weight, size_t n_iter, size_t nthreads,
    bool is_batch,
    size_t N, size_t ncomp, size_t nalm, size_t npix,
    mxClassID class_id)
{
    const size_t map_stride = ncomp * npix;
    const size_t alm_stride = ncomp * nalm;

    array<size_t,1> ms_shape = {lmax + 1};
    cmav<size_t,1> mstart_v(mstart_vec.data(), ms_shape);

    /* Copy map: MATLAB col-major -> row-major */
    vector<T> map_buf(N * map_stride);
    {
        const T *real_data;
        if (class_id == mxDOUBLE_CLASS)
            real_data = (const T *)mxGetPr(map_arr);
        else
            real_data = (const T *)mxGetData(map_arr);

        if (is_batch) {
            for (size_t ib = 0; ib < N; ++ib)
                for (size_t ic = 0; ic < ncomp; ++ic)
                    for (size_t ip = 0; ip < npix; ++ip) {
                        size_t idx_ml = ib + ic * N + ip * N * ncomp;
                        size_t idx_rm = ib * map_stride + ic * npix + ip;
                        map_buf[idx_rm] = real_data[idx_ml];
                    }
        } else {
            for (size_t ic = 0; ic < ncomp; ++ic)
                for (size_t ip = 0; ip < npix; ++ip) {
                    size_t idx_ml = ic + ip * ncomp;
                    size_t idx_rm = ic * npix + ip;
                    map_buf[idx_rm] = real_data[idx_ml];
                }
        }
    }

    vector<complex<T>> alm_buf(N * alm_stride, complex<T>(0));
    vector<T> map_scratch(N * map_stride);
    vector<complex<T>> dalm(N * alm_stride);
    const size_t total_map = N * map_stride;
    const size_t total_alm = N * alm_stride;

    if (is_batch) {
        array<size_t,3> map_shape3 = {N, ncomp, npix};
        array<size_t,3> alm_shape3 = {N, ncomp, nalm};

        /* Initial adjoint: alm = A^T(W * map) */
        for (size_t i = 0; i < total_map; ++i)
            map_scratch[i] = map_buf[i] * weight;
        {
            cmav<T,3> mw_view(map_scratch.data(), map_shape3);
            vmav<complex<T>,3> alm_view(alm_buf.data(), alm_shape3);
            adjoint_synthesis_batch(alm_view, mw_view, spin, lmax, mstart_v, 1,
                                    theta_v, nphi_v, phi0_v, ringstart_v,
                                    ringfactor_v, 1, nthreads, STANDARD, false);
        }

        /* Jacobi iterations */
        for (size_t it = 0; it < n_iter; ++it) {
            /* Forward: map_scratch = A * alm */
            {
                cmav<complex<T>,3> alm_cv(alm_buf.data(), alm_shape3);
                vmav<T,3> ms_view(map_scratch.data(), map_shape3);
                synthesis_batch(alm_cv, ms_view, spin, lmax, mstart_v, 1,
                                theta_v, nphi_v, phi0_v, ringstart_v,
                                ringfactor_v, 1, nthreads, STANDARD, false);
            }
            /* Weighted residual: map_scratch = W * (map - A*alm) */
            for (size_t i = 0; i < total_map; ++i)
                map_scratch[i] = (map_buf[i] - map_scratch[i]) * weight;
            /* Adjoint of residual: dalm = A^T(W * residual) */
            {
                cmav<T,3> dmap_view(map_scratch.data(), map_shape3);
                vmav<complex<T>,3> dalm_view(dalm.data(), alm_shape3);
                adjoint_synthesis_batch(dalm_view, dmap_view, spin, lmax, mstart_v, 1,
                                        theta_v, nphi_v, phi0_v, ringstart_v,
                                        ringfactor_v, 1, nthreads, STANDARD, false);
            }
            /* Update: alm += dalm */
            for (size_t i = 0; i < total_alm; ++i)
                alm_buf[i] += dalm[i];
        }
    } else {
        array<size_t,2> map_shape = {ncomp, npix};
        array<size_t,2> alm_shape = {ncomp, nalm};

        /* Initial adjoint: alm = A^T(W * map) */
        for (size_t i = 0; i < total_map; ++i)
            map_scratch[i] = map_buf[i] * weight;
        {
            cmav<T,2> mw_view(map_scratch.data(), map_shape);
            vmav<complex<T>,2> alm_view(alm_buf.data(), alm_shape);
            adjoint_synthesis(alm_view, mw_view, spin, lmax, mstart_v, 1,
                              theta_v, nphi_v, phi0_v, ringstart_v,
                              ringfactor_v, 1, nthreads, STANDARD, false);
        }

        /* Jacobi iterations */
        for (size_t it = 0; it < n_iter; ++it) {
            {
                cmav<complex<T>,2> alm_cv(alm_buf.data(), alm_shape);
                vmav<T,2> ms_view(map_scratch.data(), map_shape);
                synthesis(alm_cv, ms_view, spin, lmax, mstart_v, 1,
                          theta_v, nphi_v, phi0_v, ringstart_v,
                          ringfactor_v, 1, nthreads, STANDARD, false);
            }
            for (size_t i = 0; i < total_map; ++i)
                map_scratch[i] = (map_buf[i] - map_scratch[i]) * weight;
            {
                cmav<T,2> dmap_view(map_scratch.data(), map_shape);
                vmav<complex<T>,2> dalm_view(dalm.data(), alm_shape);
                adjoint_synthesis(dalm_view, dmap_view, spin, lmax, mstart_v, 1,
                                  theta_v, nphi_v, phi0_v, ringstart_v,
                                  ringfactor_v, 1, nthreads, STANDARD, false);
            }
            for (size_t i = 0; i < total_alm; ++i)
                alm_buf[i] += dalm[i];
        }
    }

    /* Copy alm to MATLAB output (complex, col-major, split real/imag) */
    if (is_batch) {
        mwSize out_dims[3] = {(mwSize)N, (mwSize)ncomp, (mwSize)nalm};
        alm_out = mxCreateNumericArray(3, out_dims, class_id, mxCOMPLEX);
    } else {
        mwSize out_dims[2] = {(mwSize)ncomp, (mwSize)nalm};
        alm_out = mxCreateNumericArray(2, out_dims, class_id, mxCOMPLEX);
    }

    T *out_re, *out_im;
    if (class_id == mxDOUBLE_CLASS) {
        out_re = (T *)mxGetPr(alm_out);
        out_im = (T *)mxGetPi(alm_out);
    } else {
        out_re = (T *)mxGetData(alm_out);
        out_im = (T *)mxGetImagData(alm_out);
    }

    if (is_batch) {
        for (size_t ib = 0; ib < N; ++ib)
            for (size_t ic = 0; ic < ncomp; ++ic)
                for (size_t ia = 0; ia < nalm; ++ia) {
                    size_t idx_rm = ib * alm_stride + ic * nalm + ia;
                    size_t idx_ml = ib + ic * N + ia * N * ncomp;
                    out_re[idx_ml] = alm_buf[idx_rm].real();
                    out_im[idx_ml] = alm_buf[idx_rm].imag();
                }
    } else {
        for (size_t ic = 0; ic < ncomp; ++ic)
            for (size_t ia = 0; ia < nalm; ++ia) {
                size_t idx_rm = ic * nalm + ia;
                size_t idx_ml = ic + ia * ncomp;
                out_re[idx_ml] = alm_buf[idx_rm].real();
                out_im[idx_ml] = alm_buf[idx_rm].imag();
            }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    try {
        if (nrhs < 10) {
            mexErrMsgIdAndTxt("holysht:map2alm:InputError",
                "10 inputs required: map, lmax, spin, theta, nphi, phi0, "
                "ringstart, weight, n_iter, nthreads");
        }

        const mxArray *map_arr = prhs[0];
        if (mxIsEmpty(map_arr) || mxIsComplex(map_arr))
            mexErrMsgIdAndTxt("holysht:map2alm:InputError",
                "map must be a non-empty real array");

        size_t lmax     = (size_t)mxGetScalar(prhs[1]);
        size_t spin     = (size_t)mxGetScalar(prhs[2]);
        double weight_d = mxGetScalar(prhs[7]);
        size_t n_iter   = (size_t)mxGetScalar(prhs[8]);
        size_t nthreads = (size_t)mxGetScalar(prhs[9]);

        if (spin != 0 && spin != 2)
            mexErrMsgIdAndTxt("holysht:map2alm:InputError",
                "spin must be 0 or 2");

        size_t ncomp = (spin == 0) ? 1 : 2;

        const mxArray *theta_arr     = prhs[3];
        const mxArray *nphi_arr      = prhs[4];
        const mxArray *phi0_arr      = prhs[5];
        const mxArray *ringstart_arr = prhs[6];

        size_t nrings = mxGetNumberOfElements(theta_arr);

        vector<double> theta(nrings);
        vector<size_t> nphi(nrings);
        vector<double> phi0(nrings);
        vector<size_t> ringstart(nrings);
        vector<double> ringfactor(nrings, 1.0);

        const double *td = mxGetPr(theta_arr);
        const double *nd = mxGetPr(nphi_arr);
        const double *pd = mxGetPr(phi0_arr);
        const double *rd = mxGetPr(ringstart_arr);
        for (size_t i = 0; i < nrings; ++i) {
            theta[i]     = td[i];
            nphi[i]      = (size_t)nd[i];
            phi0[i]      = pd[i];
            ringstart[i] = (size_t)rd[i];
        }

        size_t npix = min_mapdim(nphi, ringstart);
        vector<size_t> mstart_vec = build_mstart(lmax);
        size_t nalm = ((lmax + 1) * (lmax + 2)) / 2;

        array<size_t,1> ring_shape = {nrings};
        cmav<double,1> theta_v(theta.data(), ring_shape);
        cmav<size_t,1> nphi_v(nphi.data(), ring_shape);
        cmav<double,1> phi0_v(phi0.data(), ring_shape);
        cmav<size_t,1> ringstart_v(ringstart.data(), ring_shape);
        cmav<double,1> ringfactor_v(ringfactor.data(), ring_shape);

        mwSize ndim = mxGetNumberOfDimensions(map_arr);
        const mwSize *dims = mxGetDimensions(map_arr);
        bool is_batch = (ndim == 3);
        size_t N = 1;

        if (is_batch) {
            N = dims[0];
            if ((size_t)dims[1] != ncomp || (size_t)dims[2] != npix)
                mexErrMsgIdAndTxt("holysht:map2alm:InputError",
                    "Batch map must be [N, %zu, %zu]", ncomp, npix);
        } else if (ndim == 2) {
            if ((size_t)dims[0] != ncomp || (size_t)dims[1] != npix)
                mexErrMsgIdAndTxt("holysht:map2alm:InputError",
                    "Map must be [%zu, %zu]", ncomp, npix);
        } else {
            mexErrMsgIdAndTxt("holysht:map2alm:InputError",
                "Map must be 2D [ncomp, npix] or 3D [N, ncomp, npix]");
        }

        mxClassID class_id = mxGetClassID(map_arr);

        if (class_id == mxDOUBLE_CLASS) {
            map2alm_impl<double>(map_arr, plhs[0], lmax, spin,
                mstart_vec, theta_v, nphi_v, phi0_v, ringstart_v,
                ringfactor_v, (double)weight_d, n_iter, nthreads,
                is_batch, N, ncomp, nalm, npix, class_id);
        } else if (class_id == mxSINGLE_CLASS) {
            map2alm_impl<float>(map_arr, plhs[0], lmax, spin,
                mstart_vec, theta_v, nphi_v, phi0_v, ringstart_v,
                ringfactor_v, (float)weight_d, n_iter, nthreads,
                is_batch, N, ncomp, nalm, npix, class_id);
        } else {
            mexErrMsgIdAndTxt("holysht:map2alm:TypeError",
                "Only double and single precision supported.");
        }

    } catch (const exception &e) {
        handleDuccError(e);
    }
}
