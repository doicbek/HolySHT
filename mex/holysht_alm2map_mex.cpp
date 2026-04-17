/*
 * holysht_alm2map MEX gateway.
 *
 * Synthesis (alm -> map) on a set of HEALPix-style rings.  Supports
 * single-map and batch (3-D) inputs in both double and single precision.
 * Spin must be 0 or 2.
 *
 * Usage:
 *   map = holysht_alm2map_mex(alm, lmax, spin,
 *             theta, nphi, phi0, ringstart, nthreads)
 *
 * Inputs:
 *   alm:       complex array [ncomp, nalm] or [N, ncomp, nalm]
 *   lmax:      maximum multipole order (mmax = lmax)
 *   spin:      0 or 2
 *   theta, nphi, phi0, ringstart: ring geometry (double, as in map2alm)
 *   nthreads:  number of threads (0 = std::thread::hardware_concurrency)
 *
 * Output:
 *   map: real array [ncomp, npix] or [N, ncomp, npix]
 */

#include "mex.h"
#include "ducc0_mex_utils.h"
#include "ducc0/sht/sht.h"
#include "ducc0/infra/error_handling.h"
#include <vector>
#include <complex>
#include <array>
#include <thread>
#include <algorithm>

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
static void alm2map_impl(
    const mxArray *alm_arr, mxArray *&map_out,
    size_t lmax, size_t spin,
    const vector<size_t> &mstart_vec,
    const cmav<double,1> &theta_v, const cmav<size_t,1> &nphi_v,
    const cmav<double,1> &phi0_v, const cmav<size_t,1> &ringstart_v,
    const cmav<double,1> &ringfactor_v,
    size_t nthreads,
    bool is_batch,
    size_t N, size_t ncomp, size_t nalm, size_t npix,
    mxClassID class_id)
{
    array<size_t,1> ms_shape = {lmax + 1};
    cmav<size_t,1> mstart_v(mstart_vec.data(), ms_shape);

    const size_t alm_stride = ncomp * nalm;
    const size_t map_stride = ncomp * npix;

    /* Copy alm: MATLAB col-major split real/imag -> row-major interleaved */
    vector<complex<T>> alm_buf(N * alm_stride);
    {
        const T *pr, *pi;
        if (class_id == mxDOUBLE_CLASS) {
            pr = (const T *)mxGetPr(alm_arr);
            pi = (const T *)mxGetPi(alm_arr);
        } else {
            pr = (const T *)mxGetData(alm_arr);
            pi = (const T *)mxGetImagData(alm_arr);
        }
        bool has_imag = (pi != nullptr);

        if (is_batch) {
            for (size_t ib = 0; ib < N; ++ib)
                for (size_t ic = 0; ic < ncomp; ++ic)
                    for (size_t ia = 0; ia < nalm; ++ia) {
                        size_t idx_ml = ib + ic * N + ia * N * ncomp;
                        size_t idx_rm = ib * alm_stride + ic * nalm + ia;
                        T re = pr[idx_ml];
                        T im = has_imag ? pi[idx_ml] : T(0);
                        alm_buf[idx_rm] = complex<T>(re, im);
                    }
        } else {
            /* 2D [ncomp, nalm] col-major */
            for (size_t ic = 0; ic < ncomp; ++ic)
                for (size_t ia = 0; ia < nalm; ++ia) {
                    size_t idx_ml = ic + ia * ncomp;
                    size_t idx_rm = ic * nalm + ia;
                    T re = pr[idx_ml];
                    T im = has_imag ? pi[idx_ml] : T(0);
                    alm_buf[idx_rm] = complex<T>(re, im);
                }
        }
    }

    vector<T> map_buf(N * map_stride);

    if (is_batch) {
        array<size_t,3> alm_shape3 = {N, ncomp, nalm};
        array<size_t,3> map_shape3 = {N, ncomp, npix};
        cmav<complex<T>,3> alm_v(alm_buf.data(), alm_shape3);
        vmav<T,3> map_v(map_buf.data(), map_shape3);
        synthesis_batch(alm_v, map_v, spin, lmax, mstart_v, 1,
                        theta_v, nphi_v, phi0_v, ringstart_v,
                        ringfactor_v, 1, nthreads, STANDARD, false);
    } else {
        array<size_t,2> alm_shape = {ncomp, nalm};
        array<size_t,2> map_shape = {ncomp, npix};
        cmav<complex<T>,2> alm_v(alm_buf.data(), alm_shape);
        vmav<T,2> map_v(map_buf.data(), map_shape);
        synthesis(alm_v, map_v, spin, lmax, mstart_v, 1,
                  theta_v, nphi_v, phi0_v, ringstart_v,
                  ringfactor_v, 1, nthreads, STANDARD, false);
    }

    /* Output: matches input dimensionality */
    if (is_batch) {
        mwSize out_dims[3] = {(mwSize)N, (mwSize)ncomp, (mwSize)npix};
        map_out = mxCreateNumericArray(3, out_dims, class_id, mxREAL);
    } else {
        mwSize out_dims[2] = {(mwSize)ncomp, (mwSize)npix};
        map_out = mxCreateNumericArray(2, out_dims, class_id, mxREAL);
    }
    T *out_data = (class_id == mxDOUBLE_CLASS)
        ? (T *)mxGetPr(map_out)
        : (T *)mxGetData(map_out);

    if (is_batch) {
        for (size_t ib = 0; ib < N; ++ib)
            for (size_t ic = 0; ic < ncomp; ++ic)
                for (size_t ip = 0; ip < npix; ++ip) {
                    size_t idx_rm = ib * map_stride + ic * npix + ip;
                    size_t idx_ml = ib + ic * N + ip * N * ncomp;
                    out_data[idx_ml] = map_buf[idx_rm];
                }
    } else {
        for (size_t ic = 0; ic < ncomp; ++ic)
            for (size_t ip = 0; ip < npix; ++ip) {
                size_t idx_rm = ic * npix + ip;
                size_t idx_ml = ic + ip * ncomp;
                out_data[idx_ml] = map_buf[idx_rm];
            }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    try {
        if (nrhs < 8) {
            mexErrMsgIdAndTxt("holysht:alm2map:InputError",
                "8 inputs required: alm, lmax, spin, theta, nphi, phi0, "
                "ringstart, nthreads");
        }

        const mxArray *alm_arr = prhs[0];
        if (mxIsEmpty(alm_arr) || !mxIsComplex(alm_arr))
            mexErrMsgIdAndTxt("holysht:alm2map:InputError",
                "alm must be a non-empty complex array");

        size_t lmax     = (size_t)mxGetScalar(prhs[1]);
        size_t spin     = (size_t)mxGetScalar(prhs[2]);
        size_t nthreads = (size_t)mxGetScalar(prhs[7]);

        if (spin != 0 && spin != 2)
            mexErrMsgIdAndTxt("holysht:alm2map:InputError",
                "spin must be 0 or 2");
        size_t exp_ncomp = (spin == 0) ? 1 : 2;

        mwSize ndim = mxGetNumberOfDimensions(alm_arr);
        const mwSize *dims = mxGetDimensions(alm_arr);
        bool is_batch;
        size_t N = 1, ncomp, nalm;
        if (ndim == 3) {
            is_batch = true;
            N     = dims[0];
            ncomp = dims[1];
            nalm  = dims[2];
        } else if (ndim == 2) {
            is_batch = false;
            ncomp = dims[0];
            nalm  = dims[1];
        } else {
            mexErrMsgIdAndTxt("holysht:alm2map:InputError",
                "alm must be 2D [ncomp, nalm] or 3D [N, ncomp, nalm]");
        }
        if (ncomp != exp_ncomp)
            mexErrMsgIdAndTxt("holysht:alm2map:InputError",
                "ncomp dimension must equal %zu for spin %zu", exp_ncomp, spin);
        size_t exp_nalm = ((lmax + 1) * (lmax + 2)) / 2;
        if (nalm != exp_nalm)
            mexErrMsgIdAndTxt("holysht:alm2map:InputError",
                "nalm = %zu, expected (lmax+1)*(lmax+2)/2 = %zu", nalm, exp_nalm);

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

        array<size_t,1> ring_shape = {nrings};
        cmav<double,1> theta_v(theta.data(), ring_shape);
        cmav<size_t,1> nphi_v(nphi.data(), ring_shape);
        cmav<double,1> phi0_v(phi0.data(), ring_shape);
        cmav<size_t,1> ringstart_v(ringstart.data(), ring_shape);
        cmav<double,1> ringfactor_v(ringfactor.data(), ring_shape);

        mxClassID class_id = mxGetClassID(alm_arr);

        if (class_id == mxDOUBLE_CLASS) {
            alm2map_impl<double>(alm_arr, plhs[0], lmax, spin,
                mstart_vec, theta_v, nphi_v, phi0_v, ringstart_v,
                ringfactor_v, nthreads, is_batch,
                N, ncomp, nalm, npix, class_id);
        } else if (class_id == mxSINGLE_CLASS) {
            alm2map_impl<float>(alm_arr, plhs[0], lmax, spin,
                mstart_vec, theta_v, nphi_v, phi0_v, ringstart_v,
                ringfactor_v, nthreads, is_batch,
                N, ncomp, nalm, npix, class_id);
        } else {
            mexErrMsgIdAndTxt("holysht:alm2map:TypeError",
                "Only double and single precision supported.");
        }

    } catch (const exception &e) {
        handleDuccError(e);
    }
}
