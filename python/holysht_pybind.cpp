/*
 * HolySHT Python bindings via pybind11.
 *
 * Provides _alm2map and _map2alm that operate on numpy arrays and call
 * DUCC's ring-based spherical harmonic transforms.  The Python layer
 * (holysht/transforms.py) handles validation, ring geometry, and the
 * user-facing API.
 *
 * Batch transforms use DUCC's native synthesis_batch / adjoint_synthesis_batch
 * which parallelise over rings and batch items internally, reusing FFT plans
 * across the batch dimension.
 *
 * Unlike the MEX gateway, numpy arrays are already row-major with
 * interleaved complex, so we can pass data directly to DUCC.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ducc0/sht/sht.h"
#include "ducc0/infra/error_handling.h"

#include <vector>
#include <complex>
#include <array>
#include <stdexcept>

namespace py = pybind11;
using namespace ducc0;
using namespace std;

static vector<size_t> build_mstart(size_t lmax)
{
    vector<size_t> ms(lmax + 1);
    for (size_t m = 0; m <= lmax; ++m)
        ms[m] = m * (2 * lmax + 1 - m) / 2;
    return ms;
}

static size_t compute_npix(const vector<size_t> &nphi,
                           const vector<size_t> &ringstart)
{
    size_t mx = 0;
    for (size_t i = 0; i < nphi.size(); ++i) {
        size_t ring_end = ringstart[i] + nphi[i] - 1;
        if (ring_end > mx) mx = ring_end;
    }
    return mx + 1;
}

/* ------------------------------------------------------------------ */
/*  alm2map: synthesis (alm -> map)                                    */
/* ------------------------------------------------------------------ */

template<typename T>
static py::array alm2map_impl(
    py::array_t<complex<T>, py::array::c_style | py::array::forcecast> alm_arr,
    size_t lmax, size_t spin,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> nphi_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi0_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ringstart_arr,
    size_t nthreads)
{
    auto alm_info = alm_arr.request();
    size_t ndim = alm_info.ndim;

    bool is_batch;
    size_t N = 1, ncomp, nalm;
    if (ndim == 3) {
        is_batch = true;
        N     = alm_info.shape[0];
        ncomp = alm_info.shape[1];
        nalm  = alm_info.shape[2];
    } else if (ndim == 2) {
        is_batch = false;
        ncomp = alm_info.shape[0];
        nalm  = alm_info.shape[1];
    } else {
        throw std::invalid_argument("alm must be 2D [ncomp, nalm] or 3D [N, ncomp, nalm]");
    }

    size_t exp_ncomp = (spin == 0) ? 1 : 2;
    if (ncomp != exp_ncomp)
        throw std::invalid_argument("ncomp must be " + to_string(exp_ncomp) + " for spin " + to_string(spin));
    size_t exp_nalm = ((lmax + 1) * (lmax + 2)) / 2;
    if (nalm != exp_nalm)
        throw std::invalid_argument("nalm = " + to_string(nalm) +
            ", expected (lmax+1)*(lmax+2)/2 = " + to_string(exp_nalm));

    /* Ring geometry */
    size_t nrings = theta_arr.size();
    vector<double> theta(nrings), phi0(nrings), ringfactor(nrings, 1.0);
    vector<size_t> nphi(nrings), ringstart(nrings);
    auto th = theta_arr.unchecked<1>();
    auto np = nphi_arr.unchecked<1>();
    auto p0 = phi0_arr.unchecked<1>();
    auto rs = ringstart_arr.unchecked<1>();
    for (size_t i = 0; i < nrings; ++i) {
        theta[i]     = th(i);
        nphi[i]      = (size_t)np(i);
        phi0[i]      = p0(i);
        ringstart[i] = (size_t)rs(i);
    }

    size_t npix = compute_npix(nphi, ringstart);
    vector<size_t> mstart_vec = build_mstart(lmax);

    array<size_t,1> ring_shape  = {nrings};
    array<size_t,1> ms_shape    = {lmax + 1};
    cmav<double,1> theta_v(theta.data(), ring_shape);
    cmav<size_t,1> nphi_v(nphi.data(), ring_shape);
    cmav<double,1> phi0_v(phi0.data(), ring_shape);
    cmav<size_t,1> ringstart_v(ringstart.data(), ring_shape);
    cmav<double,1> ringfactor_v(ringfactor.data(), ring_shape);
    cmav<size_t,1> mstart_v(mstart_vec.data(), ms_shape);

    /* Allocate output */
    py::array_t<T> map_out;
    if (is_batch)
        map_out = py::array_t<T>({N, ncomp, npix});
    else
        map_out = py::array_t<T>({ncomp, npix});

    auto map_buf = map_out.mutable_unchecked();

    const complex<T> *alm_ptr = static_cast<const complex<T>*>(alm_info.ptr);
    T *map_ptr = static_cast<T*>(map_out.mutable_data());

    py::gil_scoped_release release;

    if (is_batch) {
        array<size_t,3> alm_shape3 = {N, ncomp, nalm};
        array<size_t,3> map_shape3 = {N, ncomp, npix};
        cmav<complex<T>,3> alm_v(alm_ptr, alm_shape3);
        vmav<T,3> map_v(map_ptr, map_shape3);
        synthesis_batch(alm_v, map_v, spin, lmax, mstart_v, 1,
                        theta_v, nphi_v, phi0_v, ringstart_v,
                        ringfactor_v, 1, nthreads, STANDARD, false);
    } else {
        array<size_t,2> alm_shape = {ncomp, nalm};
        array<size_t,2> map_shape = {ncomp, npix};
        cmav<complex<T>,2> alm_v(alm_ptr, alm_shape);
        vmav<T,2> map_v(map_ptr, map_shape);
        synthesis(alm_v, map_v, spin, lmax, mstart_v, 1,
                  theta_v, nphi_v, phi0_v, ringstart_v,
                  ringfactor_v, 1, nthreads, STANDARD, false);
    }

    return map_out;
}

/* ------------------------------------------------------------------ */
/*  map2alm: analysis (map -> alm)                                     */
/* ------------------------------------------------------------------ */

template<typename T>
static py::array map2alm_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> map_arr,
    size_t lmax, size_t spin,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> nphi_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> phi0_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> ringstart_arr,
    double weight, size_t n_iter, size_t nthreads)
{
    auto map_info = map_arr.request();
    size_t ndim = map_info.ndim;

    bool is_batch;
    size_t N = 1, ncomp, npix;
    if (ndim == 3) {
        is_batch = true;
        N     = map_info.shape[0];
        ncomp = map_info.shape[1];
        npix  = map_info.shape[2];
    } else if (ndim == 2) {
        is_batch = false;
        ncomp = map_info.shape[0];
        npix  = map_info.shape[1];
    } else {
        throw std::invalid_argument("map must be 2D [ncomp, npix] or 3D [N, ncomp, npix]");
    }

    size_t exp_ncomp = (spin == 0) ? 1 : 2;
    if (ncomp != exp_ncomp)
        throw std::invalid_argument("ncomp must be " + to_string(exp_ncomp) + " for spin " + to_string(spin));

    /* Ring geometry */
    size_t nrings = theta_arr.size();
    vector<double> theta(nrings), phi0(nrings), ringfactor(nrings, 1.0);
    vector<size_t> nphi_vec(nrings), ringstart(nrings);
    auto th = theta_arr.unchecked<1>();
    auto np = nphi_arr.unchecked<1>();
    auto p0 = phi0_arr.unchecked<1>();
    auto rs = ringstart_arr.unchecked<1>();
    for (size_t i = 0; i < nrings; ++i) {
        theta[i]       = th(i);
        nphi_vec[i]    = (size_t)np(i);
        phi0[i]        = p0(i);
        ringstart[i]   = (size_t)rs(i);
    }

    size_t nalm = ((lmax + 1) * (lmax + 2)) / 2;
    vector<size_t> mstart_vec = build_mstart(lmax);

    array<size_t,1> ring_shape  = {nrings};
    array<size_t,1> ms_shape    = {lmax + 1};
    cmav<double,1> theta_v(theta.data(), ring_shape);
    cmav<size_t,1> nphi_v(nphi_vec.data(), ring_shape);
    cmav<double,1> phi0_v(phi0.data(), ring_shape);
    cmav<size_t,1> ringstart_v(ringstart.data(), ring_shape);
    cmav<double,1> ringfactor_v(ringfactor.data(), ring_shape);
    cmav<size_t,1> mstart_v(mstart_vec.data(), ms_shape);

    /* Allocate output */
    py::array_t<complex<T>> alm_out;
    if (is_batch)
        alm_out = py::array_t<complex<T>>({N, ncomp, nalm});
    else
        alm_out = py::array_t<complex<T>>({ncomp, nalm});

    const T *map_ptr = static_cast<const T*>(map_info.ptr);
    complex<T> *alm_ptr = static_cast<complex<T>*>(alm_out.mutable_data());

    const size_t map_stride = ncomp * npix;
    const size_t alm_stride = ncomp * nalm;

    T w = static_cast<T>(weight);

    /* Scratch buffers */
    const size_t total_map = N * map_stride;
    const size_t total_alm = N * alm_stride;
    vector<T> map_scratch(total_map);
    vector<complex<T>> dalm(total_alm);

    py::gil_scoped_release release;

    if (is_batch) {
        array<size_t,3> map_shape3 = {N, ncomp, npix};
        array<size_t,3> alm_shape3 = {N, ncomp, nalm};

        /* Initial adjoint: alm = A^T(W * map) */
        for (size_t i = 0; i < total_map; ++i)
            map_scratch[i] = map_ptr[i] * w;
        {
            cmav<T,3> mw_view(map_scratch.data(), map_shape3);
            vmav<complex<T>,3> alm_view(alm_ptr, alm_shape3);
            adjoint_synthesis_batch(alm_view, mw_view, spin, lmax, mstart_v, 1,
                                    theta_v, nphi_v, phi0_v, ringstart_v,
                                    ringfactor_v, 1, nthreads, STANDARD, false);
        }

        /* Jacobi iterations */
        for (size_t it = 0; it < n_iter; ++it) {
            {
                cmav<complex<T>,3> alm_cv(alm_ptr, alm_shape3);
                vmav<T,3> ms_view(map_scratch.data(), map_shape3);
                synthesis_batch(alm_cv, ms_view, spin, lmax, mstart_v, 1,
                                theta_v, nphi_v, phi0_v, ringstart_v,
                                ringfactor_v, 1, nthreads, STANDARD, false);
            }
            for (size_t i = 0; i < total_map; ++i)
                map_scratch[i] = (map_ptr[i] - map_scratch[i]) * w;
            {
                cmav<T,3> dmap_view(map_scratch.data(), map_shape3);
                vmav<complex<T>,3> dalm_view(dalm.data(), alm_shape3);
                adjoint_synthesis_batch(dalm_view, dmap_view, spin, lmax, mstart_v, 1,
                                        theta_v, nphi_v, phi0_v, ringstart_v,
                                        ringfactor_v, 1, nthreads, STANDARD, false);
            }
            for (size_t i = 0; i < total_alm; ++i)
                alm_ptr[i] += dalm[i];
        }
    } else {
        array<size_t,2> map_shape = {ncomp, npix};
        array<size_t,2> alm_shape = {ncomp, nalm};

        /* Initial adjoint: alm = A^T(W * map) */
        for (size_t i = 0; i < total_map; ++i)
            map_scratch[i] = map_ptr[i] * w;
        {
            cmav<T,2> mw_view(map_scratch.data(), map_shape);
            vmav<complex<T>,2> alm_view(alm_ptr, alm_shape);
            adjoint_synthesis(alm_view, mw_view, spin, lmax, mstart_v, 1,
                              theta_v, nphi_v, phi0_v, ringstart_v,
                              ringfactor_v, 1, nthreads, STANDARD, false);
        }

        /* Jacobi iterations */
        for (size_t it = 0; it < n_iter; ++it) {
            {
                cmav<complex<T>,2> alm_cv(alm_ptr, alm_shape);
                vmav<T,2> ms_view(map_scratch.data(), map_shape);
                synthesis(alm_cv, ms_view, spin, lmax, mstart_v, 1,
                          theta_v, nphi_v, phi0_v, ringstart_v,
                          ringfactor_v, 1, nthreads, STANDARD, false);
            }
            for (size_t i = 0; i < total_map; ++i)
                map_scratch[i] = (map_ptr[i] - map_scratch[i]) * w;
            {
                cmav<T,2> dmap_view(map_scratch.data(), map_shape);
                vmav<complex<T>,2> dalm_view(dalm.data(), alm_shape);
                adjoint_synthesis(dalm_view, dmap_view, spin, lmax, mstart_v, 1,
                                  theta_v, nphi_v, phi0_v, ringstart_v,
                                  ringfactor_v, 1, nthreads, STANDARD, false);
            }
            for (size_t i = 0; i < total_alm; ++i)
                alm_ptr[i] += dalm[i];
        }
    }

    return alm_out;
}

/* ------------------------------------------------------------------ */
/*  Module definition                                                   */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(_holysht_core, m) {
    m.doc() = "HolySHT: fast HEALPix spherical harmonic transforms (C++ backend)";

    m.def("alm2map_f64", &alm2map_impl<double>,
          py::arg("alm"), py::arg("lmax"), py::arg("spin"),
          py::arg("theta"), py::arg("nphi"), py::arg("phi0"),
          py::arg("ringstart"), py::arg("nthreads"),
          "Synthesis (alm -> map), double precision");

    m.def("alm2map_f32", &alm2map_impl<float>,
          py::arg("alm"), py::arg("lmax"), py::arg("spin"),
          py::arg("theta"), py::arg("nphi"), py::arg("phi0"),
          py::arg("ringstart"), py::arg("nthreads"),
          "Synthesis (alm -> map), single precision");

    m.def("map2alm_f64", &map2alm_impl<double>,
          py::arg("map"), py::arg("lmax"), py::arg("spin"),
          py::arg("theta"), py::arg("nphi"), py::arg("phi0"),
          py::arg("ringstart"), py::arg("weight"),
          py::arg("n_iter"), py::arg("nthreads"),
          "Analysis (map -> alm), double precision");

    m.def("map2alm_f32", &map2alm_impl<float>,
          py::arg("map"), py::arg("lmax"), py::arg("spin"),
          py::arg("theta"), py::arg("nphi"), py::arg("phi0"),
          py::arg("ringstart"), py::arg("weight"),
          py::arg("n_iter"), py::arg("nthreads"),
          "Analysis (map -> alm), single precision");
}
