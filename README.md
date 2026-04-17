# HolySHT

Minimal MATLAB/MEX wrapper for HEALPix spherical harmonic transforms, built on
a trimmed copy of the [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) C++ library.

Supports spin-0 (temperature) and spin-2 (polarization) transforms on HEALPix
grids, with optional partial-sky latitude bands, batch mode, and single/double
precision.

## Building

Requirements: MATLAB R2018b+, a C++17 compiler (GCC 7+, Clang 6+), CMake 3.15+.

```bash
cd HolySHT/mex
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
```

If MATLAB is not on your `PATH`, point CMake to it:

```bash
cmake -DMATLAB_ROOT=/usr/local/MATLAB/R2024a ..
```

## Setup

From MATLAB, run the setup script once per session (or add it to your
`startup.m`):

```matlab
run('/path/to/HolySHT/setup_holysht.m')
```

This adds the `+holysht` package and `mex/build/` directories to
your MATLAB path.

## API Reference

### `holysht.alm2map` -- Spherical harmonic synthesis (alm to map)

```matlab
map = holysht.alm2map(alm, spin, nside, lmax)
map = holysht.alm2map(alm, spin, nside, lmax, 'Name', Value, ...)
```

**Inputs:**

| Argument | Type | Description |
|----------|------|-------------|
| `alm` | complex `[ncomp, nalm]` or `[N, ncomp, nalm]` | Spherical harmonic coefficients. `nalm = (lmax+1)*(lmax+2)/2`. |
| `spin` | `0` or `2` | Spin weight. `ncomp = 1` for spin-0, `ncomp = 2` for spin-2. |
| `nside` | positive integer | HEALPix resolution parameter. |
| `lmax` | non-negative integer | Maximum multipole. |

**Name-value options:**

| Name | Default | Description |
|------|---------|-------------|
| `'lat_range'` | `[]` (full sky) | `[lat_min, lat_max]` in degrees (+90 = north pole). Restricts output to rings within the band. |
| `'nthreads'` | `0` (auto) | Number of threads. |

**Output:** real `[ncomp, npix]` or `[N, ncomp, npix]`, matching input
precision. For full sky, `npix = 12*nside^2`. With `lat_range`, `npix` is
the total pixel count of selected rings.

---

### `holysht.map2alm` -- Spherical harmonic analysis (map to alm)

```matlab
alm = holysht.map2alm(map, spin, nside, lmax)
alm = holysht.map2alm(map, spin, nside, lmax, 'Name', Value, ...)
```

**Inputs:**

| Argument | Type | Description |
|----------|------|-------------|
| `map` | real `[ncomp, npix]` or `[N, ncomp, npix]` | HEALPix map(s) in ring order. |
| `spin` | `0` or `2` | Spin weight. |
| `nside` | positive integer | HEALPix resolution parameter. |
| `lmax` | non-negative integer | Maximum multipole. |

**Name-value options:**

| Name | Default | Description |
|------|---------|-------------|
| `'n_iter'` | `3` | Number of Jacobi refinement iterations. More iterations improve accuracy (10 gives ~1e-12 relative error). |
| `'lat_range'` | `[]` (full sky) | `[lat_min, lat_max]` in degrees. The input map must contain only pixels from the selected rings, packed contiguously. |
| `'nthreads'` | `0` (auto) | Number of threads. |

**Output:** complex `[ncomp, nalm]` or `[N, ncomp, nalm]`, matching
input precision.

---

### alm ordering convention

Coefficients use DUCC's `mstart` layout: for a given `(l, m)`, the
flat index is `mstart(m) + l`, where `mstart(m) = m * (2*lmax + 1 - m) / 2`.
This is a triangular layout with all `l` values for each `m` stored
contiguously. The total number of coefficients is
`nalm = (lmax + 1) * (lmax + 2) / 2`.

For spin-0, `ncomp = 1` and the alm array is `[1, nalm]`.
For spin-2, `ncomp = 2` and the alm array is `[2, nalm]` (E-modes in row 1,
B-modes in row 2).

## Examples

### Spin-0 round-trip

```matlab
run('setup_holysht.m');

nside = 128;
lmax  = 2 * nside;
nalm  = (lmax + 1) * (lmax + 2) / 2;

% Generate random alm (m=0 modes must be real for physical fields)
rng(42);
alm = randn(1, nalm) + 1i * randn(1, nalm);
alm(1:lmax+1) = real(alm(1:lmax+1));  % m=0 modes are real

% Synthesize map and recover alm
map  = holysht.alm2map(alm, 0, nside, lmax);
alm2 = holysht.map2alm(map, 0, nside, lmax, 'n_iter', 10);

err = norm(alm(:) - alm2(:)) / norm(alm(:));
fprintf('Round-trip error: %.2e\n', err);  % ~1e-12
```

### Spin-2 (polarization)

```matlab
nside = 64;
lmax  = 2 * nside;
nalm  = (lmax + 1) * (lmax + 2) / 2;

% alm is [2, nalm]: row 1 = E-modes, row 2 = B-modes
alm_EB = randn(2, nalm) + 1i * randn(2, nalm);
alm_EB(:, 1:lmax+1) = real(alm_EB(:, 1:lmax+1));  % m=0 real
% Zero unphysical modes: l < |spin| = 2
alm_EB(:, 1) = 0;  % l=0, m=0
alm_EB(:, 2) = 0;  % l=1, m=0
alm_EB(:, lmax+2) = 0;  % l=1, m=1

% map is [2, npix]: row 1 = Q, row 2 = U
map_QU = holysht.alm2map(alm_EB, 2, nside, lmax);
alm_rec = holysht.map2alm(map_QU, 2, nside, lmax, 'n_iter', 10);

err = norm(alm_EB(:) - alm_rec(:)) / norm(alm_EB(:));
fprintf('Spin-2 round-trip error: %.2e\n', err);  % ~1e-12
```

### Batch mode

Pass a 3-D array `[N, ncomp, nalm]` to transform multiple maps at once.
The batch dimension is parallelized across threads.

```matlab
N = 100;
nside = 256;
lmax  = 2 * nside;
nalm  = (lmax + 1) * (lmax + 2) / 2;

alm_batch = randn(N, 1, nalm) + 1i * randn(N, 1, nalm);
alm_batch(:, 1, 1:lmax+1) = real(alm_batch(:, 1, 1:lmax+1));

map_batch = holysht.alm2map(alm_batch, 0, nside, lmax);  % [N, 1, npix]
alm_rec   = holysht.map2alm(map_batch, 0, nside, lmax, 'n_iter', 3);
```

### Partial-sky latitude band

Restrict the transform to a band of HEALPix rings by geographic latitude
(degrees, +90 = north pole, -90 = south pole):

```matlab
nside = 128;
lmax  = 2 * nside;
nalm  = (lmax + 1) * (lmax + 2) / 2;

alm = randn(1, nalm) + 1i * randn(1, nalm);
alm(1:lmax+1) = real(alm(1:lmax+1));

% Synthesize only the equatorial band [-30, +30] degrees
map_band = holysht.alm2map(alm, 0, nside, lmax, 'lat_range', [-30, 30]);
% map_band has fewer pixels than the full 12*nside^2

% Analyze the band back to alm (same lat_range)
alm_band = holysht.map2alm(map_band, 0, nside, lmax, ...
    'lat_range', [-30, 30], 'n_iter', 3);
```

### Single precision

Input precision propagates to output:

```matlab
alm_s = single(randn(1, nalm)) + 1i * single(randn(1, nalm));
alm_s(1:lmax+1) = real(alm_s(1:lmax+1));

map_s = holysht.alm2map(alm_s, 0, nside, lmax);   % returns single
alm_s2 = holysht.map2alm(map_s, 0, nside, lmax);  % returns single
```

## Tests

```bash
matlab -nodisplay -r "run('/path/to/HolySHT/test/run_tests.m')"
```

Runs 7 tests covering spin-0/2 round-trips, batch consistency, partial-sky,
single precision, and error handling.

## Licensing

This project includes code from DUCC, which is GPLv2-licensed. The full text
is in `LICENSE`.
