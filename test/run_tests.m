% run_tests.m  Run holysht test suite with nthreads=1.
try
    here = fileparts(mfilename('fullpath'));
    cd(fullfile(here, '..'));
    run('setup_holysht.m');
    nt = 1;

    fprintf('=== Test 1: spin-0 round-trip (nside=64) ===\n');
    nside = 64; lmax = 2*nside; nalm = (lmax+1)*(lmax+2)/2;
    rng(42);
    alm0 = randn(1, nalm) + 1i*randn(1, nalm);
    % m=0 modes (indices 1..lmax+1) must be real (Y_{l,0} is real)
    alm0(1:lmax+1) = real(alm0(1:lmax+1));
    map  = holysht.alm2map(alm0, 0, nside, lmax, 'nthreads', nt);
    alm1 = holysht.map2alm(map, 0, nside, lmax, 'n_iter', 10, 'nthreads', nt);
    err  = norm(alm0(:) - alm1(:)) / norm(alm0(:));
    fprintf('  err = %.2e\n', err);
    assert(err < 1e-10, 'spin-0 round-trip error too large');
    fprintf('  PASSED\n');

    fprintf('=== Test 2: spin-2 round-trip (nside=64) ===\n');
    rng(43);
    alm0 = randn(2, nalm) + 1i*randn(2, nalm);
    % m=0 modes must be real
    alm0(:, 1:lmax+1) = real(alm0(:, 1:lmax+1));
    % Zero unphysical low-l modes (l<2 for spin-2)
    alm0(:, 1) = 0;  % (m=0,l=0)
    alm0(:, 2) = 0;  % (m=0,l=1)
    alm0(:, lmax + 2) = 0;  % (m=1,l=1) in DUCC mstart layout
    map  = holysht.alm2map(alm0, 2, nside, lmax, 'nthreads', nt);
    alm1 = holysht.map2alm(map, 2, nside, lmax, 'n_iter', 10, 'nthreads', nt);
    err  = norm(alm0(:) - alm1(:)) / norm(alm0(:));
    fprintf('  err = %.2e\n', err);
    assert(err < 1e-9, 'spin-2 round-trip error too large');
    fprintf('  PASSED\n');

    fprintf('=== Test 3: batch == loop ===\n');
    Nb = 4;
    alm_b = randn(Nb, 1, nalm) + 1i*randn(Nb, 1, nalm);
    % m=0 modes must be real
    alm_b(:, 1, 1:lmax+1) = real(alm_b(:, 1, 1:lmax+1));
    map_b = holysht.alm2map(alm_b, 0, nside, lmax, 'nthreads', nt);
    maxerr = 0;
    for ib = 1:Nb
        a = squeeze(alm_b(ib,1,:)).';
        m = holysht.alm2map(a, 0, nside, lmax, 'nthreads', nt);
        maxerr = max(maxerr, max(abs(squeeze(map_b(ib,1,:)).' - m)));
    end
    fprintf('  max abs diff = %.2e\n', maxerr);
    assert(maxerr < 1e-13, 'batch / loop mismatch');
    fprintf('  PASSED\n');

    fprintf('=== Test 4: lat_range reduces output size ===\n');
    npix_full = 12 * nside^2;
    alm_lr = randn(1, nalm) + 1i*randn(1, nalm);
    alm_lr(1:lmax+1) = real(alm_lr(1:lmax+1));
    map_band = holysht.alm2map(alm_lr, 0, nside, lmax, ...
        'lat_range', [-30, 30], 'nthreads', nt);
    assert(size(map_band, 2) < npix_full, 'lat_range did not reduce npix');
    fprintf('  full=%d, band=%d\n', npix_full, size(map_band, 2));
    fprintf('  PASSED\n');

    fprintf('=== Test 5: lat_range partial-sky is finite ===\n');
    alm_band = holysht.map2alm(map_band, 0, nside, lmax, ...
        'lat_range', [-30, 30], 'n_iter', 0, 'nthreads', nt);
    map_back = holysht.alm2map(alm_band, 0, nside, lmax, ...
        'lat_range', [-30, 30], 'nthreads', nt);
    assert(all(isfinite(map_back(:))), 'partial-sky NaN/Inf');
    fprintf('  PASSED\n');

    fprintf('=== Test 6: single precision ===\n');
    alm0_s = single(randn(1, nalm)) + 1i*single(randn(1, nalm));
    alm0_s(1:lmax+1) = real(alm0_s(1:lmax+1));
    map_s = holysht.alm2map(alm0_s, 0, nside, lmax, 'nthreads', nt);
    assert(isa(map_s, 'single'), 'alm2map not single');
    alm1_s = holysht.map2alm(map_s, 0, nside, lmax, 'n_iter', 5, 'nthreads', nt);
    assert(isa(alm1_s, 'single'), 'map2alm not single');
    err = norm(double(alm0_s(:)) - double(alm1_s(:))) / norm(double(alm0_s(:)));
    fprintf('  single err = %.2e\n', err);
    assert(err < 1e-3, 'single round-trip too large');
    fprintf('  PASSED\n');

    fprintf('=== Test 7: error cases ===\n');
    threw = false;
    try holysht.alm2map(alm_lr, 1, nside, lmax); catch, threw = true; end
    assert(threw, 'spin=1 should error');
    threw = false;
    try holysht.alm2map(alm_lr, 0, nside, lmax, 'lat_range', [30, -30]); catch, threw = true; end
    assert(threw, 'inverted lat_range should error');
    fprintf('  PASSED\n');

    fprintf('\nAll 7 tests passed.\n');
catch e
    fprintf('FAILED: %s: %s\n', e.identifier, e.message);
    for k = 1:length(e.stack)
        fprintf('  %s:%d\n', e.stack(k).file, e.stack(k).line);
    end
end
