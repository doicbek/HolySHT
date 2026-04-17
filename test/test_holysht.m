function test_holysht
%TEST_HOLYSHT Smoke tests for the holysht MATLAB wrapper.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here, '..', 'setup_holysht.m'));

    fprintf('=== Test 1: spin-0 round-trip (full sky, nside=64) ===\n');
    nside = 64; lmax = 2*nside; nalm = (lmax+1)*(lmax+2)/2;
    rng(42);
    alm0 = randn(1, nalm) + 1i*randn(1, nalm);
    map  = holysht.alm2map(alm0, 0, nside, lmax);
    alm1 = holysht.map2alm(map, 0, nside, lmax, 'n_iter', 10);
    err  = norm(alm0(:) - alm1(:)) / norm(alm0(:));
    fprintf('   relative error = %.2e\n', err);
    assert(err < 1e-10, 'spin-0 round-trip error too large');

    fprintf('=== Test 2: spin-2 round-trip (full sky, nside=64) ===\n');
    rng(43);
    alm0 = randn(2, nalm) + 1i*randn(2, nalm);
    % Zero out unphysical low-l modes (l<2 don't exist for spin 2).
    for ell = 0:1
        for m = 0:ell
            idx = m_l_to_idx(lmax, m, ell);
            alm0(:, idx) = 0;
        end
    end
    map  = holysht.alm2map(alm0, 2, nside, lmax);
    alm1 = holysht.map2alm(map, 2, nside, lmax, 'n_iter', 10);
    err  = norm(alm0(:) - alm1(:)) / norm(alm0(:));
    fprintf('   relative error = %.2e\n', err);
    assert(err < 1e-9, 'spin-2 round-trip error too large');

    fprintf('=== Test 3: batch alm2map equals per-element loop ===\n');
    Nb = 4;
    alm_b = randn(Nb, 1, nalm) + 1i*randn(Nb, 1, nalm);
    map_b = holysht.alm2map(alm_b, 0, nside, lmax);
    err = 0;
    for ib = 1:Nb
        a = squeeze(alm_b(ib,1,:)).';        % [1, nalm]
        m = holysht.alm2map(a, 0, nside, lmax);  % [1, npix]
        err = max(err, max(abs(squeeze(map_b(ib,1,:)).' - m)));
    end
    fprintf('   max abs diff = %.2e\n', err);
    assert(err < 1e-13, 'batch / loop mismatch');

    fprintf('=== Test 4: lat_range reduces output size ===\n');
    npix_full = 12 * nside^2;
    map_band = holysht.alm2map(alm0, 2, nside, lmax, ...
        'lat_range', [-30, 30]);
    assert(size(map_band, 2) < npix_full, ...
        'lat_range did not reduce npix');
    fprintf('   full npix = %d, band npix = %d\n', npix_full, size(map_band, 2));

    fprintf('=== Test 5: lat_range partial-sky map2alm is finite ===\n');
    alm_band = holysht.map2alm(map_band, 2, nside, lmax, ...
        'lat_range', [-30, 30], 'n_iter', 0);
    map_back = holysht.alm2map(alm_band, 2, nside, lmax, ...
        'lat_range', [-30, 30]);
    assert(all(isfinite(map_back(:))), 'partial-sky output has NaN/Inf');

    fprintf('=== Test 6: single precision round-trip ===\n');
    alm0_s = single(real(alm0(1,:))) + 1i*single(imag(alm0(1,:)));
    map_s  = holysht.alm2map(alm0_s, 0, nside, lmax);
    assert(isa(map_s, 'single'), 'alm2map did not preserve single precision');
    alm1_s = holysht.map2alm(map_s, 0, nside, lmax, 'n_iter', 5);
    assert(isa(alm1_s, 'single'), 'map2alm did not preserve single precision');
    err = norm(double(alm0_s(:)) - double(alm1_s(:))) / norm(double(alm0_s(:)));
    fprintf('   single-precision relative error = %.2e\n', err);
    assert(err < 1e-3, 'single round-trip error too large');

    fprintf('=== Test 7: error cases ===\n');
    threw = false;
    try
        holysht.alm2map(alm0(1,:), 1, nside, lmax);
    catch
        threw = true;
    end
    assert(threw, 'spin=1 should have errored');
    threw = false;
    try
        holysht.alm2map(alm0(1,:), 0, nside, lmax, ...
            'lat_range', [30, -30]);
    catch
        threw = true;
    end
    assert(threw, 'inverted lat_range should have errored');

    fprintf('\nAll tests passed.\n');
end

function idx = m_l_to_idx(lmax, m, ell)
% 1-based MATLAB index of (m, l) in the standard col-major mstart layout.
    idx = m * (lmax + 1) - m*(m-1)/2 + (ell - m) + 1;
end
