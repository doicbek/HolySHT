function alm = map2alm(map, spin, nside, lmax, varargin)
%MAP2ALM HEALPix map -> alm spherical-harmonic transform (spin 0 or 2).
%   ALM = MAP2ALM(MAP, SPIN, NSIDE, LMAX) computes the iterative SHT
%   (adjoint_synthesis + Jacobi refinement) on a HEALPix map.
%
%   ALM = MAP2ALM(..., 'lat_range', [LATMIN LATMAX]) restricts the
%   transform to a contiguous latitude band (degrees, +90 = north pole).
%   In that case MAP must be packed to contain only the pixels of the
%   selected rings, in ring order. NPIX_EFF = sum(nphi(selected_rings)).
%   For full-sky calls (default), NPIX_EFF = 12*NSIDE^2.
%
%   ALM = MAP2ALM(..., 'n_iter', N)   number of Jacobi iterations (default 3)
%   ALM = MAP2ALM(..., 'nthreads', N) thread count (default 0 = auto)
%
%   Inputs
%   ------
%   map  : real array, [ncomp, npix_eff] or [N, ncomp, npix_eff] (batch),
%          where ncomp = 1 for spin=0 and 2 for spin=2.
%          Single or double precision.
%   spin : 0 or 2.
%   nside, lmax : non-negative integer scalars.
%
%   Output
%   ------
%   alm  : complex array, [ncomp, nalm] or [N, ncomp, nalm], where
%          nalm = (lmax+1)*(lmax+2)/2.

    % Lightweight name-value parsing.
    lat_range = [];
    n_iter    = 3;
    nthreads  = 0;
    for k = 1:2:numel(varargin)
        switch lower(varargin{k})
            case 'lat_range'
                lat_range = double(varargin{k+1});
            case 'n_iter'
                n_iter = double(varargin{k+1});
            case 'nthreads'
                nthreads = double(varargin{k+1});
            otherwise
                error('holysht:map2alm:InputError', ...
                    'Unknown parameter: %s', varargin{k});
        end
    end

    if spin ~= 0 && spin ~= 2
        error('holysht:map2alm:InputError', 'spin must be 0 or 2');
    end
    if ~isscalar(nside) || nside < 0 || nside ~= floor(nside)
        error('holysht:map2alm:InputError', ...
            'nside must be a non-negative integer scalar');
    end
    if ~isscalar(lmax) || lmax < 0 || lmax ~= floor(lmax)
        error('holysht:map2alm:InputError', ...
            'lmax must be a non-negative integer scalar');
    end

    info  = healpix_ring_info(nside, lat_range);
    ncomp = 1 + double(spin ~= 0);
    npix  = info.npix;

    % Normalise vector input to [ncomp, npix].
    if isvector(map) && numel(map) == ncomp * npix
        map = reshape(map, [ncomp, npix]);
    end
    sz = size(map);
    if ndims(map) == 3
        if sz(2) ~= ncomp || sz(3) ~= npix
            error('holysht:map2alm:InputError', ...
                'Batch map must be [N, %d, %d]', ncomp, npix);
        end
    elseif numel(sz) == 2 && sz(1) == ncomp && sz(2) == npix
        % single-map OK
    else
        error('holysht:map2alm:InputError', ...
            'Map must be [%d, %d] or [N, %d, %d]', ncomp, npix, ncomp, npix);
    end

    alm = holysht_map2alm_mex( ...
        map, double(lmax), double(spin), ...
        info.theta, info.nphi, info.phi0, info.ringstart, ...
        info.weight, double(n_iter), double(nthreads));
end
