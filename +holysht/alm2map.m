function map = alm2map(alm, spin, nside, lmax, varargin)
%ALM2MAP HEALPix alm -> map spherical-harmonic synthesis (spin 0 or 2).
%   MAP = ALM2MAP(ALM, SPIN, NSIDE, LMAX) synthesises a HEALPix map from
%   spherical-harmonic coefficients.
%
%   MAP = ALM2MAP(..., 'lat_range', [LATMIN LATMAX]) returns only the
%   pixels of HEALPix rings whose geographic latitude (degrees) lies in
%   the requested band. NPIX_EFF = sum(nphi(selected_rings)).
%
%   MAP = ALM2MAP(..., 'nthreads', N) thread count (default 0 = auto).
%
%   Inputs
%   ------
%   alm  : complex array, [ncomp, nalm] or [N, ncomp, nalm] (batch),
%          ncomp = 1 (spin=0) or 2 (spin=2), nalm = (lmax+1)*(lmax+2)/2.
%   spin : 0 or 2.
%   nside, lmax : non-negative integer scalars.
%
%   Output
%   ------
%   map : real array, [ncomp, npix_eff] (or [N, ncomp, npix_eff] in batch
%         mode), in single or double precision matching the input.

    lat_range = [];
    nthreads  = 0;
    for k = 1:2:numel(varargin)
        switch lower(varargin{k})
            case 'lat_range'
                lat_range = double(varargin{k+1});
            case 'nthreads'
                nthreads = double(varargin{k+1});
            otherwise
                error('holysht:alm2map:InputError', ...
                    'Unknown parameter: %s', varargin{k});
        end
    end

    if spin ~= 0 && spin ~= 2
        error('holysht:alm2map:InputError', 'spin must be 0 or 2');
    end
    if ~isscalar(nside) || nside < 0 || nside ~= floor(nside)
        error('holysht:alm2map:InputError', ...
            'nside must be a non-negative integer scalar');
    end
    if ~isscalar(lmax) || lmax < 0 || lmax ~= floor(lmax)
        error('holysht:alm2map:InputError', ...
            'lmax must be a non-negative integer scalar');
    end

    info  = healpix_ring_info(nside, lat_range);
    ncomp = 1 + double(spin ~= 0);
    nalm  = (lmax + 1) * (lmax + 2) / 2;

    if isvector(alm)
        alm = reshape(alm, [ncomp, nalm]);
    end
    if isreal(alm)
        warning('holysht:alm2map:RealAlm', ...
            'alm should be complex; converting.');
        alm = complex(alm);
    end

    map = holysht_alm2map_mex( ...
        alm, double(lmax), double(spin), ...
        info.theta, info.nphi, info.phi0, info.ringstart, ...
        double(nthreads));
end
