function info = healpix_ring_info(nside, lat_range)
%HEALPIX_RING_INFO Compute HEALPix ring geometry, optionally limited by latitude.
%   INFO = HEALPIX_RING_INFO(NSIDE) returns the full set of HEALPix ring
%   metadata for the given NSIDE.
%   INFO = HEALPIX_RING_INFO(NSIDE, LAT_RANGE) keeps only those rings whose
%   geographic latitude (degrees, +90 = north pole) lies in
%   [LAT_RANGE(1), LAT_RANGE(2)]. RINGSTART is repacked to be contiguous
%   over the kept rings (so the MEX layer treats the supplied map as a
%   flat packed-pixel layout).
%
%   Returned struct fields: theta, phi0, nphi, ringstart, weight, npix.

    nside  = double(nside);
    npix0  = 12 * nside * nside;
    rings  = (1:(4*nside - 1))';
    nring  = numel(rings);

    theta = zeros(nring, 1, 'double');
    phi0  = zeros(nring, 1, 'double');
    nphi  = zeros(nring, 1, 'double');

    northrings = rings;
    northrings(rings > 2*nside) = 4*nside - rings(rings > 2*nside);

    cap = northrings < nside;
    theta(cap) = 2*asin(northrings(cap)/(sqrt(6)*nside));
    nphi(cap)  = 4*northrings(cap);
    phi0(cap)  = pi ./ (4*northrings(cap));

    rest = ~cap;
    theta(rest) = acos((2*nside - northrings(rest)) * (8*nside/npix0));
    nphi(rest)  = 4*nside;
    phi0(rest)  = (pi/(4*nside)) * double(mod(northrings(rest) - nside, 2) == 0);

    south = northrings ~= rings;
    theta(south) = pi - theta(south);

    weight = 4*pi/npix0;

    if nargin >= 2 && ~isempty(lat_range)
        if numel(lat_range) ~= 2 || lat_range(1) >= lat_range(2)
            error('holysht:RingInfo:BadLatRange', ...
                'lat_range must be a 2-vector with lat_range(1) < lat_range(2)');
        end
        if lat_range(1) < -90 || lat_range(2) > 90
            error('holysht:RingInfo:BadLatRange', ...
                'lat_range entries must lie in [-90, 90] (degrees)');
        end
        theta_min = (90 - lat_range(2)) * pi / 180;
        theta_max = (90 - lat_range(1)) * pi / 180;
        sel = (theta >= theta_min) & (theta <= theta_max);
        theta = theta(sel);
        phi0  = phi0(sel);
        nphi  = nphi(sel);
        if isempty(theta)
            error('holysht:RingInfo:EmptySelection', ...
                'lat_range selects zero HEALPix rings at nside=%d', nside);
        end
    end

    % Repack ringstart contiguously over the kept rings.
    ringstart = [0; cumsum(nphi(1:end-1))];
    info = struct();
    info.theta     = theta;
    info.phi0      = phi0;
    info.nphi      = nphi;
    info.ringstart = ringstart;
    info.weight    = weight;
    info.npix      = sum(nphi);
end
