% setup_holysht  Configure MATLAB path for the holysht wrapper.
%
%   Run this script once from the MATLAB command window (or add it to your
%   startup.m) to make the holysht package available in your session.
%
%   Usage:
%     >> run('/path/to/HolySHT/setup_holysht.m')

this_dir = fileparts(mfilename('fullpath'));

% Add the +holysht package directory
addpath(this_dir);

% Locate and add the MEX binary directory.
mex_build = fullfile(this_dir, 'mex', 'build');

if exist(mex_build, 'dir')
    addpath(mex_build);
else
    warning('holysht:setup:mexNotFound', ...
        ['MEX binaries not found in %s.\n' ...
         'Build them first:\n' ...
         '  cd %s\n' ...
         '  mkdir build && cd build\n' ...
         '  cmake ..\n' ...
         '  cmake --build . -j$(nproc)\n' ...
         'Then re-run setup_holysht.m.'], ...
        mex_build, fullfile(this_dir, 'mex'));
    return;
end

% Quick sanity check: verify a representative MEX file is reachable.
if exist('holysht_alm2map_mex', 'file') ~= 3
    warning('holysht:setup:mexNotOnPath', ...
        'MEX binaries were found in %s but cannot be loaded.\n%s', ...
        mex_build, ...
        'Ensure the MEX files were compiled for this platform.');
    return;
end

fprintf('holysht: paths configured.\n');
fprintf('  package : %s\n', this_dir);
fprintf('  MEX     : %s\n', mex_build);
