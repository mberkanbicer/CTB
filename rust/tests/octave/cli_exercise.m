% Exercise the Rust CTB command line from Octave through the delivered M-files
% (RM-116). Run with CTB_BIN pointing at the `ctb` executable:
%
%   CTB_BIN=<repo>/target/debug/ctb octave --no-gui --quiet cli_exercise.m
%
% The native M-files (write_touchstone.m, read_touchstone.m, s2z.m) are used
% unchanged as the reference.

script_dir = fileparts(mfilename('fullpath'));
ctb_root = fullfile(script_dir, '..', '..', '..');      % CTB/
addpath(ctb_root);
addpath(fullfile(script_dir, '..', '..', 'matlab'));    % CTB/rust/matlab

work = tempname();
mkdir(work);
input = fullfile(work, 'input.s2p');
z_file = fullfile(work, 'converted.s2p');
back_file = fullfile(work, 'back.s2p');

freq = [1e9; 2e9];
s = zeros(2, 2, 2);
s(1, 1, 1) = 0.1 + 0.2i;  s(1, 2, 1) = 0.7 - 0.1i;
s(2, 1, 1) = 0.6 + 0.05i; s(2, 2, 1) = -0.2 + 0.3i;
s(1, 1, 2) = 0.05 - 0.1i; s(1, 2, 2) = 0.8;
s(2, 1, 2) = 0.75 + 0.1i; s(2, 2, 2) = 0.2 - 0.4i;
write_touchstone('s', freq, s, input, 50, 'octave cli exercise');

% convert to Z through the CLI (scalar reference: the native reader is v1)
converted = ctb_convert(input, z_file, 'z', 50);
[type, f_z, data_z, ref_z] = read_touchstone(z_file);
assert(strcmp(type, 'z'), 'converted type %s', type);
assert(max(abs(f_z(:) - freq(:))) == 0, 'frequency mismatch');
assert(abs(ref_z - 50) < 1e-12, 'reference mismatch');
expected = s2z(s, 50);
err = max(max(max(abs(data_z - expected))));
fprintf('Z conversion error: %.3e\n', err);
% the native `.6e` format keeps 7 significant digits
assert(err < 1e-5 * max(1, max(max(max(abs(expected))))), ...
       'Z conversion differs by %.3e', err);

% and back to S
back = ctb_convert(z_file, back_file, 's', 50);
[type, ~, data_s, ~] = read_touchstone(back_file);
assert(strcmp(type, 's'), 'back type %s', type);
err = max(max(max(abs(data_s - s))));
fprintf('S round trip error: %.3e\n', err);
assert(err < 1e-5, 'S round trip differs by %.3e', err);

lines = ctb_info(back_file);
joined = strjoin(lines, ' ');
assert(~isempty(strfind(joined, 'ports: 2')), 'info did not report two ports');
fprintf('CLI info: %s\n', joined);

rmdir(work, 's');
fprintf('CTB CLI Octave exercise: PASS\n');
