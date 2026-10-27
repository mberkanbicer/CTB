% gen_reference.m - regenerate the Octave reference fixtures used by the Rust
% differential tests of the CTB port.
%
% Run from anywhere:
%   octave --no-gui --quiet CTB/rust/tests/octave/gen_reference.m
%
% The script only reads the CTB M-files; it writes into CTB/rust/tests/data/.
% Every fixture is a plain text file:
%   * matrices: "<rows> <cols> <freqs>" then one "re im" line per element
%     (row major, frequency major),
%   * Touchstone metadata: type/version/order/ref/comments.

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
ctb_dir = fullfile(script_dir, '..', '..', '..');
addpath(ctb_dir);
data_dir = fullfile(script_dir, '..', 'data');
if ~exist(data_dir, 'dir')
  mkdir(data_dir);
endif

printf('CTB reference generator\n  CTB:  %s\n  data: %s\n', ctb_dir, data_dir);

% ---------------------------------------------------------------------------
% Inputs
% ---------------------------------------------------------------------------
f2 = [1e6 2e6];
f3 = [1e6 2e6 3e6];

s2 = zeros(2, 2, 2);
s2(:, :, 1) = [0.10 + 0.20i, 0.30 - 0.10i; 0.20 + 0.05i, -0.15 + 0.40i];
s2(:, :, 2) = [0.11 + 0.21i, 0.31 - 0.11i; 0.21 + 0.06i, -0.16 + 0.41i];

s3 = zeros(3, 3, 3);
s3(:, :, 1) = [0.10 + 0.20i, 0.30 - 0.10i, 0.00 + 0.05i; ...
               0.20 + 0.00i, -0.15 + 0.40i, 0.01 + 0.00i; ...
               0.00 + 0.30i, 0.10 + 0.00i, 0.25 - 0.05i];
s3(:, :, 2) = [0.11 + 0.21i, 0.31 - 0.11i, 0.00 + 0.051i; ...
               0.21 + 0.00i, -0.16 + 0.41i, 0.011 + 0.00i; ...
               0.00 + 0.31i, 0.11 + 0.00i, 0.26 - 0.06i];
s3(:, :, 3) = [0.12 + 0.22i, 0.32 - 0.12i, 0.00 + 0.052i; ...
               0.22 + 0.00i, -0.17 + 0.42i, 0.012 + 0.00i; ...
               0.00 + 0.32i, 0.12 + 0.00i, 0.27 - 0.07i];

a2 = zeros(2, 2, 2);
a2(:, :, 1) = [1.10 + 0.20i, 20.0 - 5.0i; 0.005 + 0.001i, 0.90 - 0.10i];
a2(:, :, 2) = [1.20 + 0.10i, 25.0 - 4.0i; 0.004 + 0.002i, 0.95 - 0.05i];

z0 = [50; 75; 60];

% ---------------------------------------------------------------------------
% RM-112: per-port reference impedances (3 ports, 3 frequencies)
% ---------------------------------------------------------------------------
ctb_ref_write_mat3(fullfile(data_dir, 's3_input.txt'), s3);
ctb_ref_write_mat3(fullfile(data_dir, 'z3_perport_50_75_60.txt'), s2z(s3, z0));
ctb_ref_write_mat3(fullfile(data_dir, 'y3_perport_50_75_60.txt'), s2y(s3, z0));
ctb_ref_write_mat3(fullfile(data_dir, 's3_roundtrip_z.txt'), z2s(s2z(s3, z0), z0));
ctb_ref_write_mat3(fullfile(data_dir, 's3_roundtrip_y.txt'), y2s(s2y(s3, z0), z0));
ctb_ref_write_mat3(fullfile(data_dir, 'y3_from_z3.txt'), z2y(s2z(s3, z0)));
ctb_ref_write_mat3(fullfile(data_dir, 's3_renorm_perport.txt'), ...
                   s_renorm(s3, [50; 75; 60], [75; 50; 50]));
ctb_ref_write_mat3(fullfile(data_dir, 's3_renorm_scalar.txt'), s_renorm(s3, 50, 75));
ctb_ref_write_mat3(fullfile(data_dir, 'z3_scalar_50.txt'), s2z(s3, 50));
ctb_ref_write_mat3(fullfile(data_dir, 'y3_scalar_50.txt'), s2y(s3, 50));

% ---------------------------------------------------------------------------
% RM-113: 2-port S<->ABCD and the ABCD building blocks
% ---------------------------------------------------------------------------
ctb_ref_write_mat3(fullfile(data_dir, 's2_input.txt'), s2);
ctb_ref_write_mat3(fullfile(data_dir, 'a2_from_s2_ref50.txt'), s2a(s2, 50));
ctb_ref_write_mat3(fullfile(data_dir, 'a2_from_s2_ref75.txt'), s2a(s2, 75));
ctb_ref_write_mat3(fullfile(data_dir, 'a2_input.txt'), a2);
ctb_ref_write_mat3(fullfile(data_dir, 's2_from_a2_ref50.txt'), a2s(a2, 50));
ctb_ref_write_mat3(fullfile(data_dir, 's2_from_a2_ref75.txt'), a2s(a2, 75));

% a_PI.m cannot be executed: line 14 writes `Y1.Y2` (field access on a numeric
% array) instead of `Y1.*Y2`, so every call aborts with
%   error: complex matrix cannot be indexed with .
% There is therefore no native reference for a_pi; the Rust unit test pins it
% against a_series/a_shunt cascades instead. The other three entries of a_PI.m
% are the standard Pi-network formulas.
%
%   pi_y1 = [0.010 + 0.020i, 0.020 + 0.010i, 0.030 + 0.000i];
%   pi_y2 = [0.030 - 0.010i, 0.000 + 0.050i, 0.010 + 0.010i];
%   pi_y3 = [0.500 + 0.100i, 0.400 - 0.200i, 0.300 + 0.050i];
%   ctb_ref_write_mat3(fullfile(data_dir, 'a_pi.txt'), a_PI(pi_y1, pi_y2, pi_y3));


t_z1 = [10.0 + 5.0i, 20.0 - 5.0i, 30.0 + 1.0i];
t_z2 = [5.0 + 1.0i, 1.0 + 2.0i, 2.0 - 1.0i];
t_z3 = [50.0 + 0.0i, 30.0 - 10.0i, 40.0 + 5.0i];
ctb_ref_write_mat3(fullfile(data_dir, 'a_t.txt'), a_T(t_z1, t_z2, t_z3));

series_z = [10.0 + 20.0i, 15.0 - 5.0i, 20.0 + 0.0i];
shunt_y = [0.010 + 0.002i, 0.020 - 0.001i, 0.005 + 0.003i];
ctb_ref_write_mat3(fullfile(data_dir, 'a_series.txt'), a_series(series_z));
ctb_ref_write_mat3(fullfile(data_dir, 'a_shunt.txt'), a_shunt(shunt_y));
ctb_ref_write_mat3(fullfile(data_dir, 'a_mul.txt'), ...
                   a_mul(a_series(series_z), a_shunt(shunt_y)));

% ---------------------------------------------------------------------------
% RM-115: network operations (3 ports, 3 frequencies)
% ---------------------------------------------------------------------------
net = InitNetwork(f3, s3);
ctb_ref_write_mat3(fullfile(data_dir, 'net_init_s.txt'), net.s);
ctb_ref_write_mat3(fullfile(data_dir, 'net_init_z.txt'), net.z);
ctb_ref_write_mat3(fullfile(data_dir, 'net_init_y.txt'), net.y);
for n = 1:3
  ctb_ref_write_mat3(fullfile(data_dir, sprintf('net_init_abcd_%d.txt', n)), ...
                     net.ABCD{n});
endfor

net = AddElement2Port(net, 2, 'C_se', 1e-12);
net = AddElement2Port(net, 1:3, 'R_se', 50);
net = AddElement2Port(net, 3, 'L_sh', 1e-9);
net = AddElement2Port(net, 2, {'C_sh', 'R_sh'}, [2e-12, 75]);
ctb_ref_write_mat3(fullfile(data_dir, 'net_add_s.txt'), net.s);
ctb_ref_write_mat3(fullfile(data_dir, 'net_add_z.txt'), net.z);
ctb_ref_write_mat3(fullfile(data_dir, 'net_add_y.txt'), net.y);
for n = 1:3
  ctb_ref_write_mat3(fullfile(data_dir, sprintf('net_add_abcd_%d.txt', n)), ...
                     net.ABCD{n});
endfor

[I_single, Iorig_single] = ApplyCurrent2Port(net, 1, 1e-3);
ctb_ref_write_mat3(fullfile(data_dir, 'current_single_i.txt'), I_single);
ctb_ref_write_mat3(fullfile(data_dir, 'current_single_iorig.txt'), Iorig_single);
[I_multi, Iorig_multi] = ApplyCurrent2Port(net, [1 3], [1e-3 2e-3]);
ctb_ref_write_mat3(fullfile(data_dir, 'current_multi_i.txt'), I_multi);
ctb_ref_write_mat3(fullfile(data_dir, 'current_multi_iorig.txt'), Iorig_multi);
[I_power, Iorig_power] = ApplyRFPower2Port(net, 2, 1e-3);
ctb_ref_write_mat3(fullfile(data_dir, 'power_i.txt'), I_power);
ctb_ref_write_mat3(fullfile(data_dir, 'power_iorig.txt'), Iorig_power);

net = SetPortTermination(net, 2, 75);
ctb_ref_write_mat3(fullfile(data_dir, 'net_term_s.txt'), net.s);
ctb_ref_write_mat3(fullfile(data_dir, 'net_term_z.txt'), net.z);
% net.y is intentionally NOT refreshed by SetPortTermination.m
ctb_ref_write_mat3(fullfile(data_dir, 'net_term_y.txt'), net.y);
ctb_ref_write_mat3(fullfile(data_dir, 'net_term_z0.txt'), ...
                   reshape(net.Z0, 1, numel(net.Z0), 1));

% ---------------------------------------------------------------------------
% RM-114: native Touchstone files plus what read_touchstone.m makes of them
% ---------------------------------------------------------------------------
ctb_ref_write_case(data_dir, 'native_2port.s2p', 's', f2, s2, 50, ...
                   'ctb differential 2-port');
ctb_ref_write_case(data_dir, 'native_3port.s3p', 's', f3, s3, 50, ...
                   'ctb differential 3-port');

z1 = zeros(1, 1, 3);
z1(1, 1, :) = [50.0 + 0.0i, 75.0 + 25.0i, 100.0 - 50.0i];
ctb_ref_write_case(data_dir, 'native_1port_z.s1p', 'z', f3, z1, 75, ...
                   'one-port z');

s4 = zeros(4, 4, 2);
for f = 1:2
  for i = 1:4
    for j = 1:4
      s4(i, j, f) = 0.01 * i * j + 0.02 * f + 1i * (0.03 * i - 0.01 * j);
    endfor
  endfor
  s4(:, :, f) = 0.5 * (s4(:, :, f) + s4(:, :, f).');
endfor
ctb_ref_write_case(data_dir, 'native_4port.s4p', 's', f2, s4, 50, 'four port');

s5 = zeros(5, 5, 1);
for i = 1:5
  for j = 1:5
    s5(i, j, 1) = 0.02 * (i + j) + 1i * 0.01 * (i - j);
  endfor
endfor
s5 = 0.5 * (s5 + s5.');
ctb_ref_write_case(data_dir, 'native_5port.s5p', 's', [3e6], s5, 50, 'five port');

% A complex reference impedance: read_touchstone.m accepts "R 50+5i".
ctb_ref_write_case(data_dir, 'native_complex_ref.s2p', 's', f2, s2, 50 + 5i, ...
                   'complex reference');

% Hand written v1 files the native writer never produces: dB and MA formats,
% other frequency units, mixed case keywords, comments and blank lines.
ctb_ref_write_hand(data_dir, 'hand_db_2port.s2p', sprintf([ ...
  '! hand written dB file\n', ...
  '# MHz S DB R 50\n', ...
  '! second comment\n', ...
  '100  -20.0 0.0   0.0 0.0   -6.020599913279624 90.0   0.0 0.0\n', ...
  '200  -10.0 45.0  0.0 0.0   -3.010299956639812 45.0   -40.0 -90.0\n']));

ctb_ref_write_hand(data_dir, 'hand_ma_1port.s1p', sprintf([ ...
  '# KHZ z ma r 75\n', ...
  '2 0.5 180.0\n', ...
  '4 0.25 -90.0\n']));

ctb_ref_write_hand(data_dir, 'hand_ghz_comments.s2p', sprintf([ ...
  '# GHz Y RI R 50\n', ...
  '10  0.001 0.0  0.002 0.0  0.002 0.0  0.003 0.0\n', ...
  '\n', ...
  '! trailing comment\n', ...
  '20  0.0011 0.0001  0.0021 0.0001  0.0021 0.0001  0.0031 0.0001\n']));

printf('done\n');
