function varargout = ctb_convert( infile, outfile, to_type, ref )
% CTB_CONVERT( infile, outfile, to_type [,ref] ) - convert a Touchstone file
% with the Rust `ctb` command line and read the result back with the native
% `read_touchstone.m` (RM-116 delivery form).
%
% infile   input Touchstone file (.sNp)
% outfile  output Touchstone file (.sNp); the parameter type is stored in the
%          option line, exactly like write_touchstone.m
% to_type  's', 'y', 'z', 'h' or 'g' (Touchstone cannot store ABCD)
% ref      optional reference impedance: scalar (50) or per port ([50 75 60]).
%          A per-port vector is written as a Touchstone v2 [Reference] block,
%          which the native read_touchstone.m cannot parse; read those files
%          with the Rust library or `ctb info`.
%
% The executable is taken from the CTB_BIN environment variable; the Rust
% binary is built with `cargo build --release` in CTB/rust.

if nargin < 3
    error('ctb_convert:arguments', 'usage: ctb_convert(infile, outfile, to_type [,ref])');
end
binary = getenv('CTB_BIN');
if isempty(binary)
    error('ctb_convert:noBinary', ...
          'set CTB_BIN to the ctb executable (CTB/rust target directory)');
end

command = sprintf('"%s" convert --to %s', binary, lower(to_type));
if nargin >= 4 && ~isempty(ref)
    if isscalar(ref)
        reference = sprintf('%.17g', ref);
    else
        reference = strjoin(arrayfun(@(value) sprintf('%.17g', value), ...
                                     ref(:).', 'UniformOutput', false), ',');
    end
    command = sprintf('%s --ref %s', command, reference);
end
command = sprintf('%s "%s" "%s"', command, infile, outfile);

[status, output] = system(command);
if status ~= 0
    error('ctb_convert:failed', 'ctb failed (%d): %s', status, output);
end

[~, freq, data, ref_out, comment] = read_touchstone(outfile);
if nargout == 0
    fprintf('converted %s -> %s (%s)\n', infile, outfile, lower(to_type));
else
    varargout{1} = struct('type', lower(to_type), 'freq', freq, 'data', data, ...
                          'ref', ref_out, 'comment', comment);
end
end
