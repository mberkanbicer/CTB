function info = ctb_info( filename )
% CTB_INFO( filename ) - report a Touchstone file through the Rust `ctb` CLI.
%
% Returns a struct with the parsed `info` text lines; the executable is taken
% from the CTB_BIN environment variable.

binary = getenv('CTB_BIN');
if isempty(binary)
    error('ctb_info:noBinary', ...
          'set CTB_BIN to the ctb executable (CTB/rust target directory)');
end
[status, output] = system(sprintf('"%s" info "%s"', binary, filename));
if status ~= 0
    error('ctb_info:failed', 'ctb failed (%d): %s', status, output);
end
info = strtrim(strsplit(output, sprintf('\n')));
if nargout == 0
    fprintf('%s\n', info{:});
end
end
