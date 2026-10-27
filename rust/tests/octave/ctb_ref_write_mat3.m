function ctb_ref_write_mat3(filename, a)
% Write an n x m x f complex array as text for the Rust differential tests.
%
% Header: "<rows> <cols> <frequencies>"
% Body:   one "re im" line per element, row major, frequency major.
  [n, m, nf] = size(a);
  fid = fopen(filename, 'w');
  if fid < 0
    error('cannot write %s', filename);
  endif
  fprintf(fid, '%d %d %d\n', n, m, nf);
  for f = 1:nf
    for i = 1:n
      for j = 1:m
        fprintf(fid, '%.17g %.17g\n', real(a(i, j, f)), imag(a(i, j, f)));
      endfor
    endfor
  endfor
  fclose(fid);
endfunction
