function ctb_ref_write_meta(filename, type, version, order, refs, comments)
% Write the non-matrix part of a Touchstone file for the Rust differential
% tests. Every reference impedance is written as a "re im" pair so complex
% values survive.
  fid = fopen(filename, 'w');
  if fid < 0
    error('cannot write %s', filename);
  endif
  fprintf(fid, 'type %s\n', type);
  fprintf(fid, 'version %d\n', version);
  fprintf(fid, 'order %s\n', order);
  fprintf(fid, 'ref');
  for n = 1:numel(refs)
    fprintf(fid, ' %.17g %.17g', real(refs(n)), imag(refs(n)));
  endfor
  fprintf(fid, '\n');
  fprintf(fid, 'comments %d\n', numel(comments));
  for n = 1:numel(comments)
    fprintf(fid, '%s\n', comments{n});
  endfor
  fclose(fid);
endfunction
