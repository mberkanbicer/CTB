function ctb_ref_write_hand(data_dir, name, text)
% Write a hand-written Touchstone file (not produced by write_touchstone.m)
% and record what read_touchstone.m returns for it. Used to cover v1 files the
% native writer never emits (dB/MA formats, other units, comments, blank lines).
  file = fullfile(data_dir, name);
  fid = fopen(file, 'w');
  if fid < 0
    error('cannot write %s', file);
  endif
  fprintf(fid, '%s', text);
  fclose(fid);
  [rtype, rfreq, rdata, rref, rcomment] = read_touchstone(file);
  ctb_ref_write_mat3(fullfile(data_dir, [name '.freq.txt']), ...
                     reshape(rfreq, 1, numel(rfreq), 1));
  ctb_ref_write_mat3(fullfile(data_dir, [name '.data.txt']), rdata);
  ctb_ref_write_meta(fullfile(data_dir, [name '.meta.txt']), rtype, 1, '21', ...
                     rref * ones(1, size(rdata, 1)), rcomment);
  printf('  hand written %-20s %d ports, %d freqs, ref %s\n', name, ...
         size(rdata, 1), numel(rfreq), mat2str(rref));
endfunction
