function ctb_ref_write_case(data_dir, name, type, freq, data, ref, comment)
% Write a Touchstone file with the native writer and record what
% read_touchstone.m returns for it.
  file = fullfile(data_dir, name);
  if isempty(ref)
    write_touchstone(type, freq, data, file, [], comment);
  else
    write_touchstone(type, freq, data, file, ref, comment);
  endif
  [rtype, rfreq, rdata, rref, rcomment] = read_touchstone(file);
  ctb_ref_write_mat3(fullfile(data_dir, [name '.input.txt']), data);
  ctb_ref_write_mat3(fullfile(data_dir, [name '.freq.txt']), ...
                     reshape(rfreq, 1, numel(rfreq), 1));
  ctb_ref_write_mat3(fullfile(data_dir, [name '.data.txt']), rdata);
  ctb_ref_write_meta(fullfile(data_dir, [name '.meta.txt']), rtype, 1, '21', ...
                     rref * ones(1, size(rdata, 1)), rcomment);
  printf('  touchstone %-22s %d ports, %d freqs, ref %s\n', name, ...
         size(rdata, 1), numel(rfreq), mat2str(rref));
endfunction
