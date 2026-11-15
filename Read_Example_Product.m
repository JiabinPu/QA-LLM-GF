% Read one final product; MATLAB orientation is row,col without transposing.
[name, folder] = uigetfile('*.h5', 'Select a final QA-LLM-GF example product');
if isequal(name, 0), return; end
file = fullfile(folder, name);
LAI_DN = h5read(file, '/Lai');
FPAR_DN = h5read(file, '/Fpar');
LAI = double(LAI_DN) * 0.001; LAI(LAI_DN == 65535) = NaN;
FPAR = double(FPAR_DN) * 0.001; FPAR(FPAR_DN == 65535) = NaN;
LAI_u_DN = h5read(file, '/Lai_Uncertainty');
FPAR_u_DN = h5read(file, '/Fpar_Uncertainty');
LAI_u = double(LAI_u_DN) * 0.001; LAI_u(LAI_u_DN == 65535) = NaN;
FPAR_u = double(FPAR_u_DN) * 0.001; FPAR_u(FPAR_u_DN == 65535) = NaN;
LAI_QA = h5read(file, '/Lai_QA');
FPAR_QA = h5read(file, '/Fpar_QA');
figure; tiledlayout(1,2);
nexttile; imagesc(LAI, [0 7]); axis image; colorbar; title('LAI');
nexttile; imagesc(FPAR, [0 1]); axis image; colorbar; title('FPAR');
