function Example03_Produce_Tile(dataRoot, years, pythonExe, outputDir, device, cpuWorkers, gpuWorkers, mode)
%EXAMPLE03_PRODUCE_TILE Run one native tile with the frozen models.
% dataRoot must be the tile directory containing Module00/Module01 inputs.
% Example03_Produce_Tile('/path/h23v04') % Check only; no production
% Example03_Produce_Tile('/path/h23v04',2001:2024,'python','/path/output','cpu_gpu',5,5,'produce')
% Years can also be '2001:2024' or '2001,2005,2013'. Device is cpu, mps,
% cuda, auto, or cpu_gpu. Auto uses CPU+GPU when available and CPU otherwise.
% MATLAB GPU/Parallel toolboxes are not required.
% Zero workers disables that lane; at least one lane must remain enabled.
assert(nargin >= 1 && ~isempty(dataRoot),'QALLMGPU:DataRoot', ...
    'Pass the tile input directory, for example /path/h23v04.');
if nargin < 2 || isempty(years); years = 2001:2024; end
if nargin < 3 || isempty(pythonExe); pythonExe = 'python'; end
if nargin < 4 || isempty(outputDir); outputDir = ''; end
if nargin < 5 || isempty(device); device = 'auto'; end
if nargin < 6 || isempty(cpuWorkers); cpuWorkers = 5; end
if nargin < 7 || isempty(gpuWorkers); gpuWorkers = 5; end
if nargin < 8 || isempty(mode); mode = 'check'; end
assert((ischar(dataRoot) && isrow(dataRoot)) || (isstring(dataRoot) && isscalar(dataRoot)), ...
    'QALLMGPU:DataRoot','dataRoot must be one tile directory path.');
dataRoot = char(dataRoot);
assert(isfolder(dataRoot),'QALLMGPU:DataRoot','Tile input directory is missing: %s',dataRoot);
mode = validatestring(mode,{'check','produce','smoke'});
device = validatestring(device,{'cpu','mps','cuda','auto','cpu_gpu'});
validateattributes(cpuWorkers,{'numeric'},{'scalar','real','finite','integer','nonnegative'});
validateattributes(gpuWorkers,{'numeric'},{'scalar','real','finite','integer','nonnegative'});
assert(cpuWorkers + gpuWorkers > 0,'QALLMGPU:Workers','Enable at least one CPU or GPU worker.');
assert((ischar(outputDir) && (isrow(outputDir) || isempty(outputDir))) || ...
    (isstring(outputDir) && isscalar(outputDir)), ...
    'QALLMGPU:Argument','outputDir must be one path or empty.');
outputDir = char(outputDir);
yearText = encode_years(years);
folder = fileparts(mfilename('fullpath'));
script = fullfile(folder,'example_03_produce_tile.py');
assert(isfile(script),'QALLMGPU:Script','Python tile example is missing: %s',script);
args = {pythonExe,'-B','-u',script,'--data-root',dataRoot,'--years',yearText, ...
    '--device',device,'--cpu-workers',sprintf('%d',cpuWorkers), ...
    '--gpu-workers',sprintf('%d',gpuWorkers),'--threads','1'};
if strcmp(mode,'check'); args{end+1} = '--check-only'; end
if strcmp(mode,'smoke'); args{end+1} = '--smoke'; end
if ~isempty(outputDir); args = [args,{'--output',outputDir}]; end
quoted = cellfun(@literal_argument,args,'UniformOutput',false);
status = system(strjoin(quoted,' '),'-echo');
assert(status == 0,'QALLMGPU:PythonFailure', ...
    'Tile example exited with status %d. See the Python output above.',status);
end

function text = encode_years(value)
if ischar(value) || (isstring(value) && isscalar(value))
    assert(isrow(char(value)),'QALLMGPU:Years','Pass one year expression.');
    value = regexprep(strtrim(char(value)),'\s','');
    assert(~isempty(regexp(value,'^\d{4}(:\d{4}|(,\d{4})*)$','once')), ...
        'QALLMGPU:Years','Use a year, an inclusive year range, or a comma-separated year list.');
    if contains(value,':')
        bounds = str2double(strsplit(value,':'));
        value = bounds(1):bounds(2);
    else
        value = str2double(strsplit(value,','));
    end
end
validateattributes(value,{'numeric'}, ...
    {'vector','real','finite','integer','>=',2001,'<=',2024,'nonempty'});
value = sort(double(value(:).'));
assert(numel(unique(value)) == numel(value),'QALLMGPU:Years','Years must not be repeated.');
if isscalar(value)
    text = sprintf('%d',value);
elseif all(diff(value) == 1)
    text = sprintf('%d:%d',value(1),value(end));
else
    text = strjoin(arrayfun(@(year) sprintf('%d',year),value,'UniformOutput',false),',');
end
end

function quoted = literal_argument(value)
assert((ischar(value) && (isrow(value) || isempty(value))) || ...
    (isstring(value) && isscalar(value)), ...
    'QALLMGPU:Argument','Each command argument must be one string.');
value = char(value);
assert(~any(ismember(value,[char(0),newline,char(13)])), ...
    'QALLMGPU:Argument','Command arguments cannot contain NUL or line breaks.');
if ispc
    assert(~any(ismember(value,[char(34),char(37),char(33)])), ...
        'QALLMGPU:WindowsPath','Use resolved Windows paths without double quotes, percent signs or exclamation marks.');
    trailing = 0;
    while trailing < numel(value) && value(end-trailing) == char(92)
        trailing = trailing + 1;
    end
    quoted = [char(34),value,repmat(char(92),1,trailing),char(34)];
else
    replacement = [char(39),char(34),char(39),char(34),char(39)];
    quoted = [char(39),strrep(value,char(39),replacement),char(39)];
end
end
