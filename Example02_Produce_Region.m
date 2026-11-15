function Example02_Produce_Region(variable, mode, pythonExe, outputDir, device, cpuWorkers, gpuWorkers)
%EXAMPLE02_PRODUCE_REGION Run the frozen 200x200 example from MATLAB.
% No arguments check the installation without starting production.
% Example02_Produce_Region('both','smoke','python','','cpu_gpu',5,5)
% Example02_Produce_Region('both','full','python','/path/output','auto',5,5)
% Device is cpu, mps, cuda, auto, or cpu_gpu. Python owns all computation;
% MATLAB GPU and Parallel Computing toolboxes are not required.
% Auto uses CPU+GPU when available and CPU otherwise; zero disables a lane.
if nargin < 1 || isempty(variable); variable = 'both'; end
if nargin < 2 || isempty(mode); mode = 'check'; end
if nargin < 3 || isempty(pythonExe); pythonExe = 'python'; end
if nargin < 4 || isempty(outputDir); outputDir = ''; end
if nargin < 5 || isempty(device); device = 'auto'; end
if nargin < 6 || isempty(cpuWorkers); cpuWorkers = 5; end
if nargin < 7 || isempty(gpuWorkers); gpuWorkers = 5; end
variable = validatestring(variable,{'LAI','FPAR','both'});
mode = validatestring(mode,{'check','smoke','full'});
device = validatestring(device,{'cpu','mps','cuda','auto','cpu_gpu'});
validateattributes(cpuWorkers,{'numeric'},{'scalar','real','finite','integer','nonnegative'});
validateattributes(gpuWorkers,{'numeric'},{'scalar','real','finite','integer','nonnegative'});
assert(cpuWorkers + gpuWorkers > 0,'QALLMGPU:Workers','Enable at least one CPU or GPU worker.');
assert((ischar(outputDir) && (isrow(outputDir) || isempty(outputDir))) || ...
    (isstring(outputDir) && isscalar(outputDir)), ...
    'QALLMGPU:Argument','outputDir must be one path or empty.');
outputDir = char(outputDir);
folder = fileparts(mfilename('fullpath'));
script = fullfile(folder,'example_02_produce_region.py');
assert(isfile(script),'QALLMGPU:Script','Python example is missing: %s',script);
args = {pythonExe,'-B','-u',script,'--variable',variable,'--device',device, ...
    '--cpu-workers',sprintf('%d',cpuWorkers),'--gpu-workers',sprintf('%d',gpuWorkers), ...
    '--threads','1'};
if strcmp(mode,'check'); args{end+1} = '--check-only'; end
if strcmp(mode,'smoke'); args{end+1} = '--smoke'; end
if ~isempty(outputDir); args = [args,{'--output',outputDir}]; end
quoted = cellfun(@literal_argument,args,'UniformOutput',false);
status = system(strjoin(quoted,' '),'-echo');
assert(status == 0,'QALLMGPU:PythonFailure', ...
    'Region example exited with status %d. See the Python output above.',status);
end

function quoted = literal_argument(value)
assert((ischar(value) && (isrow(value) || isempty(value))) || ...
    (isstring(value) && isscalar(value)), ...
    'QALLMGPU:Argument','Each command argument must be one string.');
value = char(value);
assert(~any(ismember(value,[char(0),newline,char(13)])), ...
    'QALLMGPU:Argument','Command arguments cannot contain NUL or line breaks.');
if ispc
    % cmd expands percent/delayed-expansion markers even inside quotes.
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
