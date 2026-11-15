function Example01_Train_Validate(variable, stage, pythonExe, mode, outputDir)
% Public small-data training example. Python performs the numerical work.
% Example01_Train_Validate('LAI', 'all', 'python')
if nargin < 1, variable = 'LAI'; end
if nargin < 2, stage = 'prepare'; end
if nargin < 3, pythonExe = 'python'; end
if nargin < 4, mode = 'tutorial'; end
if nargin < 5, outputDir = ''; end
variable = validatestring(variable, {'LAI','FPAR','both'});
stage = validatestring(stage, {'prepare','train','budget','evaluate','uncertainty','all'});
mode = validatestring(mode, {'tutorial','smoke','check'});
folder = fileparts(mfilename('fullpath'));
script = fullfile(folder, 'example_01_train_validate.py');
assert(~contains(pythonExe, '"') && ~contains(outputDir, '"'), ...
    'Do not include quote characters in executable/output paths.');
extra = '';
if strcmp(mode, 'smoke'), extra = ' --smoke'; end
if strcmp(mode, 'check'), extra = ' --check-only'; end
if ~isempty(outputDir), extra = [extra sprintf(' --output "%s"', outputDir)]; end
cmd = sprintf('"%s" -u "%s" --variable "%s" --stage "%s"', ...
    pythonExe, script, variable, stage);
cmd = [cmd extra];
status = system(cmd, '-echo');
assert(status == 0, 'Training example failed. See Python output above.');
end
