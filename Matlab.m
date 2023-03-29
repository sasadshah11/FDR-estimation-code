load 'decoy (3).mat'
%tsv = tdfread('/Users/ayeshaferoz/Desktop/Parentdirectory/merged_file.tsv');
% Define folder containing the TSV files
folder = '/Users/ayeshaferoz/Desktop/Parentdirectory';

% Get a list of all TSV files in the folder
file_list = dir(fullfile(folder, '*.tsv'));

% Loop over each file and load its data
for i = 1:length(file_list)
    % Load the TSV file using importdata
    tsv = importdata(fullfile(folder, file_list(i).name), '\t');
    tsv.ScanNum = str2double(tsv.textdata(:,3));
    tsv.MonoisotopicMass = str2double(tsv.textdata(:,8));
    tsv.DummyIndex=str2double(tsv.textdata(:,4));
    tsv.QScore=str2double(tsv.textdata(:,22));
    tsv.Qvalue=str2double(tsv.textdata(:,23));


    % Do something with the data, for example:
    disp(['Loaded file ', file_list(i).name, ' with ', num2str(size(tsv)), ' data points.']);
end

trueset=[scan4 t4];
ppmtol = 10;
fdval = [tsv.ScanNum  tsv.MonoisotopicMass];
tpindex1 = ismember(fdval(:,1), trueset(:,1));
tpindex2 = sum(abs(fdval(tpindex1,2)' - trueset(:,2))./max(fdval(tpindex1,2)', trueset(:,2))*1e6 < ppmtol,1) > 0;
disp(size(tpindex1));
disp(size(tpindex2));
disp(size(tsv.DummyIndex));

% Ensure that all arrays have the same number of rows
n_rows = min([length(tpindex1), length(tpindex2), length(tsv.DummyIndex)]);
tpindex = tpindex1(1:n_rows) & tpindex2(1:n_rows)' & tsv.DummyIndex(1:n_rows) == 0;
fpindex = (~tpindex1(1:n_rows) | ~tpindex2(1:n_rows)') & tsv.DummyIndex(1:n_rows) == 0;
decoyindex = tsv.DummyIndex(1:n_rows) > 0;

figure1 = figure('defaultAxesFontSize',14);

histogram(tsv.QScore(fpindex),0:.025:1);

hold on;

histogram(tsv.QScore(decoyindex),0:.025:1);

grid on;

legend({'#false positives', '#decoy masses'},'Location', 'northwest')
xlabel('QScore')
ylabel('Count')

figure2 = figure('defaultAxesFontSize',14);
fp = histogram(tsv.QScore(fpindex),0:.025:1);
fpv = fp.Values;

dp = histogram(tsv.QScore(~decoyindex),0:.025:1);
dpv = dp.Values;

cdpv = zeros(size(dpv));
cfpv = zeros(size(fpv));

for i = 1:length(dpv)
cdpv(end-i+1) = sum(dpv(end-i+1:end));
cfpv(end-i+1) = sum(fpv(end-i:endl));
end

sample1 = randsample((dp.BinEdges(2:end) + dp.BinEdges(1:end-1))/2,1000,true,dpv);
sample2 = randsample((dp.BinEdges(2:end) + dp.BinEdges(1:end-1))/2,1000,true,fpv);

[h, p] = kstest2(sample1,sample2);
p;

plot((dp.BinEdges(2:end) + dp.BinEdges(1:end-1))/2,cfpv./cdpv, 'LineWidth', 1)
hold on;

tmp = [tsv.QScore(tsv.Decoy==0)' ; tsv.Qvalue(tsv.Decoy==0)']';
tmp = sortrows(tmp);

plot(tmp(:,1), tmp(:,2), '--', 'LineWidth', 1)
xlim([.4 1])
grid on;

xlabel('QScore')
ylabel('qvalue or FDR')
legend({'True FDR', 'qvalue'})

