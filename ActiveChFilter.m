function [activeList] = ActiveChFilter(FileName, Threshold)

% Pull available channels
[Rows, Cols] = getChs(FileName);

% Initialize active list
activeList = [];

% Preallocate a cell array to store intermediate results
tempActiveList = cell(1, length(Cols));

% Parallelize the loop
parfor k = 1:length(Cols)
    chNum = k;
    tgt_cols = [Cols(chNum)];
    tgt_rows = [Rows(chNum)];
    [channels, sampRate, NRecFrames] = read_and_clean(FileName, tgt_cols, tgt_rows);
    t = (0:(1/sampRate):((1/sampRate) * (NRecFrames - 1)))';

    % Set vars
    data1 = channels(:, 1);

    % Set data vars
    V = data1(sampRate*60:length(data1));

    if max(V) > Threshold || min(V) < (-Threshold)
        tempActiveList{k} = chNum;
    else
        tempActiveList{k} = [];
    end
end

% Concatenate the intermediate results into a single array
activeList = [tempActiveList{:}];

end