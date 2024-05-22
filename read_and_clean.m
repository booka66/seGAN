function [channel_data, sampRate, NRecFrames] = read_and_clean(FilePath, tgt_cols, tgt_rows)
%READ_AND_CLEAN takes an hdf5 brainwave file and cleans it
%   INPUTS: file_path, array of target columns, array of target rows
%   OUTPUTS: array of channel data, sampRate, NRecFrames
    recElectrodeList = h5read(FilePath, '/3BRecInfo/3BMeaStreams/Raw/Chs');
    NRecFrames = h5read(FilePath, '/3BRecInfo/3BRecVars/NRecFrames');
    NRecFrames = double(NRecFrames);
    sampRate = h5read(FilePath, '/3BRecInfo/3BRecVars/SamplingRate');
    signalInversion = h5read(FilePath, '/3BRecInfo/3BRecVars/SignalInversion');
    maxUVolt = h5read(FilePath, '/3BRecInfo/3BRecVars/MaxVolt');
    maxUVolt = double(maxUVolt);
    minUVolt = h5read(FilePath, '/3BRecInfo/3BRecVars/MinVolt');
    minUVolt = double(minUVolt);
    bitDepth = h5read(FilePath, '/3BRecInfo/3BRecVars/BitDepth');
    qLevel = bitxor(2,bitDepth);
    qLevel = double(qLevel);
    fromQLevelToUVolt = (maxUVolt - minUVolt) / qLevel;
    
    ADCCountsToMV = signalInversion * fromQLevelToUVolt;
    ADCCountsToMV = double(ADCCountsToMV);
    MVOffset = signalInversion * minUVolt;
    rows = recElectrodeList.Row;
    cols = recElectrodeList.Col;
    channels = horzcat(cols, rows);
    total_channels = length(channels(:,1));
    
    full_data = h5read(FilePath, '/3BData/Raw');
    full_data = double(full_data);
    
    reshaped_full_data = zeros(NRecFrames, total_channels);
    
    for i=1:total_channels
        reshaped_full_data(:, i) = full_data(i:total_channels:end);
    end
    full_data = reshaped_full_data;
    
    tgt_indexes = zeros(1, length(tgt_rows));
    for i = 1:length(tgt_rows)
        [rowIndex, ~] = find((channels(:,1) == tgt_cols(i)) & (channels(:,2) == tgt_rows(i)));
        tgt_indexes(i) = rowIndex;
    end
    
    channel_data = zeros(NRecFrames, length(tgt_indexes));
    for i = 1:length(tgt_indexes)
        channel_data(:, i) = full_data(:, tgt_indexes(i));
        channel_data(:, i) = (channel_data(:, i) * ADCCountsToMV) + MVOffset;
        channel_data(:, i) = channel_data(:, i) / 1000000;
        channel_data(:, i) = channel_data(:, i) - mean(channel_data(:, i)); %Set to baseline
    end
end