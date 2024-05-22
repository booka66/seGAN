function [data_cell,total_channels,sampRate,NRecFrames] = vectorized_state_extractor(FileName)

FilePath = FileName;

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

for i = 1:total_channels
    reshaped_full_data(:, i) = full_data(i:total_channels:end);
end

full_data = reshaped_full_data;
[Rows, Cols] = getChs(FileName);

data = struct();
for x = 1:64
    for y = 1:64
        data(x, y).signal = zeros(NRecFrames, 1);
        data(x, y).name = [x y];
        data(x, y).SzEventsTimes = [];
        data(x, y).SE_List = [];
    end
end

t = (0:(1/sampRate):((1/sampRate) * (NRecFrames - 1)))';

temp_data = cell(total_channels, 1);

parfor k = 1:total_channels
    chNum = k;
    tgt_cols = [Cols(chNum)];
    tgt_rows = [Rows(chNum)];
    tgt_indexes = zeros(1, length(tgt_rows));
    
    for i = 1:length(tgt_rows)
        [rowIndex, ~] = find((channels(:,1) == tgt_cols(i)) & (channels(:,2) == tgt_rows(i)));
        tgt_indexes(i) = rowIndex;
    end
    
    channel_data = zeros(NRecFrames, length(tgt_indexes));
    
    channel_data = full_data(:, tgt_indexes);
    channel_data = (channel_data * ADCCountsToMV) + double(MVOffset);
    channel_data = channel_data / 1000000;
    channel_data = channel_data - mean(channel_data, 1);
    
    signal = channel_data(:, 1);
    
    [SzEventsTimes, SE_List, Sz_power_list] = getSzEnvelop_wSE2(signal, sampRate, t);
    
    %%append Sz_power_list
    SzEventsTimes = [SzEventsTimes Sz_power_list'];
    %%

    temp_data{k} = struct('signal', signal, ...
                          'name', [tgt_rows tgt_cols], ...
                          'SzEventsTimes', SzEventsTimes, ...
                          'SE_List', SE_List);
end

for k = 1:total_channels
    tgt_rows = temp_data{k}.name(1);
    tgt_cols = temp_data{k}.name(2);
    data(tgt_rows, tgt_cols) = temp_data{k};
end

data_cell = cell(1, 64*64);
index = 1;
for x = 1:64
    for y = 1:64
        data_cell{index} = data(x, y);
        index = index + 1;
    end
end

end
