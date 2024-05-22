function powerList = calculatePower(data, fs,freqRange, windowSize, overlap)
    % Calculate spectrogram
    [S, f, ~] = spectrogram(data, windowSize, overlap, [], fs);

    % Find indices corresponding to frequencies 1-50
    freqIndices = f >= freqRange(1) & f <= freqRange(2);

    % Calculate power for each frequency and time point
    powerMatrix = abs(S(freqIndices, :)).^2;

    %Sum the frequencies
    powerList = sum(powerMatrix, 1);
end