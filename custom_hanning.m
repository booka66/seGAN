% custom hanning function
function window = custom_hanning(sampling_frequency, signal_length)
    hanning_length = length(0:(2/sampling_frequency):2)-1;
    hanning_window = hann(hanning_length);   % make this just the length of 2 seconds; one second before and one second after the interesting signal; fs, dt
    plateau_length = signal_length - hanning_length;
    flat = ones(1, plateau_length);
    flat = flat';
    max_hann = max(hanning_window);
    index_to_insert = find(hanning_window == max_hann, 1) + 1;
    if ~isempty(index_to_insert)
        part1 = hanning_window(1:(index_to_insert-1));
        part2 = hanning_window((index_to_insert):end);
        window = vertcat(part1, flat, part2);
    else
        window = hanning_window;
    end
end