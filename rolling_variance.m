function variance_values = rolling_variance(data,sampRate)
    window_size = round(sampRate/10);
    num_windows = length(data) - window_size + 1;
    variance_values = zeros(1, num_windows);
    
    for i = 1:num_windows
        window_data = data(i:i+window_size-1);
        variance_values(i) = var(window_data);
    end
end