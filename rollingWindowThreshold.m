function logical_list = rollingWindowThreshold(data1, data2, window_size, threshold1, threshold2)
    % Initialize logical list
    logical_list = false(1, length(data1));

    % Iterate through the data vectors using a rolling window
    for i = 1:length(data1) - window_size + 1
        % Extract data points within the window
        window_data1 = data1(i:i+window_size-1);
        window_data2 = data2(i:i+window_size-1);

        % Check if both data points within the window exceed their thresholds
        if mean(window_data1 > threshold1) && mean(window_data2 > threshold2)
            % Set corresponding indices to true, including window before and after
            logical_list(max(1,i-3*window_size):min(length(data1),i+window_size-1)) = true;
        end
    end
end