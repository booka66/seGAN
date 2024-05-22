function cleaned_logical_list = szLenLim(logical_list, threshold)
    % Initialize cleaned logical list
    cleaned_logical_list = logical_list;

    % Initialize a counter to track consecutive true values
    consecutive_true_count = 0;

    % Iterate through the logical list
    for i = 1:length(logical_list)
        if logical_list(i) == true
            % Increment consecutive true count
            consecutive_true_count = consecutive_true_count + 1;
        else
            % Check if consecutive true count is less than threshold
            if consecutive_true_count <= threshold
                % Set consecutive true values to false
                cleaned_logical_list(i - consecutive_true_count:i - 1) = false;
            end
            % Reset consecutive true count
            consecutive_true_count = 0;
        end
    end
end