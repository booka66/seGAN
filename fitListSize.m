function averaged_list = fitListSize(original_list, target_list)
    % Determine the size of the target list
    target_size = length(target_list);

    % Determine the size of the original list
    original_size = length(original_list);

    % Create an array representing the indices of the target list
    target_indices = 1:target_size;

    % Determine the indices where the original list will be evaluated
    original_indices = linspace(1, original_size, target_size);

    % Interpolate or average values to match the size of the target list
    averaged_list = interp1(1:original_size, original_list, original_indices, 'linear');
end