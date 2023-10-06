function resultStruct = structArrayToStructWithArrays(inputStructArray)
% Turn a struct array into a singular struct with arrays of values.

    % Get the field names of the input struct array
    fieldNames = fieldnames(inputStructArray);

    % Initialize the result struct
    resultStruct = struct();

    % Loop through each field and convert it into an array
    for i = 1:numel(fieldNames)
        fieldName = fieldNames{i};
        values = [inputStructArray.(fieldName)]; % Extract values from the struct array
        resultStruct.(fieldName) = values;
    end
end