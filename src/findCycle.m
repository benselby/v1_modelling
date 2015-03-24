function found = findCycle(A, partialPath)
    % Looks for cycles in a network connectivity matrix and prints the path of
    % the first one found. 
    % 
    % A: connectivity matrix (ones in each row are in-connections) 
    % partialPath: a row vector that contains indices of nodes in the beginning
    %   of a path to check; this can be just a starting node
    
    found = 0;
    children = find(A(:,partialPath(end)) == 1);
    for i = 1:length(children)
        newPath = [partialPath children(i)]; 
        if any(partialPath == children(i))
            disp(newPath)
            found = 1;
            break
        else 
            if findCycle(A, newPath)
                found = 1;
                break
            end
        end
    end
end