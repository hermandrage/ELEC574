function [Ca, Ha] = make_CaHa(dim,A)
    Ca = zeros(dim); 
    Ha = zeros(dim, length(A)-1);
    for i=1:min(dim,length(A))
        Ca(i:dim+1:(dim*(dim-i+1))) = A(i);
        Ha(i,:) = [A(i+1:end)' zeros(1,i-1)];
    end
end

