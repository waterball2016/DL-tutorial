function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 

epslion = 10^-4;
epslion_vector = zeros(size(theta));

for i = 1:size(theta)
    epslion_vector(i) = epslion;
    numgrad(i) = (J(theta + epslion_vector) - J(theta - epslion_vector)) * (2*epslion)^-1;
    epslion_vector(i) = 0;
end

disp(numgrad);




%% ---------------------------------------------------------------
end
