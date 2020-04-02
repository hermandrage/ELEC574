function [thetaest,P] = RLS(y,x,thetaest,P)
% RLS
% y,x: current measurement and regressor
%thetaest, P: parameter estimates and covariance matrix
lambda = 0.93; %forgetting factor
K = P*x/(lambda+x'*P*x);
P = (P- (P*x*x'*P)/(lambda+x'*P*x))*(1/lambda);
thetaest = (thetaest +K*(y-x'*thetaest))';

end

