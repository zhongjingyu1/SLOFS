function [W_s,obj] = SLOFS(X,Y,n_class,alpha,beta,lamda1,lamda2,delta)

% Initialization
[nSamp,nFeat] = size(X);
[~,nlable] = size(Y);
E = rand(nSamp,n_class);
B = rand(n_class,n_class);
W_s = ones(nFeat,n_class);
V = rand(n_class,nlable);

% Constructing a similarity graph
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.t = 1e+4;
options.WeightMode = 'Heatkernel';
S2 = constructW(X,options);
D2 = diag(sum(S2));
Lx = D2 - S2;

for iter = 1:10
    % Update Z
    Z = max(E,0);
    % Constructing a dynamic similarity graph
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.t = 1e+4;
    options.WeightMode = 'Heatkernel';
    S1 = constructW(E',options);
    Ds = diag(sum(S1));
    LE = Ds - S1;
    % Update E
    AA = lamda1 * Lx;
    E1 = 2*X*W_s*B + 2*alpha.*Y*V' + 2*delta*Z;
    [~,y1] = eigs(AA);
    m1 = diag(y1);
    u1 = max(m1);
    AA1 = u1*eye(nSamp) - AA;
    P = AA1*E + E1 / 2;
    [Um,~,Vm] = svd(P);
    E = Um * eye(nSamp,n_class) * Vm';
    E = real(E);
    % Update B
    T1 = E'*X*W_s;
    [UB,~,VB] = svd(T1);
    B = VB*eye(n_class,n_class)*UB';
    % Update V
    V = V.*(2*E'*Y./(V+eps));
    % Update W
    Wi = sqrt(sum(W_s.*W_s,2)+eps) ;
    d = 0.5./Wi;
    Da = diag(d);
    W_s = W_s.*(X'*E*B'./((X'*X*W_s)+beta.*Da*W_s+(lamda2.*X'*X*W_s*LE)+eps));
    % Objective function
    obj(iter) = norm((X*W_s-E*B'),'fro')^2 + alpha*norm((Y-E*V),'fro')^2 + lamda1*trace(E'*Lx*E) +...
        lamda2*trace(X*W_s*LE*(X*W_s)') + beta*trace(W_s'*Da*W_s) + delta*norm((E-Z),'fro')^2;
end