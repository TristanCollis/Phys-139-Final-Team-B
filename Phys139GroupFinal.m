% Imported and formatted data, Graphed it
data = [9  3 11 16 14  9  4 15  7 17 10  7 11  8 10  3  5 11  6  9  4  9 ...
    10 8 12  9 11  4  8 11 21 13 12  6  6  6 14 10 12  9 15  8  8  8 15 ... 
    10 12  8 11 21 11 13  9  7 15  8 17  4  7 10  5 12 13 10 10  8  8  4 ...
    15 11  9 12 12 13 17 17 10 15 15 20 10 18 19 15 18 21 21 15 16 14 15 ... 
    18 13 17 19 14 11 11 18 17 10 12 27 19 15 19 14 12 16 12 14  8 16 10 ...
    20  7 19 16 15 21 18 15 12 18 12 25 23 19 15 12 22 18 20 17 17 13 23 ...
    25 14 14 16  7 13 13 12 18 14 16 19 22 13 13 11 16 19 12 24 17 20 22 ...
    13 12 17 13 12 16 19 23 16 25 15 12 10 17 10 12 10 17 18 13 20 12 15 15];
x = 2017:2200;
scatter(x, data)
xlabel('Year')
ylabel('Gravitational Wave Detection Events')
% Initializing and preallocating before simulations
a = 8;
b = 1;
prob = zeros(183, 1);
n_0 = 103;
reps = 100000;
n = zeros(reps, 1);
lambdas = zeros(reps, 2);
% Starting simulations
for k = 1:reps
    % Lambda 1
    data_1 = data(1:n_0);
    a_1 = a + sum(data_1);
    b_1 = b + n_0;
    lambda_1 = gamrnd(a_1,1/b_1);
    % Lambda 2
    data_2 = data(n_0 + 1: 184);
    a_2 = a + sum(data_2);
    b_2 = b + 183 - n_0;
    lambda_2 = gamrnd(a_2,1/b_2);
    lambdas(k,1) = lambda_1;
    lambdas(k,2) = lambda_2;
    % new n_0
    for m = 1:183
        data_m = data(1:m);
        data_m1 = data(m+1:184);
        P = exp(log(lambda_1*sum(data_m) - (m*lambda_1)) ...
            + log(lambda_2*sum(data_m1) - ((184 - m)*lambda_2)));
        prob(m, 1) = P;
    end
    q = sum(prob);
    pnorm = prob / q;
    r = rand(1);
    for l = 1:183
        a_k = sum(pnorm(1:l-1));
        b_k = sum(pnorm(1:l));
        if r < b_k && r >= a_k
            n_0 = l;
            n(k, 1) = n_0;
        end
    end
end
% Getting rid of burn-in and rescaling
n = n(1000:reps, 1);
lambdas = lambdas(1000:reps, 1:2);
n = n + 2016;
% Plotting
histogram(n)
title('Probability Distribution of n_0');
xlabel('Year');
ylabel('P(n_0|x(1:N))');
l_1 = lambdas(1:(reps - 999), 1);
l_2 = lambdas(1:(reps - 999), 2);
scatter(l_1, l_2)
title('Lambda 1 vs Lambda 2');
xlabel('Lambda 1');
ylabel('Lambda 2');
% Computing Means
ave_n_0 = mean(n);
ave_l_1 = mean(l_1);
ave_l_2 = mean(l_2);


