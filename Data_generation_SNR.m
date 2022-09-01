symbols1 = readmatrix("experimental_data");  %symbols alphabet 3 carriers
symbols2 = readmatrix("simulation_data");

symbols= symbols2;   %Change everytime you chose a different file to generata data

SNR_values = readmatrix("SNR_data");  %matrix with the SNR-e values calculated when amplifiers are used.
[numRows,numCols] = size(symbols)

n_samples_per_symbol=100000;
t = (0:1:n_samples_per_symbol-1)';
P_tags = 30:1:50;
G_tags =[1 2 100];

v1 =[];
v2 =[];
v3 =[];

GT =[];

for a = 1:numRows

    for b = 43:58
        x = symbols(a,1).*ones(size(t));
        y = awgn(x,SNR_values(1,b),'measured');
        y= abs(y);
        v1= [v1;y];
    end

    for b = 43:58
        x = symbols(a,2).*ones(size(t));
        y = awgn(x,SNR_values(2,b),'measured');
        y= abs(y);
        v2= [v2;y];
    end

    for b = 43:58
        x = symbols(a,3).*ones(size(t));
        y = awgn(x,SNR_values(3,b),'measured');
        y= abs(y);
        Ptag = P_tags(b-42).*ones(size(t));
        Gtag = G_tags(1).*ones(size(t));
        Stag = a.*ones(size(t));
        v3= [v3;y Stag Ptag Gtag];
    end

     
end

GT=[v1 v2 v3];
%GT = GT(randperm(size(GT, 1)), :);

writematrix(GT,'simulation_30_50.csv') 


plot(t,[x y])
legend('Original Signal','Signal with AWGN')