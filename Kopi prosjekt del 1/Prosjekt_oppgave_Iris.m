clear all; 
clc; 

%% 
%testing

%for å bytte plass på 30 første eller siste som traing:
% Bytt plass på 30 og 20 i linjene under
%Må også bytte Q og T, samt P og S alle plasser i test-seksjonen

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%VED ENDRING AV ANTALL FEATURES MED FØLGANDE ENDRAST:
% MATRISE W
% FEAUTURE-VEKTOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

feauture = [1,2,3,4];
W = zeros(3,5);
W(:,5) = 1.2;
w0 = [0.1; 0.2; 0.3;];
W(:,1:2) = 0.4;
W(:,3) = 1;
W(:,4) = 0.5;
W(:,1) = w0;

disp("test start");
disp("Weigthed matrix with feature [2] (the worst feature)");

%30 første som trening, 20 siste test
[Q, T] = splitList(readFile("blomster.txt"), 30, 20, feauture);
[P, S] = defineClass(30, 20);

%30 siste som trening, 20 første test
%[T, Q] = splitList(readFile("blomster.txt"), 20, 30, [1,2,3,4]);
%[S, P] = defineClass(20, 30);

%30 første som både trening og test
%[Q, ~] = splitList(readFile("blomster.txt"), 30, 20, [1,2,3,4]);
%[P, ~] = defineClass(30, 20);
%T = Q;
%S = P;

% 30 siste som både trening og test
%[~, T] = splitList(readFile("blomster.txt"), 20, 30, [1,2,3,4]);
%[~, S] = defineClass(20, 30);
%Q = T;
%P = S;

%disp(T);
%disp(Q);
%disp(T);

%Definerer alpha og ein matrise W

alpha = 0.05;
number_iteration = 5000; 

%W = zeros(3,5);
%w0 = [0.1; 0.2; 0.3;];
%W(:,1:2) = 0.4;
%W(:,3) = 1;
%W(:,4) = 0.5;
%W(:,5) = 1.2;
%W(:,1) = w0;

%%

%disp("test start");

%r1 = Q(2:3,1:30);
%r2 = Q(2:3,31:60);
%r3 = Q(2:3,61:90);
%disp(r1(1,:));
%r2 = Q(1:2,1:end);
%scatter(r1(1,:), r1(2,:), "filled");
%hold on;
%scatter(r2(1,:), r2(2,:), "filled");
%hold on;
%scatter(r3(1,:), r3(2,:), "filled");
%hold off; 


% B er ferdigtrent matrise med gitte parameter
% Trening
B = stop_iteration(W, number_iteration, alpha, P, Q);

%Testing

% Z er matrise_vektor-produkt mellom B og testeksempel  
Z = B*T;

% G gir ei matrise med i dimensjon (3,N) med typiske kolonner (0.9, 0.7, 0.5)
G = sigmoid(Z);

% TC tilegner kvar testblom ein klasse. Tek inn Sigmoid matrisa G, og ser
% på kva kolonneelement som har høgast verdi
TC = testClass(G);
%disp(TC);

% P6 regnar ut total error rate
P6 = error_rate(TC, S, length(S)/3);

% gir ut kolonneposisjon på feilklassifiseringar. Den gir også ut gjetta
% klasse vs faktisk klasse
P1 = position_error(TC, S);

% P3 seier kor mange feil det er i klasse 1, 2 og 3
P3 = number_of_mistakes(length(S)/3, P1);

% P4 gir ut vektor med oversikt over kor mange feil ein har med gjetta
% klasse vs faktisk klasse.
P4 = mistakes(P1);

disp("Total error rate = "); disp(P6);

% P5 gir ut confusion matrix
P5 = confusion_matrix(length(S)/3, P4, P3);
P7 = confusion_matrix_percent(P5, length(S)/3);
%disp("confusion matrix with the 30 first as training example and 30 first as test:");
disp(P5);
disp(P7);

%disp("confusion matrix in percentage with the 30 first as training example and 30 first as test:");
%disp(P7);

%disp(split_feautures_hist(Q, length(Q)/3));

%%

%Her frå og ned gjeld berre dersom ein ynskjer å finne overlap
%Her må ein vere litt obs på antall feautures

%P8 = split_feautures_hist(Q, length(Q)/3);
%P9 = plot_histogram(P8, 3 ,10);
%P10 = find_mu(P8);

%P11 = find_number_of_overlap(P8, 1);
%P12 = find_number_of_overlap(P8, 2);
%P13 = find_number_of_overlap(P8, 3);
%P14 = find_number_of_overlap(P8, 4);
%P15 = [P11;P12;P13;P14];


%[P20, pos] = find_most__to_least_overlap_data(P15);

%disp("overlap score");
%disp("Array from left to right represent feauture with most to least overlap");
%disp(P20);
%disp(pos);


disp("test end");


%%

function split_feautures = split_feautures_hist(X, size)
    %Deler opp heile lista X (liste med alle verdiar på blomster) og gir ut
    %den valgte parameteren for kvar blom
    
    %Må endre manuelt når ein endrer antall features!!!
    
    array1 = transpose([X(2,1:size); X(2,size+1:2*size); X(2,2*size+1:end)]);
    array2 = transpose([X(3,1:size); X(3,size+1:2*size); X(3,2*size+1:end)]);
    array3 = transpose([X(4,1:size); X(4,size+1:2*size); X(4,2*size+1:end)]);
    %array4 = transpose([X(5,1:size); X(5,size+1:2*size); X(5,2*size+1:end)]);
    
    split_feautures = [array1, array2, array3];%, array4];
end

function [most_to_least_overlap, pos] = find_most__to_least_overlap_data(X)
    %Her må ein ta inn matrisa frå squared_list funksjonen
    %Minst tal = størst overlapp
    %Størst tal = minst overlapp

    L = length(X(:,1));
    sum_vec = zeros(1,L);

    for i = 1:L
        sum_vec(1,i) = sum(X(i,:));
    end    
    
    [most_to_least_overlap, pos] = sort(sum_vec);
    
end


function number_of_overlap = find_number_of_overlap(X, wanted_feature)
    %Gir ut vektor med length(bin_edges) antall bins.
    %Kvart vektorelement gir den kvadrerte summen av differansen i antall
    %samples i kvar bin
    N = wanted_feature;
    bin_edges = (0:39)./4;

    [N1, ~] = histcounts(X(:,N*3-2), bin_edges);
    [N2, ~] = histcounts(X(:,N*3-1), bin_edges);
    [N3, ~] = histcounts(X(:,N*3), bin_edges);

    N4 = (N1-N2).^2;
    N5 = (N1-N3).^2;
    N6 = (N2-N3).^2;

    number_of_overlap = N4+N5+N6; 
end


function histo = plot_histogram(split_features, wanted_feature, bins)
    %Plotter ønska histogram
    N = wanted_feature;

    histogram(split_features(:,N*3-2), bins);
    hold on;

    histogram(split_features(:,N*3-1), bins);
    hold on;

    histogram(split_features(:,N*3), bins);
    hold off;

    histo = [];

end

function number_of_mistakes = number_of_mistakes(test, pos)
    %A = feil i klasse 1
    %B = feil i klasse 2
    %C = feil i klasse 3

    A = length(find(pos(:,3) <= test));
    B = length(find(pos(:,3) > test & pos(:,3) <= 2*test));
    C = length(find(pos(:,3) > 2*test & pos(:,3) <= 3*test));
    
    number_of_mistakes = [A, B, C];
    
end

function mistakes = mistakes(pos)

    %funksjon returnerer vektor med [C1_2, C1_3, C2_1, C2_3, C3_1, C3_2]
    %som rekkefølge

    C1_2 = length(find(pos(1:end,1) == 1 & pos(1:end,2) == 2));
    C1_3 = length(find(pos(1:end,1) == 1 & pos(1:end,2) == 3));

    C2_1 = length(find(pos(1:end,1) == 2 & pos(1:end,2) == 1));
    C2_3 = length(find(pos(1:end,1) == 2 & pos(1:end,2) == 3));
    
    C3_1 = length(find(pos(1:end,1) == 3 & pos(1:end,2) == 1));
    C3_2 = length(find(pos(1:end,1) == 3 & pos(1:end,2) == 2));

    %disp("numbers come in this order:");
    %disp("C1_2, C1_3, C2_1, C2_3, C3_1, C3_2");

    mistakes = [C1_2, C1_3, C2_1, C2_3, C3_1, C3_2];

end

function confusion_matrix_percent = confusion_matrix_percent(D, test)
    % Gir confusion matrix i prosent
    C = D;
    for i = 2:4
        for j = 2:4
        C{i,j} = C{i,j} / test;
        end
    end    
    confusion_matrix_percent = C;
end

function confusion_matrix = confusion_matrix(test, mistakes, number_mistakes)
    %Gir confusion matrisa i absoluttverdi


    D = cell(4,4);
    D{1,1} = "Classified/True";
    D{2,1} = "Class 1";
    D{3,1} = "Class 2";
    D{4,1} = "Class 3";

    D{1,2} = "Class 1";
    D{1,3} = "Class 2";
    D{1,4} = "Class 3";

   
    D{2,2} = test-number_mistakes(1);
    D{3,3} = test-number_mistakes(2);
    D{4,4} = test-number_mistakes(3);

    D{3,2} = mistakes(1);
    D{4,2} = mistakes(2);

    D{2,3} = mistakes(3);
    D{4,3} = mistakes(4);

    D{2,4} = mistakes(5);
    D{3,4} = mistakes(6);
   
    confusion_matrix = D; 
end


function position = position_error(X, T)
    %gir ut kolonneposisjon på feilklassifiseringar
    %funksjonen tek inn argumenta
    % X = output frå class_array funksjonen
    % T = faktisk klasse

    D = X-T;
    [row_guess, ~] = find(D > 0);
    [row_true, col] = find(D < 0);
    position = [row_guess, row_true, col];
end

function error_rate = error_rate(X, T, test)
    %Funksjonen finn error rate
    %funksjonen tek inn argumenta
    % X = output frå class_array funksjonen
    % T = faktisk klasse

    D = X-T;
    col = find(D > 0);

    error_rate = length(col) / (3*test);
end


function class_array = testClass(X)
    %funksjonen tek inn sigmoid-matrise

    L1 = length(X(:,1));
    L2 = length(X);
    class_array = zeros(L1, L2);

    for i = 1:L2
        [~, P] = max(X(:,i));
        class_array(P,i) = 1; 
    end    
end


function W_final = stop_iteration(W_initial, number_iteration, alpha, T, X)
    W_final = W_initial;
    for i = 1:number_iteration
        nabla_W = derivation(T, X, W_final);
        W_final = W_final - alpha*nabla_W;
    end    
end


%Definerer nabla_W_MSE
function nabla_W = derivation(T, X, W)
    
    nabla_W = zeros(length(W(:,1)), length(W(1,:)));
    L = length(X);
    Z = W*X;
    G = sigmoid(Z);
    vec1 = (G-T).*G.*(1-G);
  
    for i = 1:L
        nabla_W = nabla_W + 0.5*vec1(:,i)*transpose(X(:,i));
    end 
end


%definerer sigmoidfunksjonen med ein matrise som input
function GK = sigmoid(X)
    GK = zeros(length(X(:,1)),length(X));
    gk = zeros(length(X(:,1)),1);

    for i=1:length(X) 
        for j = 1:length(X(:,1))
            gk(j,1) = 1./(1+exp(-X(j,i)));
        end  
    GK(:,i) = gk;  
    end
end

%funksjon som leser textfil
function TxtScan1 = readFile(file)

%Leser frå fil
fileID = fopen(file,'r');
A = fscanf(fileID, '%s');

%Fjerner alle ord i lista
newStr1 = erase(A, "Iris-setosa");
newStr2 = erase(newStr1, "Iris-versicolor");
newStr3 = erase(newStr2, "Iris-virginica");

%Deler opp tekst og gjer om string til double
TxtScan = textscan(newStr3,'%4s','Whitespace','');
TxtScan1 = str2double(TxtScan{:});

%Lukker fil
fclose(fileID);
end


%funksjon som gir arrays avhengig av kor mange test- og treningeksempel ein
%vil ha
%function [traininglist, testlist] = splitList(X, train, test)
    
    %reshaper lista
%    K = reshape(X, 4, []);   
%    L_X = length(K)/3;

%    %Definerer ein treningsliste og ein testliste
%    traininglist = zeros(5,3*train);
%    testlist = zeros(5,3*test);

    %Gir treningslista #train antall av kvar klasse til trening
%    traininglist(1,1:end) = 1;
%    traininglist(2:5,1:train) = K(1:4,1:train);
%    traininglist(2:5,train+1:2*train) = K(1:4,L_X:L_X+train-1);
%    traininglist(2:5,2*train+1:end) = K(1:4,2*L_X:2*L_X +train-1);

    %Gir testlista #test antall av kvar klasse til testing
%    testlist(1,1:end) = 1;
%    testlist(2:5,1:test) = K(1:4,train+1:train+test);
%    testlist(2:5,test+1:2*test) = K(1:4,L_X+train+1:L_X+train+test);
%    testlist(2:5,2*test+1:end) = K(1:4,2*L_X+train+1:2*L_X+train+test);

%end

function [traininglist, testlist] = splitList(X, train, test, feautures)
    
    %reshaper lista
   
    N = length(feautures) + 1;
    K = reshape(X, 4, []);   
    L_X = length(K)/3;

    %Definerer ein treningsliste og ein testliste
    traininglist = zeros(N,3*train);
    testlist = zeros(N,3*test);

    %Gir treningslista #train antall av kvar klasse til trening
    traininglist(1,1:end) = 1;
    testlist(1,1:end) = 1;

    for i = 1:N-1

        traininglist(i+1, 1:train) = K(feautures(i) , 1:train);
        traininglist(i+1, train+1:2*train) = K(feautures(i), L_X: L_X + train-1);
        traininglist(i+1, 2*train+1:end) = K(feautures(i), 2*L_X: 2*L_X + train-1);

        
        testlist(i+1, 1:test) = K(feautures(i), train+1:train+test);
        testlist(i+1, test+1:2*test) = K(feautures(i), L_X+train+1:L_X+train+test);
        testlist(i+1, 2*test+1:end) = K(feautures(i), 2*L_X+train+1:2*L_X+train+test);
    end    

end

    
%funksjon som gir ut riktig klasse til kvart eksemplar
function [T_train, T_test] = defineClass(train, test)
    %Tilegnar kvar blom ein klasse.
   
    if train+test <= 50

    T_train = zeros(3,3*train);
    T_test = zeros(3,3*test);
    
    T_train(1,1:train) = 1;
    T_train(2,train+1:2*train) = 1;
    T_train(3,2*train+1:end) = 1;

    T_test(1,1:test) = 1;
    T_test(2,test+1:2*test) = 1;
    T_test(3,2*test+1:end) = 1;

    else
    T_train = [];
    T_test = [];
    disp("wrong dimensions for defineClass");
    end

end  





