%Importation des donnees

data = readtable('Data_TP.xlsx');
SPX = flip(data.SPXIndex(7:end,1));
Rf_CAN = flip((data.Canada1YearRate(7:end,1)))/100;
USD_CAD = flip(data.USDCADBGNCurncy(7:end,1));
Rf_US_12 = flip((data.USGG12MIndex(7:end,1)))/100;
Rf_US_6 = flip((data.USGG6MIndex(7:end,1)))/100;
Rf_US_1 = flip((data.USGG1MIndex(7:end,1)))/100;
Vol_hist = flip((data.SPXIndex_1(7:end,1)))/100;
Vol_imp = flip((data.SPXIndex_2(7:end,1)))/100; 
t = flip(data.Security(7:2521));

%% Cleaning

SPX = SPX(~isnan(SPX));
Rf_CAN = Rf_CAN(~isnan(Rf_CAN));
USD_CAD = USD_CAD(~isnan(USD_CAD));
Rf_US_12 = Rf_US_12(~isnan(Rf_US_12));
Rf_US_6 = Rf_US_6(~isnan(Rf_US_6));
Rf_US_1 = Rf_US_1(~isnan(Rf_US_1));
Vol_hist = Vol_hist(~isnan(Vol_hist));
Vol_imp = Vol_imp(~isnan(Vol_imp));
t = t(1:end,1);

%% 1- Volatilité implicite et S&P500

SPX_1 = SPX(1:end-1);
t_1 = t(1:end-1);

% Création du graphique avec deux axes verticaux
figure;
yyaxis left; % Premier axe vertical
plot(t_1, SPX_1, '-b', 'LineWidth', 1.5); % Courbe de SPX en bleu
ylabel('SPX');
xlabel('Temps');
grid on;

yyaxis right; % Second axe vertical
plot(t_1, Vol_imp, '-r', 'LineWidth', 1.5); % Courbe de Vol_imp en rouge
ylabel('Volatilité Implicite');

% Personnalisation
title('Volatilité implicite et S&P500');
legend({'SPX', 'Volatilité Implicite'}, 'Location', 'best');

%% 2- Volatilité historique vs implicite

Vol_hist1 = Vol_hist(1:end-1);
figure;
yyaxis left; % Premier axe vertical
plot(t_1, Vol_imp, '-k', 'LineWidth', 1.5); % Courbe de Vol_imp en noir
ylabel('Volatilite implicite ');
xlabel('Temps');
grid on;

yyaxis right; % Second axe vertical
plot(t_1, Vol_hist1, '-r', 'LineWidth', 1.5); % Courbe de Vol_hist en rouge
ylabel('Volatilité historique ');

% Personnalisation
title('Volatilité historique Vs implicite');
legend({'Volatilité implicite', 'Volatilité historique'}, 'Location', 'best');

%% 3- Option put at the money 1 an, vendue le 31 décembre 2019 :

% a - Calculez l'évolution du prix de l'option put jusqu'à sa maturité

% Paramètres
K = SPX(1296,1); % Prix d'exercice
r = (data.USGG12MIndex(1270,1))/100; % Taux sans risque au 31 décembre 2019
sigma = data.SPXIndex_2(1226,1)/100; % Volatilité implicite initiale
T = 1; % Maturité de 1 an
S_t = SPX(1296:1549,1); % Evolution du prix du sous-jacent

% Calcul de l'évolution du prix de l'option put
d1 = (log(S_t / K) + (r + sigma^2 / 2) * T) ./ (sigma * sqrt(T));
d2 = d1 - sigma * sqrt(T);
P = K * exp(-r * T) .* normcdf(-d2) - S_t .* normcdf(-d1);

% EVolution graphique

figure;
plot (t(1296:1549,1),P);
xlabel("Temps")
ylabel("Evolution du prix de l'option")


%% b- Évaluez les paramètres de couverture, les "Grecs", à chaque date.

Delta = blsdelta(S_t,K,r,T,sigma); % Delta
Gamma = normpdf(d1) ./ (S_t * sigma * sqrt(T)); % Gamma
Vega = S_t .* sqrt(T) .* normpdf(d1); % Vega

%ux = blsdelta(S_t,K,r,T,sigma);
%ug = blsgamma(S_t,K,r,T,sigma); Test BLS
%uv = blsvega(S_t,K,r,T,sigma);

%Graphe Delta
figure
plot(t(1296:1549,1),Delta,'-r')
xlabel('Temps')
ylabel('Delta')

%Graphe Gamma
figure
plot(t(1296:1549,1),Gamma,'g')
xlabel('Temps')
ylabel('Gamma')

%Graphe Vega
figure
plot(t(1296:1549,1),Vega,'b')
xlabel('Temps')
ylabel('Vega')

%% c - Déterminez le portefeuille de réplication de la banque à chaque date en utilisant le delta hedging
% Présentez les résultats sur un graphique.

% Portefeuille de réplication
Quantite_option = 1; % Supposons 1 option vendue
Position_sous_jacent = Delta * Quantite_option;

% Graphique du portefeuille

figure;
plot(t(1296:1549,1), Position_sous_jacent, '-b', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Position sous-jacente');
title('Portefeuille de réplication via Delta Hedging');

%% d. Calcul et analyse des gains et pertes quotidiens

dS = diff(S_t); % Variation du sous-jacent
dDelta = diff(Delta); % Variation du Delta
PnL = Delta(1:end-1) .* dS - abs(dDelta) .* S_t(1:end-1);

% Graphique des Pnl

figure;
plot(t(1297:1549), PnL, '-r', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Gains/Pertes quotidiens');
title('Gains et pertes (Pnl) du delta hedging');
grid on;

%% 4- Utiliser blsbinomialprice/ blsprice
%Option put at the money 1 an, américaine vendue le 31 décembre 2019 

N = 252;
dt = T / N; % Durée d'un pas
u = exp(sigma * sqrt(dt)); % Facteur de montée
d = 1 / u; % Facteur de descente
p = (exp(r * dt) - d) / (u - d); % Probabilité risque neutre

% a) Arbre binomial

S_tree = zeros(N+1, N+1); % Initialisation de la matrice de l'arbre
S_tree(1,1) = SPX(1296); % Prix initial du sous-jacent

% Remplissage de l'arbre
for i = 2:N+1
    for j = 1:i
        if j == 1
            S_tree(j, i) = S_tree(j, i-1) * u; % Montée
        else
            S_tree(j, i) = S_tree(j-1, i-1) * d; % Descente
        end
    end
end


V_tree = zeros(N+1, N+1); % Initialisation de l'arbre de l'option
V_tree(:, end) = max(K - S_tree(:, end), 0); % Valeur terminale de l'option

% Rétropropagation dans l'arbre pour évaluer l'option
for i = N:-1:1
    for j = 1:i
        % Calcul de la valeur de maintien (Hold Value)
        hold_value = exp(-r * dt) * (p * V_tree(j, i+1) + (1-p) * V_tree(j+1, i+1));
        
        % Valeur intrinsèque de l'option (Put)
        exercise_value = max(K - S_tree(j, i), 0);
        
        % Valeur de l'option : soit exercice, soit maintien
        V_tree(j, i) = max(hold_value, exercise_value);
    end

    
end


% Prix de l'option américaine
price_american = V_tree(1,1);

% Identifier les dates optimales d'exercice
exercise_dates = zeros(N, 1); % Initialisation des dates d'exercice

for i = 1:N % Parcourir les étapes (colonnes de l'arbre)
    for j = 1:i % Parcourir les nœuds à chaque étape
        hold_value = exp(-r * dt) * (p * V_tree(j, i+1) + (1-p) * V_tree(j+1, i+1));
        exercise_value = max(K - S_tree(j, i), 0);
        
        % Vérifier si l'option est exercée
        if exercise_value > hold_value
            exercise_dates(i) = 1; % Marquer comme une date d'exercice
        end
    end
end

Optimal_date = table(t(1296:1547,1), exercise_dates);
Opt = numel(find(exercise_dates==1)); %Pour compter le nombre de 1
disp('Le nombre de dates optimales pour exercer l''option:')
disp(Opt)

figure
plot(t(1296:1547,1), exercise_dates,'-r', 'LineWidth', 1.5);
title('Dates optimales d''exercice');
%histogram(exercise_dates); 

%% 4b - Comparer le prix de l'option américaine à celui de l'option européenne

% Initialisation de l'arbre pour l'option européenne
V_tree_european = zeros(N+1, N+1); % Arbre des valeurs de l'option européenne
V_tree_european(:, end) = max(K - S_tree(:, end), 0); % Valeur intrinsèque à maturité

% Rétropropagation dans l'arbre pour l'option européenne
for i = N:-1:1 % Parcourir les colonnes de droite à gauche
    for j = 1:i % Parcourir les nœuds à chaque étape
        % Calcul de la valeur actualisée (sans exercice anticipé)
        V_tree_european(j, i) = exp(-r * dt) * ...
            (p * V_tree_european(j, i+1) + (1-p) * V_tree_european(j+1, i+1));
    end
end

% Prix de l'option européenne
price_european = V_tree_european(1,1);

% Affichage de la comparaison
fprintf('Prix de l''option américaine : %.2f\n', price_american);
fprintf('Prix de l''option européenne : %.2f\n', price_european);

% Vérification supplémentaire (différence)
fprintf('Différence entre les prix (américaine - européenne) : %.2f\n', price_american - price_european);

%% 4c - Calcul des "Grecs"

% Delta : Sensibilité au prix du sous-jacent
Delta_tree = zeros(N, 1);
for i = 1:N
    Delta_tree(i) = (V_tree(1, i+1) - V_tree(2, i+1)) / (S_tree(1, i+1) - S_tree(2, i+1));
end

% Gamma : Sensibilité du Delta au prix du sous-jacent
Gamma_tree = zeros(N-1, 1);
for i = 1:N-1
    Gamma_tree(i) = ((V_tree(1, i+2) - V_tree(2, i+2)) / (S_tree(1, i+2) - S_tree(2, i+2)) ...
        - (V_tree(2, i+2) - V_tree(3, i+2)) / (S_tree(2, i+2) - S_tree(3, i+2))) / ...
        (0.5 * (S_tree(1, i+2) - S_tree(3, i+2)));
end


% Theta : Sensibilité au temps
Theta_tree = zeros(N, 1);
for i = 1:N
    Theta_tree(i) = (V_tree(1, i+1) - V_tree(1, i)) / dt;
end

% Vega : Sensibilité à la volatilité
Vega_tree = zeros(N, 1);
for i = 1:N
    % Calcul local pour la volatilité augmentée
    sigma_plus = sigma + 0.01;
    u_plus = exp(sigma_plus * sqrt(dt)); % Nouveau facteur de montée
    d_plus = 1 / u_plus; % Nouveau facteur de descente
    p_plus = (exp(r * dt) - d_plus) / (u_plus - d_plus); % Nouvelle probabilité neutre au risque

    % Calcul de V_tree_plus(1, i) uniquement au sommet de l'arbre
    hold_value_plus = exp(-r * dt) * ...
        (p_plus * V_tree(1, i+1) + (1-p_plus) * V_tree(2, i+1));
    exercise_value_plus = max(K - S_tree(1, i), 0);
    V_tree_plus_1_i = max(hold_value_plus, exercise_value_plus);

    % Calcul de Vega pour le pas courant
    Vega_tree(i) = (V_tree_plus_1_i - V_tree(1, i)) / 0.01;
end


figure;
plot(1:N, Delta_tree, '-r', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Delta');
title('Évolution du Delta dans le temps');

figure;
plot(1:N-1, Gamma_tree, '-g', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Gamma');
title('Évolution du Gamma dans le temps');

figure;
plot(1:N, Vega_tree, '-b', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Vega');
title('Évolution du Vega dans le temps');

figure;
plot(1:N, Theta_tree, '-m', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Theta');
title('Évolution du Theta dans le temps');

%% 4d - Portefeuille de réplication via delta hedging


% Quantité d'options
Quantite_option = 1; % Par exemple, une seule option

% Initialisation des matrices
Delta = zeros(N, 1); % Stocker Delta à chaque étape
Position_sous_jacent = zeros(N, 1); % Position sous-jacente à chaque étape

% Calcul des positions sous-jacentes via Delta
for i = 1:N
    % Calcul de Delta pour le nœud supérieur à chaque étape
    Delta(i) = (V_tree(1, i+1) - V_tree(2, i+1)) / (S_tree(1, i+1) - S_tree(2, i+1));
    
    % Calcul de la position sous-jacente en fonction de Delta
    Position_sous_jacent(i) = Delta(i) * Quantite_option;
end

% Graphique du portefeuille de réplication
figure;
plot(1:N, Position_sous_jacent, '-b', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Position sous-jacente');
title('Portefeuille de réplication via Delta Hedging');
grid on;


%% 4e - Gains et pertes quotidiens (PnL)

% Calcul des variations du sous-jacent (S) et du Delta
dS_tree = diff(S_tree(1, 1:N)); % Variation du sous-jacent sur les niveaux de l'arbre
dDelta_tree = diff(Delta(1:N)); % Variation du Delta (assurez-vous que Delta est de taille N)

% Calcul du PnL quotidien
PnL_tree = Delta(1:N-1) .* dS_tree - abs(dDelta_tree) .* S_tree(1, 1:N-1);

% Graphique des PnL
figure;
plot(1:(N-1), PnL_tree, '-r', 'LineWidth', 1.5);
xlabel('Temps');
ylabel('Gains/Pertes (PnL)');
title('Gains et pertes quotidiens du delta hedging');
grid on;



%% 5 - Tarification par simulation

% 3 octobre 2024 = t(2493:2493,1)

% i) Option digitale à la monnaie sur 1 an

% Parametres

S0 = SPX(2493);
K = S0; % A la monnaie
r = (data.USGG12MIndex(28))/100; % Rf 12 mois le 03/10/2024
sigma = (data.SPXIndex_2(29))/100; % Volatilite implicite pour les options
T = 1; % 1 an
Payout = 1; % Payoff de l'option
N = 1000; % Nombre de simulations

% Simulation Monte Carlo

Z = randn(N, 1); % Variables normales
ST = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) .* Z); % Chemins simulés
Payoff = Payout * (ST > K); % Payoff pour chaque chemin
Prix = exp(-r * T) * mean(Payoff); % Prix de l'option
fprintf('Prix option digitale Monte Carlo: %.4f\n', Prix);


%% Arbre binomial

% Paramètres
n = 100; % Nombre de pas
dt = T / n;
u = exp(sigma * sqrt(dt));
d = exp(-sigma * sqrt(dt));
p = (exp(r * dt) - d) / (u - d);

% Construction de l'arbre
ST = zeros(n+1, n+1);
for i = 0:n
    for j = 0:i
        ST(j+1, i+1) = S0 * u^(i-j) * d^j;
    end
end

% Payoff à maturité
Payoff = Payout * (ST(:, end) > K);

% Backpropagation
for i = n:-1:1
    Payoff = exp(-r * dt) * (p * Payoff(1:end-1) + (1-p) * Payoff(2:end));
end
Prix = Payoff(1);
fprintf('Prix option digitale Binomial: %.4f\n', Prix);


%% Bonus : visuel graphique reduit

n_graph = 10; % Nombre de pas pour le graphique 

% Paramètres pour le graphique
dt_graph = T / n_graph;
u_graph = exp(sigma * sqrt(dt_graph));
d_graph = exp(-sigma * sqrt(dt_graph));

% Construction de l'arbre simplifié pour le graphique
ST_graph = zeros(n_graph+1, n_graph+1);
for i = 0:n_graph
    for j = 0:i
        ST_graph(j+1, i+1) = S0 * u_graph^(i-j) * d_graph^j;
    end
end

% Positionner les nœuds pour le graphique
x_graph = [];
y_graph = [];

for i = 1:n_graph+1
    for j = 1:i
        x_graph = [x_graph, i]; % Temps (périodes)
        y_graph = [y_graph, ST_graph(j, i)]; % Prix du sous-jacent
    end
end

% Tracer les connexions entre nœuds
figure;
hold on;
for i = 1:n_graph
    for j = 1:i
        % Connexion au nœud supérieur
        plot([i, i+1], [ST_graph(j, i), ST_graph(j, i+1)], '-k');
        % Connexion au nœud inférieur
        plot([i, i+1], [ST_graph(j, i), ST_graph(j+1, i+1)], '-k');
    end
end

% Tracer les nœuds
scatter(x_graph, y_graph, 50, 'filled', 'b'); % Nœuds bleus

% Personnalisation du graphique
xlabel('Temps (périodes)');
ylabel('Prix du sous-jacent');
title('Arbre Binomial Simplifié pour le Graphique');
grid on;


%% ii) Option avec barrière "down and out

% Simulation Monte Carlo avec barrière
B = 4500;
M = 252; % Nombre de pas (jours dans une année)
dt = T / M;
ST_path = zeros(N, M+1);
ST_path(:, 1) = S0;
knocked_out = false(N, 1);

for t = 2:M+1
    Z = randn(N, 1);
    ST_path(:, t) = ST_path(:, t-1) .* exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) .* Z);
    knocked_out = knocked_out | (ST_path(:, t) < B);
end

% Payoff
Payoff = Payout * (ST_path(:, end) > K) .* (~knocked_out);
Prix = exp(-r * T) * mean(Payoff);
fprintf('Prix option barrière Monte Carlo: %.4f\n', Prix);

%% (AJOUTER L'ARBRE BINOMIAL)
ST = zeros(n+1, n+1);
ST(1, 1) = S0; % Le prix initial du sous-jacent
for i = 2:n+1
    for j = 1:i
        ST(j, i) = S0 * u^(i-j) * d^(j-1); % Calcul des prix
        if ST(j, i) < B
            ST(j, i) = NaN; % Invalide si la barrière est franchie
        end
    end
end

% Payoff à maturité
Payoff = zeros(n+1, 1);
for j = 1:n+1
    if ~isnan(ST(j, end)) && ST(j, end) > K
        Payoff(j) = Payout; % Payoff pour les prix au-dessus du strike
    end
end

% Backpropagation dans l'arbre
for i = n:-1:1
    for j = 1:i
        if ~isnan(ST(j, i)) % Vérifie que la barrière n'a pas été atteinte
            Payoff(j) = exp(-r * dt) * (p * Payoff(j) + (1-p) * Payoff(j+1));
        else
            Payoff(j) = 0; % Si barrière atteinte, le payoff est annulé
        end
    end
end

% Résultat final
Prix = Payoff(1);
fprintf('Prix option barrière Binomial: %.4f\n', Prix);

%% 6 : Tarifiez les options de la question 5 pour un investisseur 
% canadien en tenant compte du taux de change

% USD/CAD le 03-10-2024 : data.USDCADBGNCurncy(29:29,1)

K_CAD = 8020;
t_change = data.USDCADBGNCurncy(29:29,1); % Taux de change le 03/10
K_USD = K_CAD/t_change; % Prix en USD
S0 = SPX(2493);
r_CAN = (data.Canada1YearRate(29))/100;

%% Simulation Monte Carlo pour l'investisseur canadien

Z = randn(N, 1); % Variables normales
ST = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) .* Z); % Chemins simulés
Payoff = Payout * (ST > K_USD); % Payoff pour chaque chemin
Prix_USD = exp(-r * T) * mean(Payoff); % Moyenne des payoffs actualisés
Prix_CAD = Prix_USD * t_change; % Conversion en CAD
fprintf('Prix option digitale Monte Carlo (CAD) : %.4f\n', Prix_CAD);

%% Arbre binomial - Option digitale
n = 100; % Nombre de pas
dt = T / n;
u = exp(sigma * sqrt(dt));
d = exp(-sigma * sqrt(dt));
p = (exp(r * dt) - d) / (u - d);

% Construction de l'arbre
ST_bin = zeros(n+1, n+1);
for i = 0:n
    for j = 0:i
        ST_bin(j+1, i+1) = S0 * u^(i-j) * d^j;
    end
end

% Payoff à maturité
Payoff = Payout * (ST_bin(:, end) > K_USD);

% Backpropagation
for i = n:-1:1
    Payoff = exp(-r * dt) * (p * Payoff(1:end-1) + (1-p) * Payoff(2:end));
end
Prix_USD = Payoff(1); % Prix de l'option en USD
Prix_CAD = Prix_USD * t_change; % Conversion en CAD
fprintf('Prix option digitale Binomial (CAD) : %.4f\n', Prix_CAD);

%% Option barrière "down and out" - Monte Carlo
B = 4500; % Barrière en USD
M = 252; % Nombre de pas (jours dans une année)
dt = T / M;
ST_path = zeros(N, M+1);
ST_path(:, 1) = S0;
knocked_out = false(N, 1);

for t = 2:M+1
    Z = randn(N, 1);
    ST_path(:, t) = ST_path(:, t-1) .* exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) .* Z);
    knocked_out = knocked_out | (ST_path(:, t) < B);
end

% Payoff
Payoff = Payout * (ST_path(:, end) > K_USD) .* (~knocked_out);
Prix_USD = exp(-r * T) * mean(Payoff); % Prix de l'option en USD
Prix_CAD = Prix_USD * t_change; % Conversion en CAD
fprintf('Prix option barrière Monte Carlo (CAD) : %.4f\n', Prix_CAD);

%% Option barrière "down and out" - Arbre binomial
ST_bin = zeros(n+1, n+1);
ST_bin(1, 1) = S0; % Le prix initial du sous-jacent
for i = 2:n+1
    for j = 1:i
        ST_bin(j, i) = S0 * u^(i-j) * d^(j-1); % Calcul des prix
        if ST_bin(j, i ) < B
            ST_bin(j, i) = NaN; % Invalide si la barrière est franchie
        end
    end
end

% Payoff à maturité
Payoff = zeros(n+1, 1);
for j = 1:n+1
    if ~isnan(ST_bin(j, end)) && ST_bin(j, end) > K_USD
        Payoff(j) = Payout; % Payoff pour les prix au-dessus du strike
    end
end

% Backpropagation dans l'arbre
for i = n:-1:1
    for j = 1:i
        if ~isnan(ST_bin(j, i)) % Vérifie que la barrière n'a pas été atteinte
            Payoff(j) = exp(-r * dt) * (p * Payoff(j) + (1-p) * Payoff(j+1));
        else
            Payoff(j) = 0; % Si barrière atteinte, le payoff est annulé
        end
    end
end

% Résultat final
Prix_USD = Payoff(1); % Prix en USD
Prix_CAD = Prix_USD * t_change; % Conversion en CAD
fprintf('Prix option barrière Binomial (CAD) : %.4f\n', Prix_CAD);