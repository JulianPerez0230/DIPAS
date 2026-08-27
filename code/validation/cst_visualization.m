%% Visualizacion Matematica de la Parametrizacion CST
% Este script genera las tres figuras clave para explicar el metodo CST
% (Class Shape Transformation) en una tesis de grado.

clear; clc; close all;

%% 1. Parametros y Definicion de la Cuerda
N_points = 500;
x = linspace(0, 1, N_points)'; % Posiciones x/c a lo largo de la cuerda

% Grado del polinomio de Bernstein (N = 5 para 6 coeficientes)
n = 5; 

%% 2. Calculo de los Polinomios de Bernstein Base
B = zeros(N_points, n + 1);
for i = 0:n
    % Coeficiente binomial: n! / (i! * (n-i)!)
    coef_binomial = nchoosek(n, i);
    % i-esimo polinomio de Bernstein
    B(:, i+1) = coef_binomial .* (x.^i) .* ((1 - x).^(n - i));
end

% Figura 1: Polinomios de Bernstein Base
figure('Color', 'w', 'Position', [100, 100, 750, 450]);
hold on;
colors = lines(n+1);
for i = 0:n
    plot(x, B(:, i+1), 'LineWidth', 2, 'Color', colors(i+1, :), ...
         'DisplayName', sprintf('B_{%d,5}(x)', i));
end
% Graficar la suma de todos los polinomios (debe ser constante = 1)
plot(x, sum(B, 2), '--k', 'LineWidth', 1.5, 'DisplayName', '\Sigma B_{i,5}(x) = 1');

grid on;
xlabel('Posición normalizada en la cuerda (x/c)', 'FontSize', 12);
ylabel('Valor del Polinomio', 'FontSize', 12);
title('Polinómios de Bernstein de Grado 5 (Base CST)', 'FontSize', 14);
legend('Location', 'eastoutside', 'FontSize', 10);
hold off;

%% 3. Coeficientes CST y Funciones de Forma S(x)
% Coeficientes aproximados para un perfil Selig S3021 (Bajo Reynolds)
a_upper = [0.135, 0.142, 0.178, 0.125, 0.138, 0.115];
a_lower = [-0.082, -0.065, -0.110, -0.052, -0.068, -0.055];

% Funcion de Forma S(x) como suma ponderada de Bernstein
S_upper = B * a_upper';
S_lower = B * a_lower';

% Funcion de Clase C(x)
C = sqrt(x) .* (1 - x);

% Figura 2: Funcion de Clase y Funciones de Forma
figure('Color', 'w', 'Position', [150, 150, 750, 450]);
subplot(2, 1, 1);
plot(x, C, 'r', 'LineWidth', 2.5);
grid on;
ylabel('C(x)', 'FontSize', 12);
title('Función de Clase C(x) = x^{0.5}(1 - x)', 'FontSize', 12);

subplot(2, 1, 2);
hold on;
plot(x, S_upper, 'b', 'LineWidth', 2, 'DisplayName', 'S_u(x) (Extradós)');
plot(x, S_lower, 'Color', [0 0.5 0], 'LineWidth', 2, 'DisplayName', 'S_l(x) (Intradós)');
grid on;
xlabel('Posición normalizada en la cuerda (x/c)', 'FontSize', 12);
ylabel('S(x)', 'FontSize', 12);
title('Funciones de Forma Combinadas S(x)', 'FontSize', 12);
legend('Location', 'best');
hold off;

%% 4. Perfil Alar Final: Combinacion C(x) * S(x)
% Espesor de borde de fuga relativo (t/c = 0.003, equivalente a ~0.6mm)
te_thickness = 0.003; 

% Ecuacion final de CST
y_upper = C .* S_upper + x .* (te_thickness / 2);
y_lower = C .* S_lower - x .* (te_thickness / 2);

% Figura 3: Perfil Alar Resultante
figure('Color', 'w', 'Position', [200, 200, 800, 350]);
hold on;
plot(x, y_upper, 'b', 'LineWidth', 2.5, 'DisplayName', 'Extradós (C\cdotS_u + x\cdot\Delta t/2)');
plot(x, y_lower, 'Color', [0 0.5 0], 'LineWidth', 2.5, 'DisplayName', 'Intradós (C\cdotS_l - x\cdot\Delta t/2)');
fill([x; flipud(x)], [y_upper; flipud(y_lower)], [0.9 0.9 0.9], 'FaceAlpha', 0.3, 'HandleVisibility', 'off');

% Graficar la linea de cuerda media como referencia
plot([0, 1], [0, 0], ':k', 'LineWidth', 1, 'HandleVisibility', 'off');

grid on;
axis equal; % Proporcion geometrica real 1:1
xlim([-0.05, 1.05]);
ylim([-0.2, 0.2]);
xlabel('Posición en la cuerda (x/c)', 'FontSize', 12);
ylabel('Espesor normalizado (y/c)', 'FontSize', 12);
title('Perfil Alar Reconstruido mediante CST (Selig S3021)', 'FontSize', 14);
legend('Location', 'northeast');
hold off;

disp('Visualizaciones generadas de forma exitosa.');
