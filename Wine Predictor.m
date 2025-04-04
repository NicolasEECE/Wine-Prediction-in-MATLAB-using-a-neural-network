% Φόρτωση δεδομένων
data = load('wine.data');

% Διαχωρισμός δεδομένων εκπαίδευσης
rows_1_to_34 = data(1:34,:);
rows_60_to_98 = data(60:98,:);
rows_131_to_156 = data(131:156,:);
train_data = [rows_1_to_34; rows_60_to_98; rows_131_to_156];

% Διαχωρισμός δεδομένων ελέγχου
rows_35_to_59 = data(35:59,:);
rows_99_to_130 = data(99:130,:);
rows_157_to_178 = data(157:178,:);
test_data = [rows_35_to_59; rows_99_to_130; rows_157_to_178];

% Καθορισμός εισόδων και εξόδων για την εκπαίδευση και τον έλεγχο
input_train = train_data(:, 2:14);
output_train = train_data(:, 1);
input_test = test_data(:, 2:14);
output_test = test_data(:, 1);

% Καθορισμός παραμέτρων του νευρωνικού δικτύου
hiddenLayerSizes = [100, 50]; 
net = patternnet(hiddenLayerSizes, 'trainlm'); 
net.trainParam.show = 50;
net.trainParam.epochs = 500;
net.trainParam.lr = 0.01;
net.trainParam.min_grad = 1e-5;

% Εκπαίδευση του νευρωνικού δικτύου
net = train(net, input_train', output_train');            

% Πρόβλεψη με το εκπαιδευμένο νευρωνικό δίκτυο
output_predicted = sim(net, input_test'); 

% Σχεδιασμός των αποτελεσμάτων
figure;
plot(output_test, 'o');
hold on;
plot(output_predicted, 'x');
legend('Actual', 'Predicted');
xlabel('Data Points');
ylabel('Class');
title('Actual vs Predicted Wine Classes');
hold off;

% Αξιολόγηση της ακρίβειας των προβλέψεων
accuracy = sum(round(output_predicted) == output_test') / length(output_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
