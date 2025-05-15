import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#DATA SET FROM GOOGLE ML COURSE

# Wczytanie danych
dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

training_dataset = dataset[['TRIP_MILES', 'FARE', 'TRIP_SECONDS']]

print("Training dataset loaded successfully\n")
print(training_dataset.head(10))

# Statystyki opisowe danych treningowych
print("Dataset statistics")
print(training_dataset.describe(include='all'))

# MODEL METHODS
def build_model(learning_rate, n_features):
    #Definicja topologi modelu
    inputs = keras.Input(shape=(n_features,))   # tworzy warstwę wejściową modelu z podaną liczbą cech
    outputs = keras.layers.Dense(units=1)(inputs)     # output layer units = 1 jeden neuron
    model = keras.Model(inputs=inputs, outputs=outputs) # łączenie warstw w pełen model

    # kompilacja modelu
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss="mean_squared_error", #ustawienie MSE
                  metrics=[keras.metrics.RootMeanSquaredError()]) #dodadkowe monitorowanie pierwiastek z MSE
    print("Model built successfully\n")
    return model

def train_model(model, features, label, epchos, batch_size):
    #Trenowanie modelu funkcja fit
    history = model.fit(x=features,
                        y=label,
                        epochs=epchos,
                        batch_size=batch_size)
    #Zapis danych
    trained_weights = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    print("model trained successfully\n")
    return trained_weights, trained_bias, epchos, rmse
    

def model_info(model_output):
    print("========== MODEL INFO ========== ")
    print(f'WEIGHTS: {model_output[0]}')
    print(f'BIAS: {model_output[1]}')
    print("================================ ")

def loss_curve(epchos, RMSE_history):
    x = [i for i in range(epchos)]
    plt.figure()
    plt.plot(x, RMSE_history.tolist(), linewidth=2, color="red")
    plt.title('Loss curve')
    plt.xlabel('Epchos')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def plot_model(dataset, bias, weight, features_name, label_name, sample_size=200):
    sample_dataset = dataset.sample(n=sample_size, random_state=42)
    x = sample_dataset[features_name[0]].values
    y = sample_dataset[label_name].values
    plt.figure()
    plt.plot(x, y, color='red', linestyle='None', marker='o', markersize=5)
    xx = np.linspace(0, max(x), 200);
    yy = [weight * i + bias for i in xx]
    plt.plot(xx, yy, color='blue', linestyle='--', linewidth=2)
    plt.title('Model')
    plt.xlabel(features_name[0])
    plt.ylabel(label_name)
    plt.grid(True)
    plt.show()


def make_plots(model_output, dataset, features_names, label_name):
    print("PLOTS MAKER")
    loss_curve(model_output[2], model_output[3])
    #Only for one features
    if len(features_names) == 1:
        plot_model(dataset, model_output[1][0], model_output[0][0][0], features_names, label_name)



def get_model(dataset, features_names, label_name, learning_rate, epchos, batch_size):
    print("INFO: starting training experiment\n")
    print(f"features = {features_names}, label = {label_name}\n")
    
    n_features = len(features_names)

    #Tworzenie macierzy 2D
    features = dataset.loc[:, features_names].values # : - wszystkie wiersze features_name - tylko te kolumny
    label = dataset.loc[:, label_name].values # values konwertuje na numpy array
    
    model = build_model(learning_rate, n_features)
    model_output = train_model(model, features, label, epchos, batch_size)

    model_info(model_output)
    make_plots(model_output, dataset, features_names, label_name)

    return model

def show_predictions(model, dataset, features, label):
    sample_dataset = dataset.sample(n=30, random_state=42)

    sample_features = sample_dataset[features].values
    sample_label = sample_dataset[label].values

    # uruchomienie modelu
    predictions = model.predict(sample_features)

    #Wyniki w dataframe
    result_dataframe = pd.DataFrame({
        'TRIP_MILES': sample_dataset['TRIP_MILES'].values,
        'TRIP_SECONDS': sample_dataset['TRIP_SECONDS'].values,
        'FARE': sample_label,
        'PREDICTED_FARE': predictions.flatten()
    })
    
    result_dataframe['LOSS_L1'] = abs(result_dataframe['FARE'] - result_dataframe['PREDICTED_FARE'])

    print(result_dataframe)

#hyperparameters
learning_rate = 0.001
epchos = 20
batch_size = 50

features = ['TRIP_MILES']
label = 'FARE'

model1 = get_model(training_dataset, features, label, learning_rate, epchos, batch_size)
show_predictions(model1, training_dataset, features, label)

features = ['TRIP_MILES', 'TRIP_SECONDS']
model2 = get_model(training_dataset, features, label, learning_rate, epchos, batch_size)
show_predictions(model2, training_dataset, features, label)

model2.save("linear_regression_model.keras")