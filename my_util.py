from sklearn.utils import shuffle

def Randomise(df, random_seed):
    df = shuffle(df, random_state=random_seed)
    df = df.reset_index(drop=True) # do not insert a new column with the new index
    return df


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# num of outputs = 1
num_outputs = 1

# num of inputs = 8 or 9
def my_model(num_inputs, num_nodes, extra_depth):
# create model
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))

    for i in range(extra_depth):
        model.add(Dense(num_nodes, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(num_outputs, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
