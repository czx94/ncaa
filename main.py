import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype="int64", name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

def create_bias(inp, n_in):
    x = Embedding(n_in, 1, input_length=1)(inp)
    return Flatten()(x)


from keras.models import Model

# Input data files are available in the "../input/" directory
# from subprocess import check_output
# print(check_output(["ls", "./input"]).decode("utf8"))

#visualize data
dr = pd.read_csv("./input/RegularSeasonDetailedResults.csv")
# print(dr.head(n=30))

# Preparing the training data Simple win to 1 loss to 0 encoding
simple_df_1 = pd.DataFrame()
simple_df_1[["team1", "team2"]] =dr[["WTeamID", "LTeamID"]].copy()
simple_df_1["pred"] = 1

simple_df_2 = pd.DataFrame()
simple_df_2[["team1", "team2"]] =dr[["LTeamID", "WTeamID"]]
simple_df_2["pred"] = 0

simple_df = pd.concat((simple_df_1, simple_df_2), axis=0)
simple_df.head()

n = simple_df.team1.nunique()

trans_dict = {t: i for i, t in enumerate(simple_df.team1.unique())}
simple_df["team1"] = simple_df["team1"].apply(lambda x: trans_dict[x])
simple_df["team2"] = simple_df["team2"].apply(lambda x: trans_dict[x])
print(simple_df.head())

train = simple_df.values
np.random.shuffle(train)

#model
n_factors = 50

team1_in, t1 = embedding_input("team1_in", n, n_factors, 1e-4)
team2_in, t2 = embedding_input("team2_in", n, n_factors, 1e-4)

b1 = create_bias(team1_in, n)
b2 = create_bias(team2_in, n)

x = merge([t1, t2], mode="dot")
x = Flatten()(x)
x = merge([x, b1], mode="sum")
x = merge([x, b2], mode="sum")
x = Dense(1, activation="sigmoid")(x)
model = Model([team1_in, team2_in], x)
model.compile(Adam(0.001), loss="binary_crossentropy")

model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
history = model.fit([train[:, 0], train[:, 1]], train[:, 2], batch_size=128, nb_epoch=100, verbose=2, callbacks=[earlystopper])

plt.plot(history.history["loss"])
plt.show()

sub =  pd.read_csv('./input/SampleSubmissionStage1.csv')
sub["team1"] = sub["ID"].apply(lambda x: trans_dict[int(x.split("_")[1])])
sub["team2"] = sub["ID"].apply(lambda x: trans_dict[int(x.split("_")[2])])
sub.head()

sub["pred"] = model.predict([sub.team1, sub.team2])
sub = sub[["ID", "pred"]]
sub.head()

sub.to_csv("CF.csv", index=False)




