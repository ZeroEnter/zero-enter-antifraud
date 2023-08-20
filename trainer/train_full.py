import numpy as np
import pandas as pd
import torch.nn as nn
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
import networkx as nx
import torch

import os
import json
import ezkl
import matplotlib.pyplot as plt

path_csv = "data/credit_card_transactions-ibm_v2.csv"
df = pd.read_csv(path_csv).sample(n=100000, random_state=42)


# The card_id is defined as one card by one user.
# A specific user can have multiple cards, which would correspond to multiple different card_ids for this graph.
# For this reason we will create a new column which is the concatenation of the column User and the Column Card
df["card_id"] = df["User"].astype(str) + "_" + df["Card"].astype(str)

# We need to strip the ‘$’ from the Amount to cast as a float
df["Amount"] = df["Amount"].str.replace("$", "").astype(float)

# time can't be casted to int so so opted to extract the hour and minute
df["Hour"] = df["Time"].str[0:2]
df["Minute"] = df["Time"].str[3:5]

# drop unnecessary columns
df = df.drop(["Time", "User", "Card"], axis=1)

# ERRORS:
# array([nan, 'Bad PIN', 'Insufficient Balance', 'Technical Glitch',
#        'Bad Card Number', 'Bad CVV', 'Bad Expiration', 'Bad Zipcode',
#        'Insufficient Balance,Technical Glitch', 'Bad Card Number,Bad CVV',
#        'Bad CVV,Insufficient Balance',
#        'Bad Card Number,Insufficient Balance'], dtype=object)

df["Errors?"] = df["Errors?"].fillna("No error")

# The two columns Zip and Merchant state contains missing values which can affect our graph.
# Moreover these information can be extracted from the column Merchant City so we will drop them.
df = df.drop(columns=["Merchant State", "Zip"], axis=1)

# change the is fraud column to binary
df["Is Fraud?"] = df["Is Fraud?"].apply(lambda x: 1 if x == "Yes" else 0)

df["Merchant City"] = LabelEncoder().fit_transform(df["Merchant City"])

# USE CHIP:
# array(['Chip Transaction', 'Online Transaction', 'Swipe Transaction'],
#       dtype=object)
df["Use Chip"] = LabelEncoder().fit_transform(df["Use Chip"])
df["Errors?"] = LabelEncoder().fit_transform(df["Errors?"])


# Create an empty graph


# Create an empty graph
G = nx.MultiGraph()

# Add nodes to the graph for each unique card_id, merchant_name
G.add_nodes_from(df["card_id"].unique(), type="card_id")
G.add_nodes_from(df["Merchant Name"].unique(), type="merchant_name")

# The code below adding edges and properties to the edges of a graph.
# The code iterates through each row of the dataframe, df,
# and creates a variable for each property then
# assign it to the edge between the card_id and merchant_name of that row.

# Add edges and properties to the edges
for _, row in df.iterrows():
    # Create a variable for each properties for each edge

    year = (row["Year"],)
    month = (row["Month"],)
    day = (row["Day"],)
    hour = (row["Hour"],)
    minute = (row["Minute"],)
    amount = (row["Amount"],)
    use_chip = (row["Use Chip"],)
    merchant_city = (row["Merchant City"],)
    errors = (row["Errors?"],)
    mcc = row["MCC"]

    G.add_edge(
        row["card_id"],
        row["Merchant Name"],
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        amount=amount,
        use_chip=use_chip,
        merchant_city=merchant_city,
        errors=errors,
        mcc=mcc,
    )

# Get the number of nodes and edges in the graph
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Print the number of nodes and edges
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)

# Convert the graph to an adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()

print(f"adj_matrix.shape: {adj_matrix.shape}")

# We define the variable "edge_list" which is a list of edges and their associated data in a graph G.
# Then we create an empty list called "x" and iterates over each edge in the edge_list.
# For each edge, it extracts the values of the edge data, converts them to floats if needed, and append them to the list "x".
# Finally, we convert the list "x" to a PyTorch tensor with float datatype

# Prepare the data for input into the model
edge_list = list(G.edges(data=True))
features = []
for edge in edge_list:
    edge_values = list(edge[2].values())
    edge_values = [
        float(i[0])
        if type(i) == tuple and type(i[0]) == str
        else i[0]
        if type(i) == tuple
        else i
        for i in edge_values
    ]
    features.append(edge_values)

print(f"features.shape: {len(features)}")

target = df["Is Fraud?"].values.tolist()


class Model(nn.Module):
    # define nn
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


train_X, test_X, train_y, test_y = train_test_split(
    features,  # use columns 0-4 as X
    target,  # use target as y
    test_size=0.2,  # use 20% of data for testing
)

# Uncomment for sanity checks
# print("train_X: ", train_X)
# print("test_X: ", test_X)
print("train_y: ", train_y)
print("test_y: ", test_y)


# our loss function
loss_fn = nn.CrossEntropyLoss()


model = Model()
# our optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# use 800 EPOCHS
EPOCHS = 800

# Convert training data to pytorch variables
tr_x = torch.Tensor(train_X).float()
tr_x = (tr_x - tr_x.mean(dim=0)) / tr_x.std(dim=0)
train_X = Variable(tr_x)

te_x = torch.Tensor(test_X).float()
te_x = (te_x - te_x.mean(dim=0)) / te_x.std(dim=0)
test_X = Variable(te_x)


train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())


loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))


# we use tqdm for nice loading bars
for epoch in tqdm.trange(EPOCHS):

    # To train, we get a prediction from the current network
    predicted_y = model(train_X)

    # Compute the loss to see how bad or good we are doing
    loss = loss_fn(predicted_y, train_y)

    # Append the loss to keep track of our performance
    loss_list[epoch] = loss.item()

    # Afterwards, we will need to zero the gradients to reset
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate the accuracy, call torch.no_grad() to prevent updating gradients
    # while calculating accuracy
    with torch.no_grad():
        y_pred = model(test_X)
        correct = (torch.argmax(y_pred, dim=1) == test_y).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()


plt.style.use("ggplot")


fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list)
ax1.set_ylabel("Accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("Loss")
ax2.set_xlabel("epochs")
fig.savefig("docs/img_plot_metrics.png")

# Specify all the files we need
zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)


model_path = os.path.join(zkp_dir, "network.onnx")
compiled_model_path = os.path.join(zkp_dir, "network.compiled")
pk_path = os.path.join(zkp_dir, "test.pk")
vk_path = os.path.join(zkp_dir, "test.vk")
settings_path = os.path.join(zkp_dir, "settings.json")
srs_path = os.path.join(zkp_dir, "kzg.srs")
witness_path = os.path.join(zkp_dir, "witness.json")
data_path = os.path.join(zkp_dir, "input.json")
proof_path = os.path.join(zkp_dir, "test.pf")


# After training, export to onnx (network.onnx) and create a data file (input.json)

# create a random input
# x = 0.1*torch.rand(*[1, 10], requires_grad=True)
# x = test_X[0, None]
x = (test_X[0, None] - train_X.mean(dim=0)) / train_X.std(dim=0)

# Flips the neural net into inference mode
model.eval()

# Export the model
torch.onnx.export(
    model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    model_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data=[data_array])

# Serialize data into file:
json.dump(data, open(data_path, "w"))

run_args = ezkl.PyRunArgs()
# run_args.input_visibility = "encrypted"
# run_args.param_visibility = "encrypted"
run_args.input_visibility = "private"
run_args.param_visibility = "private"
run_args.output_visibility = "public"
res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
assert res == True


res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
assert res == True

res = ezkl.get_srs(srs_path, settings_path)


res = ezkl.setup(
    compiled_model_path,
    vk_path,
    pk_path,
    srs_path,
    settings_path,
)

assert res == True
assert os.path.isfile(vk_path)
assert os.path.isfile(pk_path)
assert os.path.isfile(settings_path)  # Generate the Witness for the proof

# now generate the witness file


res = ezkl.gen_witness(
    data_path, compiled_model_path, witness_path, settings_path=settings_path
)
assert os.path.isfile(witness_path)  # Generate the proof

# proof_path = os.path.join('proof.json')

proof = ezkl.prove(
    witness_path,
    compiled_model_path,
    pk_path,
    proof_path,
    srs_path,
    "evm",
    "single",
    settings_path,
)

print(proof)
assert os.path.isfile(proof_path)


# verify our proof

res = ezkl.verify(
    proof_path,
    settings_path,
    vk_path,
    srs_path,
)

assert res == True
print("verified")
