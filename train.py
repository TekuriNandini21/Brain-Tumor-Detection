from data_loader import load_data
from model import unet

print("Loading Dataset...")

X, Y = load_data("archive/kaggle_3m")

print("Images Shape:", X.shape)
print("Masks Shape :", Y.shape)

print("Image Min:", X.min())
print("Image Max:", X.max())

print("Mask Min :", Y.min())
print("Mask Max :", Y.max())
print("Mask Sum :", Y.sum())

X = X.reshape(-1,128,128,1)
Y = Y.reshape(-1,128,128,1)

model = unet()

print("Training Started...")

history = model.fit(
    X,
    Y,
    epochs=30,
    batch_size=8,
    validation_split=0.2
)

model.save("tumor_model.h5")

print("Model Saved Successfully!")