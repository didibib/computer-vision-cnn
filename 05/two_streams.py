from tensorflow import keras

st40_frames = keras.models.load_model('trained_models/st40_frames_10')
tvhi_of = keras.models.load_model('trained_models/tvhi_of_40')

# remove last 4 layers of st40 model
st40_frames.layers = st40_frames.layers[:-4]
# remove last 4 layers of tvhi optical flow model
tvhi_of.layers = tvhi_of.layers[:-4]

