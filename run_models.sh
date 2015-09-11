
# commandlines from des.

DATA="../data"

# Let's assume your ultimate goal is you train a German language description
# model.

# Step 1. Train the English model

python train.py --run_string ger-256 --big_batch_size 10000 --batch_size 100 --epochs 50 --dataset $DATA/iaprtc12_eng/ --unk=3 --generation_timesteps=30 --num_layers=1 --hidden_size=256 --fixed_seed --optimiser=adam


# Step 2. Serialise the final hidden vector from the English model into
# iaprtc12_eng/dataset.h5['final_hidden_features']

python extract_hidden_features.py --big_batch_size 10000 --batch_size 100 --dataset $DATA/iaprtc12_eng/ --num_layers=1 --hidden_size=256 --h5_writeable


# Step 3. Train the German model, conditioned on the final English hidden
# vectors

python train.py --run_string ger-256 --big_batch_size 10000 --batch_size 100 --epochs 50 --dataset $DATA/iaprtc12_ger/ --source_vectors $DATA/iaprtc12_eng/ --unk=3 --generation_timesteps=30 --num_layers=1 --hidden_size=256 --fixed_seed --optimiser=adam

