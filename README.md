# Customer Churn Prediction with an ANN

A TensorFlow/Keras binary classifier that estimates whether a bank customer is likely to leave. A Streamlit interface collects customer attributes and displays a churn probability.

## Model and preprocessing

The training notebook drops identifier columns, label-encodes gender, one-hot-encodes geography, and standardizes the remaining features. It creates an 80/20 train/test split and trains a dense network with 64-unit and 32-unit ReLU hidden layers and a sigmoid output.

Training uses Adam, binary cross-entropy, early stopping, and TensorBoard. The target column is `Exited`. The application labels probabilities greater than 0.5 as likely churn.

## Run the application

Use Python compatible with the TensorFlow version in `requirements.txt`:

```bash
git clone https://github.com/Sanskar1314/Customer_churn_prediction_using_ANN.git
cd Customer_churn_prediction_using_ANN
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows activate with `.venv\Scripts\activate`. Open the local Streamlit address shown in the terminal. Run from the repository root because the app loads model and preprocessing files using relative paths.

The form collects geography, gender, age, balance, credit score, estimated salary, tenure, product count, credit-card status, and active-member status.

## Files

| File | Purpose |
| --- | --- |
| `Churn_Modelling.csv` | Included training dataset |
| `experiments.ipynb` | Preprocessing, ANN training, and artifact export |
| `prediction.ipynb` | Example inference on one customer |
| `app.py` | Streamlit interface |
| `model.h5` | Saved network |
| `label_encoder_gender.pkl` | Fitted gender encoder |
| `onehot_encoder_geo.pkl` | Fitted geography encoder |
| `scaler.pkl` | Fitted standard scaler |

The trained model and three preprocessing files are included, so retraining is optional.

## Retrain

```bash
python -m pip install jupyterlab
jupyter lab
```

Open `experiments.ipynb` and run it in order. It overwrites the model, encoders, and scaler in the working directory. Keep all four artifacts from the same training run together. View training logs using the notebook's TensorBoard cells or `tensorboard --logdir logs/fit`.

The notebook uses the held-out split as validation data for early stopping. For a separate final performance estimate, reserve another untouched test set. Churn scores describe this trained model's output; no fixed accuracy is promised.
