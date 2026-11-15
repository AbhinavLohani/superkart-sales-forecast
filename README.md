# SuperKart - Automated Sales Forecasting (MLOps)

End-to-end MLOps pipeline that predicts `Product_Store_Sales_Total` for SuperKart products and stores.

```
.
|-- .github/workflows/pipeline.yml        # CI/CD: register -> prep -> train (MLflow) -> commit model -> smoke test
`-- superkart_project/
    |-- data/SuperKart.csv                # registered dataset
    |-- model_building/
    |   |-- data_register.py              # validates columns, prints summary
    |   |-- prep.py                       # clean + split (Xtrain/Xtest/ytrain/ytest)
    |   |-- train.py                      # tune RF / GBM / XGBoost, track with MLflow, save best model
    |   `-- smoke_test.py                 # CI check of the Streamlit app
    |-- deployment/
    |   |-- app.py                        # Streamlit app
    |   |-- requirements.txt              # app dependencies (Streamlit Cloud)
    |   |-- superkart_model.joblib        # model committed by the pipeline
    |   `-- model_metrics.json
    `-- requirements.txt                  # pipeline dependencies
```

**Deploy:** Streamlit Community Cloud -> main file `superkart_project/deployment/app.py`, Python 3.11.
