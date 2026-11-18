from train import make_dataset, train_model, FEATURES
def test_dataset_has_required_columns():
 df = make_dataset()
 required_columns = [
 "study_hours",
 "attendance",
 "previous_marks",
 "passed"
 ]
 assert set(required_columns).issubset(df.columns)
def test_target_has_both_classes():
 df = make_dataset()
 assert set(df["passed"].unique()) == {0, 1}
def test_model_has_predict_method():
 model = train_model()
 assert hasattr(model, "predict")
def test_predictions_are_valid():
 model = train_model()
 df = make_dataset()
 predictions = model.predict(df[FEATURES].head(3))
 assert len(predictions) == 3
 assert set(predictions).issubset({0, 1})
