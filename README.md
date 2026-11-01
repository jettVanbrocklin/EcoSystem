# EcoSystem

Plant Health Classification
A Python machine learning project that classifies plant health using environmental and soil measurements. The goal is to turn plant monitoring data into an understandable health category that can help identify potential stress.
Project Scope
This repository focuses on data preparation, model training, and evaluation. Sensor hardware, automated watering, and a user interface are outside the current scope.

The project is in the planning stage. The dataset, final input features, health labels, and classification algorithm have not yet been selected.
Potential Input Features
The final feature set will depend on the available dataset and measurement quality.
Feature	Description
Soil moisture	Water content in the growing medium
Light exposure	Light intensity and, when available, duration of exposure
Air temperature	Temperature surrounding the plant
Relative humidity	Moisture in the surrounding air
Soil temperature	Temperature in the root zone
Soil pH	Acidity or alkalinity of the growing medium
Soil electrical conductivity (EC)	Indicator of dissolved salt levels, not a direct measurement of individual nutrients


Plant species and recent measurement history may also be included when available, since suitable conditions vary across plants and over time.
Classification Target
The model will assign a health category to each input sample. Possible categories include healthy, stressed, and unhealthy, but the final classes must match clearly defined labels in the dataset.

Health labels should ideally come from independent plant observations or assessments. Labels generated only from sensor thresholds would train the model to reproduce those thresholds rather than independently assess plant health.
Planned Workflow
1. Select or collect a labeled dataset and document measurement units and label definitions.
2. Explore feature distributions, missing values, and class balance.
3. Split data into training, validation, and test sets before fitting preprocessing steps. Keep records from the same plant together, or use a time-based split when evaluating future predictions.
4. Establish a simple baseline and compare candidate classification models.
5. Evaluate predictions using accuracy, per-class precision and recall, macro F1, and a confusion matrix.
6. Save the selected model together with its preprocessing steps for inference on new measurements.
Planned Python Tools
- pandas and NumPy for data preparation and numerical operations
- scikit-learn for preprocessing, classification, and evaluation
- Matplotlib and Seaborn for data exploration and result visualization
- Jupyter for exploratory notebooks
Dependencies and installation instructions will be added when the implementation is established.
Limitations
Environmental measurements describe growing conditions and do not capture every aspect of plant health. Pests, disease, root damage, and other issues may not be detectable from these features alone. Predictions will depend on label quality and how well the training data represents the plants and conditions being evaluated.
Next Steps
- Choose a dataset and define the classification labels.
- Confirm the available features and units.
- Build a reproducible preprocessing and training pipeline.
- Compare models and document held-out evaluation results.
