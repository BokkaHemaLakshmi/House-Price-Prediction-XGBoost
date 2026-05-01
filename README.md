# House-Price-Prediction-XGBoost
Predict property prices using this Machine Learning project. Based on the Ames Housing Dataset (1,460 rows, 81 features), it employs XGBoost and Naive Bayes to analyze trends. Key features include data cleaning, numerical mapping for categorical variables, and predictive modeling for real estate valuations.
📂 Dataset Documentation: House Price Trend Analysis1. OverviewThis project utilizes the Ames Housing Dataset, a comprehensive collection of residential property data from Ames, Iowa. It serves as a high-dimensional foundation for applying Machine Learning algorithms to predict property valuations. 
2. Dataset SpecificationsAttributeDetailsTotal Observations1,460 rowsTotal Features81 columns (including the target)Target VariableSalePrice (Continuous)Variable Split38 Numerical, 43 Categorical
3. Key Feature CategoriesThe dataset provides a granular view of property characteristics:Physical Space: Includes GrLivArea (above-grade living area), TotalBsmtSF (basement size), and LotArea.Quality Metrics: Ranked features such as OverallQual and OverallCond (1-10 scale).Historical Data: Construction and renovation timelines via YearBuilt and YearRemodAdd.
Amenities: Detailed counts of FullBath, BedroomAbvGr, and GarageCars.
4. Data Preprocessing & MethodologyAs part of the technical implementation in this repository:
Handling Nulls: Significant missing data in features like PoolQC and MiscFeature are addressed through cleaning protocols.
Feature Encoding: Categorical variables (e.g., mainroad, location) are mapped to numerical values (0, 1, 2) to ensure compatibility with gradient boosting frameworks.
Primary Model: XGBoost is utilized for its efficiency in handling tabular data and predictive accuracy. 
5. Statistical InsightsMean
Sale Price: ~$180,921.Average
Living Area: ~1,515 sq. ft..
Standard Dwelling: Typically 3 bedrooms with a construction/remodel median around 1971.


Check here out for app
streamlit app : https://house-price-prediction-xgboost-9kfg395npv5a3h3b254tge.streamlit.app/
