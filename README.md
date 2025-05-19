Restaurant Revenue Prediction
A machine learning project that predicts restaurant revenue based on various features like location, cuisine, average meal price, marketing budget, customer reviews, and more.

📁 Dataset
The dataset contains 8368 restaurant records with 16 features including:
Location
Cuisine
Rating
Seating Capacity
Average Meal Price
Marketing Budget
Social Media Followers
Chef Experience (Years)
Number of Reviews
Avg. Review Length
Ambience Score
Service Quality Score
Parking Availability
Weekend & Weekday Reservations
Revenue (target)

🔧 Preprocessing Steps
Removed irrelevant columns like Name.
Checked for missing values — no missing data found.
Encoded categorical variables (Location, Cuisine, Parking Availability) using LabelEncoder.
Dropped original categorical columns after encoding.
Scaled the target variable Revenue using MinMaxScaler to create revenue_scaled
Dropped the original Revenue column.

🧠 Tools & Libraries Used
pandas for data manipulation
numpy for numerical operations
scikit-learn (LabelEncoder, MinMaxScaler) for preprocessing
