import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("hoteldetail.csv")

X = df[["Distance_to_stadium", "Star_Rating", "days_before_match"]]
Y = df["hotel_price"]

model = LinearRegression() 
model.fit(X, Y) 
model.coef_, model.intercept_

hotel_price_pred = model.predict(X)  

# print("Koefisien (b1, b2, b3):", model.coef_)
# print("Intercept (a):", model.intercept_)

new_data = pd.DataFrame([[0.5, 5, 1]], columns=X.columns)
prediksi = model.predict(new_data)
print(f"Price Prediction: ${round(prediksi[0])} USD")

#new data for prediction
new_distance   = 0.5
new_star_rating  = 5
new_days_before_match     = 1


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

#Distance vs Price
X_distance= df[["Distance_to_stadium"]]
Y_price = df["hotel_price"]
model_trend_distance = LinearRegression() #creating a model of linear regression so easier to be called
model_trend_distance.fit(X_distance, Y_price) # finding the best fit line for the data
x_trend_d = np.linspace(X_distance.min(), X_distance.max(), 100).reshape(-1, 1) # 100 number points to create a smoother line, reshape => to convert the number to become a column
y_trend_d = model_trend_distance.predict(x_trend_d) # using the trained model to count the predicted price for x distance

ax1.scatter(df["Distance_to_stadium"], df["hotel_price"], color="blue", alpha=0.6) 
ax1.scatter(new_distance, prediksi, color="green", s=100, label="New Data", zorder=5)
ax1.legend() #to give an info
ax1.plot(x_trend_d, y_trend_d, color="green", linewidth=2)
ax1.set_title("Distance to Stadium vs Price")
ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("Price (USD)")
ax1.grid(True)

#Star Rating vs Price
X_StarRating = df[["Star_Rating"]]
model_trend_StarRating = LinearRegression()
model_trend_StarRating.fit(X_StarRating, Y_price)
x_trend_b = np.linspace(X_StarRating.min(), X_StarRating.max(), 100).reshape(-1, 1)
y_trend_b = model_trend_StarRating.predict(x_trend_b)

ax2.scatter(df["Star_Rating"], df["hotel_price"], color="orange", alpha=0.6)
ax2.scatter(new_star_rating, prediksi, color="green", s=100, label="New Data", zorder=5)
ax2.legend()
ax2.plot(x_trend_b, y_trend_b, color="green", linewidth=2)
ax2.set_title("Star Rating vs Price")
ax2.set_xlabel("Star Rating")
ax2.grid(True)

# Days vs Price 
X_daysbeforematch = df[["days_before_match"]]
model_trend_daysbeforematch = LinearRegression()
model_trend_daysbeforematch.fit(X_daysbeforematch, Y_price)
x_trend_h = np.linspace(X_daysbeforematch.min(), X_daysbeforematch.max(), 100).reshape(-1, 1)
y_trend_h = model_trend_daysbeforematch.predict(x_trend_h)

ax3.scatter(df["days_before_match"], df["hotel_price"], color="red", alpha=0.6)
ax3.scatter(new_days_before_match, prediksi, color="green", s=100, label="New Data", zorder=5)
ax3.legend()
ax3.plot(x_trend_h, y_trend_h, color="green", linewidth=2)
ax3.set_title("Days Before Match vs Price")
ax3.set_xlabel("Days Before Match")
ax3.grid(True)

# Graphic Title
fig.suptitle("Analysis of Hotel Price factor (Multiple Linear Regression)", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 1]) 
plt.show()