# TASK 2

There is currenly a basic implementation
- The missing value are filled with the avrege of the others, see plots for more details
- A try to fit the data using a kernel, but not working yet

# Missing chf Values
The missing values can be extrated with
```python
missing_chf = df["price_CHF"].isnull()
```

 A few diffrents methods has been tried:
- Fill the missing CHF with random values in the range
```python
min_price, max_price = df["price_CHF"].min(), df["price_CHF"].max()
random_chf_prices = np.random.uniform(min_price, max_price, size=len(df[missing_chf]))
df.loc[missing_chf, "price_CHF"] = random_chf_prices
```
This was creating a lot of noise in the plot since

- Replace with median:
```python
missing_chf = df["price_CHF"].isnull()
mean_price = df["price_CHF"].mean()
df.loc[missing_chf, "price_CHF"] = mean_price
```
This was creating a straight line in the plot

-  loop over missing values and impute using random noise 
```python
non_missing_chf = df[~missing_chf]["price_CHF"]
missing_chf_values = df[missing_chf]["price_CHF"]
random_val = non_missing_chf.sample(len(missing_chf_values), replace=True)
df.loc[missing_chf, "price_CHF"] = random_val.values + np.random.uniform(0, 0, size=len(missing_chf_values))
```

This was setting correctly some of the points but still a lot of noise was created

-> Current solution it's to drop the value where it's missing
It also make sense since we are trying to predict extactly that price

![Noise Example](/plots/noise.png)
In the image, the green line was the approximation using a linear regression model, and the black dots it's the noise with the missing CHF values

# Plots
In the plot directory, all the plot have been done with the price of a country compared to CHF.

The Orage dots are the values of the training data.

The Red dots are optiained by creating a `rbf` model from the training data, then a random in the range +/- 0.4 has been added to the predicted values of of the CHF from the model.

the function
```python
def fill_svr(df: pd.DataFrame, gamma=0.1, C=1, noise=0):
```

takes as parametes gamma, C and the noise. To optiain diffrent results they can be changes.


![Noise Example](/plots/price_GER_price_CHF.png)
