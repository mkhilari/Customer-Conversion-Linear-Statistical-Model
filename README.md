# Customer Conversion Linear Statistical Model 

Specification by InDebted 

Solution by Manish Khilari 

Date: 7 October 2022 

--- 

## Introduction 

Welcome to the Customer Conversion Recommender Model. 

Each message to a customer has a probability of clickthrough, and a probability of conversion. 

We want to find the day of week and time of day that maximises the probability of conversion for each customer. 

Our approach is to use a recommender model, where the users are customers, and the recommended items are messages sent at certain days of the week, and certain times of day. 

Solution details and setup instructions can be found below. 

--- 

## Model 

Our model is a user based collaborative filtering recommender model. 

This takes a dataset of (customerId, messageTimeWindow, messageRating) interactions. 

It returns a predicted rating for a (customerId, messageTimeWindow) pair based on the ratings given by similar users. 

Each user is converted to a feature vector [rating1, rating2, ...] containing ratings for each messageTimeWindow. 

In our model, the similarity between users is based on the angle between the two vectors (cosine similarity). The most similar users have a low angle between them. 

To give recommendations to a customer, we predict the rating that they would give to each messageTimeWindow, and sort these time windows by predicted rating descending. The highest rated time windows are then recommended. 

--- 

## Python Solution 

A Python solution can be found at the following location. 

> ./customerConversion.py 

## Setup Instructions 

Setup instructions can be found below. 

1. Install the latest version of Python. Installation instructions can be found at [Welcome to Python](https://www.python.org/). 

2. Ensure the the following packages are installed. 

- > pandas 
- > numpy 
- > matplotlib 
- > json 
- > datetime 
- > surprise 

    Packages can be installed with `pip install <packageName>`. 

3. Ensure that the following input files are present in the data/raw folder. These produce the datasets that are input into the model. 

- > clients.csv 
- > customers.csv 
- > messages.csv 
- > sample_customers.csv 

3. Run the following command. 

- > `python customerConversion.py`

4. Confirm that the following output files have been generated in the output folder. 

    - **bestMessageTimeWindows.json**

        The top 3 recommended message send times for each of the given sample customers based on the collaborative filtering model. 

        The recommended datetimes are provided in the original datetime format, along with the day of the week. 

        Each of the datetimes is within the 4 to 10 October 2021 time period. 

        Each of the datetimes is also within the compliant times of the customer country, or within the compliant times of the client country in the case that the customer country is not present. 

        Recommendations initially outside of compliant times have been moved to the closest compliant hour of the day. 
    
    - **customerGroups.json**

        Customers grouped by distinct values of each attribute. These include groups for customerGender, customerCountry, and customerAge. 

        The number of customers in each group is shown. 
    
    - **meanRatings.json**

        The mean rating given by each customer to each of the message time windows they have been sent. 

        Each message time window is a day of week and hour of day. 

        The group size is the number of times the customer has been sent a message in the given time window. 
    
    - **messageGroups.json**

        Messages grouped by distinct values of each attribute. These include groups for messageSentAtYear, messageSentAtMonth, and messageSentAtDay. 

        There is also the number of clicked and converted messages under messageClicked and messageConverted. 
    
    - **messageProbabilities.json**

        The probability of message conversion given a certain day of week, hour, or weekly time window. 

        Each group has a group size, a probability of message click, a probability of message conversion, and a probability of message conversion given that it has already been clicked. 

    - **nullValues.json**

        The null values present in each field of each dataset. 

        Each field has a number of null and non null values, as well as the probability of encountering a null value. 
    
