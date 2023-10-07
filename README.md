# Customer Conversion Linear Statistical Model 

Specification by InDebted 

Solution by Manish Khilari 

Date: 7 October 2022 

--- 

## Introduction 

Welcome to the Customer Conversion Linear Statistical Model. 

Each message to a customer has a probability of clickthrough, and a probability of conversion. 

We want to find the day of week and time of day that maximises the probability of conversion. 

Our approach is to use a recommender model, where the users are customers, and the recommended items are messages sent at certain times of day. 

### Inputs 

The inputs for each message are the day of week, time of day, as well as the attributes of the customer and their corresponding client. 



Solution details and setup instructions can be found below. 

## Python Solution 

A Python solution can be found at the following location. 

> customerConversion.py 

## Setup Instructions 

Setup instructions can be found below. 

1. Install the latest version of Python. Installation instructions can be found at [Welcome to Python](https://www.python.org/). 

2. Ensure that the following input files are present. These produce the cleansed datasets that are input into the model. 

- > clients.csv 
- > customers.csv 
- > messages.csv 
- > sample_customers.csv 

3. Run the following command. 

- > python customerConversion.py 

4. Confirm that the following output files have been generated. 

- > output.csv 
