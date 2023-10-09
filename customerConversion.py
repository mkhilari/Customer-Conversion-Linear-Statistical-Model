
from numpy.random import sample
import pandas 
import numpy 
import matplotlib 

import json
import datetime

import surprise

def getRawData(): 
    """
    Returns the raw clients, customers, and messages as dataframes 
    """

    clients = pandas.read_csv("./data/raw/clients.csv")
    customers = pandas.read_csv("./data/raw/customers.csv")
    messages = pandas.read_csv("./data/raw/messages.csv")

    # Rename columns 
    clients = clients.rename(
        columns = {"id" : "clientId", "name" : "clientName", 
        "product_type" : "clientProductType", 
        "country" : "clientCountry"})

    customers = customers.rename(
        columns = {"id" : "customerId", "gender" : "customerGender", 
        "client_id" : "clientId", "country" : "customerCountry", 
        "age" : "customerAge", "created_at" : "customerCreatedAt"})

    messages = messages.rename(columns = {"id" : "messageId", "customer_id" : "customerId", 
    "sent_at" : "messageSentAt", "clicked" : "messageClicked", 
    "converted" : "messageConverted"})

    # Output summary stats 
    clients.describe().to_csv("./output/rawClientsSummary.csv")
    customers.describe().to_csv("./output/rawCustomersSummary.csv")
    messages.describe().to_csv("./output/rawMessagesSummary.csv")

    return (clients, customers, messages) 

def getNullValues(dataframes): 
    """
    Returns the number of null values in each field 
    of each of the given dataframes
    """

    # Get null values by dataframe name then by field name 
    nullValues = {}

    for dataframeName in dataframes:

        dataframe = dataframes[dataframeName]

        nullValuesInDataframe = {}

        for fieldName in dataframe:

            nullValuesInField = {}

            size = dataframe.shape[0]
            nullValueCount = dataframe[fieldName].isnull().sum()
            nonNullValueCount = dataframe[fieldName].notnull().sum()

            # Get the probability of a null value in the given field 
            probabilityOfNull = (nullValueCount / size)

            nullValuesInField["size"] = size
            nullValuesInField["nullValues"] = nullValueCount
            nullValuesInField["nonNullValues"] = nonNullValueCount

            nullValuesInField["probabilityOfNull"] = probabilityOfNull

            nullValuesInDataframe[fieldName] = nullValuesInField
        
        nullValues[dataframeName] = nullValuesInDataframe
    
    with open("./output/nullValues.json", "w") as nullValuesJson:

        json.dump(nullValues, nullValuesJson, default = int64Encoder, indent = 4)
    
    return nullValues

def getCustomerGroups(customers): 

    customerGroups = {}

    # Get the customer groups represented by distinct values for each attribute 
    for customerAttribute in ["customerGender", "customerCountry", 
    "customerAge", "clientProductType", "clientCountry"]: 

        group = customers.groupby(customerAttribute).size()

        # Get the count of each group of distinct attribute values 
        customerGroups[customerAttribute] = group.to_dict()

    with open("./output/customerGroups.json", "w") as customerGroupsJson: 

        json.dump(customerGroups, customerGroupsJson, indent = 4) 

    return customerGroups

def int64Encoder(object): 
    """
    Returns the given numpy int64 instance as an integer for JSON supported output
    """

    if isinstance(object, numpy.int64):

        return int(object)
    else:
        raise TypeError(f"The given object {object} is not supported by JSON")

def getMessageGroups(messages): 

    messageGroups = {}

    # Get the message groups represented by distinct values for each attribute 
    for messageAttribute in ["messageSentAtYear", "messageSentAtMonth", 
    "messageSentAtDay", "messageSentAtHour", "messageClicked", "messageConverted"]: 

        group = messages.groupby(messageAttribute).size()

        # Get the count of each group of distinct attribute values 
        messageGroups[messageAttribute] = group.to_dict()

    with open("./output/messageGroups.json", "w") as messageGroupsJson: 

        json.dump(messageGroups, messageGroupsJson, indent = 4) 

    return messageGroups

def getCustomersJoinedToClients(customers, clients): 
    """
    Returns the customers left joined to clients
    """

    customersJoinedToClients = pandas.merge(customers, clients, on = "clientId", how = "left") 
    return customersJoinedToClients

def getMessagesJoinedToCustomers(messages, customersJoinedToClients): 
    """
    Returns the messages left joined to customers with their corresponding clients 
    """

    messagesJoinedToCustomers = pandas.merge(messages, customersJoinedToClients, on = "customerId", how = "left") 
    return messagesJoinedToCustomers

def getMessagesWithSentDayTime(messages): 
    """
    Returns the messages with messageSentAt converted to a day of week and hour of day 
    """

    # Convert the messageSentAt string to a datetime 
    messages["messageSentAtDateTime"] = pandas.to_datetime(messages["messageSentAt"], utc = True)

    # Get the datetime parts 
    messages['messageSentAtYear'] = messages['messageSentAtDateTime'].dt.year
    messages['messageSentAtMonth'] = messages['messageSentAtDateTime'].dt.month
    messages['messageSentAtDay'] = messages['messageSentAtDateTime'].dt.day
    messages['messageSentAtHour'] = messages['messageSentAtDateTime'].dt.hour
    messages['messageSentAtMinute'] = messages['messageSentAtDateTime'].dt.minute
    messages['messageSentAtSecond'] = messages['messageSentAtDateTime'].dt.second
    messages['messageSentAtMicrosecond'] = messages['messageSentAtDateTime'].dt.microsecond

    # Get the day of week given the date 
    messages['messageSentAtDayOfWeek'] = messages['messageSentAtDateTime'].dt.day_name()

    # Convert the customerCreatedAt string to a datetime 
    messages["customerCreatedAtDateTime"] = pandas.to_datetime(messages["customerCreatedAt"], utc = True)

    # Get the days between the customer creation date and the message date 
    messages["customerCreatedToMessageSentDays"] = (messages["messageSentAtDateTime"] - messages["customerCreatedAtDateTime"]).dt.days

    # Get the message time window based on the day of the week and the hour of the day 
    # Messages in the same time window will be sent at the same day of week and hour of day 
    messages["messageTimeWindow"] = messages["messageSentAtDayOfWeek"].astype(str) + "_" + messages["messageSentAtHour"].astype(str).str.zfill(2)

    # Output summary stats 
    messages.describe().to_csv("./output/messagesWithSentDateTimeSummary.csv")
    
    return messages

def getMessagesWithRatings(messages, convertedRating): 
    """
    Returns the messages with ratings based on clicks and conversions 
    """

    # Set the rating of clicked only messages to 1, 
    # and the rating of converted messages to the given converted rating 
    messages["messageRating"] = messages["messageClicked"] + (convertedRating - 1) * messages["messageConverted"]

    return messages

def getMessageProbabilities(messages, groupFieldNames):
    """
    Returns the probability of click and probability of conversion 
    of messages grouped by each of the given fields
    """

    # Get probabilities of clicks and conversions by field value 
    messageProbabilities = {}

    for groupFieldName in groupFieldNames:

        messageFieldProbabilities = {}

        messageGroups = messages.groupby(groupFieldName)

        # Group messages by the value of the given field 
        for fieldValue, messageGroup in messageGroups:

            messageFieldValueProbabilities = {}
            
            groupSize = messageGroup.shape[0]

            numberOfClicks = messageGroup["messageClicked"].sum()
            numberOfConversions = messageGroup["messageConverted"].sum()

            probabilityOfClick = (numberOfClicks / groupSize)
            probabilityOfConversion = (numberOfConversions / groupSize)

            probabilityOfConversionGivenClicked = (probabilityOfConversion / probabilityOfClick)

            messageFieldValueProbabilities["groupSize"] = groupSize

            messageFieldValueProbabilities["numberOfClicks"] = numberOfClicks
            messageFieldValueProbabilities["numberOfConversions"] = numberOfConversions

            messageFieldValueProbabilities["probabilityOfClick"] = probabilityOfClick
            messageFieldValueProbabilities["probabilityOfConversion"] = probabilityOfConversion

            messageFieldValueProbabilities["probabilityOfConversionGivenClicked"] = probabilityOfConversionGivenClicked

            messageFieldProbabilities[fieldValue] = messageFieldValueProbabilities

        messageProbabilities[groupFieldName] = messageFieldProbabilities
    
    with open("./output/messageProbabilities.json", "w") as messageProbabilitiesJson:

        json.dump(messageProbabilities, messageProbabilitiesJson, 
        default = int64Encoder, indent = 4, sort_keys = True)
    
    return messageProbabilities

def getSampleCustomers(customers): 
    """
    Returns the sample customers joined to the customers dataset 
    """

    sampleCustomerIds = pandas.read_csv("./data/raw/sample_customer_ids.csv")

    # Rename columns 
    sampleCustomerIds = sampleCustomerIds.rename(
        columns = {"customer_id" : "customerId"})

    sampleCustomers = pandas.merge(sampleCustomerIds, customers, on = "customerId", how = "left")

    # Set customerId as the index for the dataframe 
    sampleCustomers.set_index("customerId", inplace = True)

    return sampleCustomers

def getCustomerMeanRatingsForMessageTimeWindows(messages):
    """
    Returns the mean rating given by each user to each message time window
    """

    # Group messages by customer ID then by message time window 
    customerIdMessageGroups = messages.groupby("customerId")

    meanRatings = {}

    for customerId, customerIdMessageGroup in customerIdMessageGroups:

        customerIdTimeWindowMessageGroups = customerIdMessageGroup.groupby("messageTimeWindow")

        customerIdMeanRatings = {}

        for messageTimeWindow, customerIdTimeWindowMessageGroup in customerIdTimeWindowMessageGroups:

            customerIdTimeWindowMeanRatings = {}

            groupSize = customerIdTimeWindowMessageGroup.shape[0]
            meanRating = customerIdTimeWindowMessageGroup["messageRating"].mean()
            
            customerIdTimeWindowMeanRatings["groupSize"] = groupSize
            customerIdTimeWindowMeanRatings["meanRating"] = meanRating

            customerIdMeanRatings[messageTimeWindow] = customerIdTimeWindowMeanRatings
        
        meanRatings[customerId] = customerIdMeanRatings

    with open("./output/meanRatings.json", "w") as meanRatingsJson:

        json.dump(meanRatings, meanRatingsJson, 
        default = int64Encoder, indent = 4, sort_keys = True)
    
    return meanRatings 

def getDateTime(sampleCustomers, customerId, customerTimeWindow):

    """
    Returns the date time representation of the given time window, 
    considering the compliant times of the customer country
    """

    # Convert the predicted hours to compliant hours by country 
    compliantHours = {

        # Canada 9am to 5pm 
        "CA" : {
            "00" : "09",
            "01" : "09",
            "02" : "09",
            "03" : "09",
            "04" : "09",
            "05" : "09",
            "06" : "09",
            "07" : "09",
            "08" : "09",
            "09" : "09",
            "10" : "10",
            "11" : "11",
            "12" : "12",
            "13" : "13",
            "14" : "14",
            "15" : "15",
            "16" : "16",
            "17" : "17",
            "18" : "17",
            "19" : "17",
            "20" : "17",
            "21" : "17",
            "22" : "17",
            "23" : "17",
            "24" : "17"
        }, 

        # New Zealand 9am to 6pm 
        "NZ" : {
            "00" : "09",
            "01" : "09",
            "02" : "09",
            "03" : "09",
            "04" : "09",
            "05" : "09",
            "06" : "09",
            "07" : "09",
            "08" : "09",
            "09" : "09",
            "10" : "10",
            "11" : "11",
            "12" : "12",
            "13" : "13",
            "14" : "14",
            "15" : "15",
            "16" : "16",
            "17" : "17",
            "18" : "18",
            "19" : "18",
            "20" : "18",
            "21" : "18",
            "22" : "18",
            "23" : "18",
            "24" : "18"
        }, 

        # UK 8am to 8pm 
        "UK" : {
            "00" : "08",
            "01" : "08",
            "02" : "08",
            "03" : "08",
            "04" : "08",
            "05" : "08",
            "06" : "08",
            "07" : "08",
            "08" : "08",
            "09" : "09",
            "10" : "10",
            "11" : "11",
            "12" : "12",
            "13" : "13",
            "14" : "14",
            "15" : "15",
            "16" : "16",
            "17" : "17",
            "18" : "18",
            "19" : "19",
            "20" : "20",
            "21" : "20",
            "22" : "20",
            "23" : "20",
            "24" : "20"
        }
    }

    # Convert the predicted days of the week to days within the date range 4 to 10 October 2021 
    compliantDates = {
        "Monday" : "2021-10-04", 
        "Tuesday" : "2021-10-05", 
        "Wednesday" : "2021-10-06", 
        "Thursday" : "2021-10-07", 
        "Friday" : "2021-10-08"
    }

    customerTimeWindowParts = customerTimeWindow.split("_")

    (customerTimeWindowDayOfWeek, customerTimeWindowHour) = (
        customerTimeWindowParts[0], customerTimeWindowParts[1])

    customerCountry = sampleCustomers.loc[customerId, "customerCountry"]

    # If the customer country is not present, use the client country 
    if pandas.isna(customerCountry):

        customerCountry = sampleCustomers.loc[customerId, "clientCountry"]

    compliantDate = compliantDates[customerTimeWindowDayOfWeek]

    compliantDateParts = compliantDate.split("-")

    # Remove leading zeros and convert to integers 
    (compliantYear, compliantMonth, compliantDay) = (
        int(compliantDateParts[0].lstrip("0")), 
        int(compliantDateParts[1].lstrip("0")), 
        int(compliantDateParts[2].lstrip("0"))
    )
    
    compliantHour = int(compliantHours[customerCountry][customerTimeWindowHour].lstrip("0"))

    compliantDateTime = datetime.datetime(
        year = compliantYear, 
        month = compliantMonth, 
        day = compliantDay, 
        hour = compliantHour, 
        minute = 0, 
        second = 0, 
        microsecond = 0
    )

    # Convert the datetime to a string format 
    compliantDateTimeString = compliantDateTime.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    return compliantDateTimeString

def getCollaborativeFilteringRecommendations(meanRatings, convertedRating, sampleCustomers, recommendationsPerCustomer):
    """
    Returns the collaborative filtering recommendations for the given customer IDs 
    """

    ratingReader = surprise.Reader(rating_scale = (1, convertedRating))

    # Convert the mean ratings to a dataset of (customerId, messageTimeWindow, meanRating)
    meanRatingsData = {"customerId" : [], "messageTimeWindow" : [], "meanRating" : []}

    for customerId in meanRatings:

        customerMeanRatings = meanRatings[customerId]

        for messageTimeWindow in customerMeanRatings:

            customerMessageTimeWindowMeanRatings = customerMeanRatings[messageTimeWindow]

            meanRating = customerMessageTimeWindowMeanRatings["meanRating"]

            meanRatingsData["customerId"].append(customerId)
            meanRatingsData["messageTimeWindow"].append(messageTimeWindow)
            meanRatingsData["meanRating"].append(meanRating)

    meanRatingsDataFrame = pandas.DataFrame(meanRatingsData)

    ratingDataset = surprise.Dataset.load_from_df(meanRatingsDataFrame, ratingReader)

    # Split the rating data into 80% training and 20% testing datasets 
    # Set the random seed for reproducibility 
    (trainingDataset, testingDataset) = surprise.model_selection.train_test_split(ratingDataset, test_size = 0.2, random_state = 7)

    # Select the k nearest neighbours collaborative filtering model 
    recommenderModel = surprise.SVD()

    recommenderModel.fit(trainingDataset)

    messagePredictions = recommenderModel.test(testingDataset)

    # Get the test accuracy 
    rmse = surprise.accuracy.rmse(messagePredictions)
    print(f"RMSE: {rmse}")

    # Get the best message time windows by customer ID 
    bestMessageTimeWindows = {}

    messageTimeWindows = trainingDataset.all_items()

    for customerId in sampleCustomers.index:

        # Get message time windows by rating descending 
        messagePredictions = [(messageTimeWindow, 

        # Get the predicted rating for the (customerId, messageTimeWindow) combination 
        recommenderModel.predict(customerId, trainingDataset.to_raw_iid(messageTimeWindow)).est) 
        for messageTimeWindow in messageTimeWindows]

        messagePredictions.sort(key = lambda messagePrediction : messagePrediction[1], reverse = True)

        # Get up to the given recommendations per customer 
        customerTimeWindows = [trainingDataset.to_raw_iid(messagePredictions[i][0]) 
        for i in range(min(recommendationsPerCustomer, len(messagePredictions)))]

        bestMessageTimeWindows[customerId] = [
            {
                "messageTimeWindow" : customerTimeWindow, 
                "messageDateTime" : getDateTime(sampleCustomers, customerId, customerTimeWindow)
            } 
            for customerTimeWindow in customerTimeWindows]

    with open("./output/bestMessageTimeWindows.json", "w") as bestMessageTimeWindowsJson:

        json.dump(bestMessageTimeWindows, bestMessageTimeWindowsJson, 
        default = int64Encoder, indent = 4, sort_keys = True)
    
    return bestMessageTimeWindows

def main(): 

    (clients, customers, messages) = getRawData() 

    getNullValues({"clients" : clients, 
    "customers" : customers, "messages" : messages}) 

    customersJoinedToClients = getCustomersJoinedToClients(customers, clients) 

    customerGroups = getCustomerGroups(customersJoinedToClients) 

    messagesJoinedToCustomers = getMessagesJoinedToCustomers(messages, customersJoinedToClients) 

    messagesWithSentDayTime = getMessagesWithSentDayTime(messagesJoinedToCustomers)

    messageGroups = getMessageGroups(messagesWithSentDayTime)

    messageProbabilities = getMessageProbabilities(messagesWithSentDayTime, 
    ["messageSentAtDayOfWeek", "messageSentAtHour", "messageTimeWindow"])

    convertedRating = 5

    messagesWithRatings = getMessagesWithRatings(messagesWithSentDayTime, convertedRating)

    sampleCustomers = getSampleCustomers(customersJoinedToClients)

    meanRatings = getCustomerMeanRatingsForMessageTimeWindows(messagesWithRatings)

    bestMessageTimeWindows = getCollaborativeFilteringRecommendations(meanRatings, convertedRating, sampleCustomers, recommendationsPerCustomer = 3)

main() 