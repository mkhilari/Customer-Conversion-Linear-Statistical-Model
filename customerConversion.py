
import pandas 
import numpy 
import matplotlib 

import json

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
    for customerAttribute in ["customerGender", "customerCountry", "customerAge"]: 

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



def main(): 

    (clients, customers, messages) = getRawData() 

    getNullValues({"clients" : clients, 
    "customers" : customers, "messages" : messages}) 

    customerGroups = getCustomerGroups(customers) 

    customersJoinedToClients = getCustomersJoinedToClients(customers, clients) 

    messagesJoinedToCustomers = getMessagesJoinedToCustomers(messages, customersJoinedToClients) 

    messagesWithSentDayTime = getMessagesWithSentDayTime(messagesJoinedToCustomers)

    messageGroups = getMessageGroups(messagesWithSentDayTime)

    messageProbabilities = getMessageProbabilities(messagesWithSentDayTime, 
    ["messageSentAtDayOfWeek", "messageSentAtHour", "messageTimeWindow"])

main() 