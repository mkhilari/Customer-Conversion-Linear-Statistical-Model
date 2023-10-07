
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

def getDataQuality(clients, customers, messages): 

    size = clients.shape[0]
    numberOfNonNullValues = clients.notnull().sum()
    numberOfNullValues = clients.isnull().sum()

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

    # Output summary stats 
    messages.describe().to_csv("./output/messagesWithSentDateTimeSummary.csv")
    
    return messages

def main(): 

    (clients, customers, messages) = getRawData() 

    getDataQuality(clients, customers, messages) 

    customerGroups = getCustomerGroups(customers) 

    customersJoinedToClients = getCustomersJoinedToClients(customers, clients) 

    messagesJoinedToCustomers = getMessagesJoinedToCustomers(messages, customersJoinedToClients) 

    messagesWithSentDayTime = getMessagesWithSentDayTime(messagesJoinedToCustomers)

    messageGroups = getMessageGroups(messagesWithSentDayTime)

main() 