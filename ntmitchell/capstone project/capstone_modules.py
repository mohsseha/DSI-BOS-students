
# coding: utf-8

import pandas

# ------------------------- Production Data -------------------------

class Production_Data(object):
    production_data = None
    ICO_categories = ["Brazilian Naturals", "Colombian Milds", "Other Milds", "Robustas"]
    countries_in_ICO_category = dict.fromkeys(ICO_categories, None)
    production_by_ICO_category = dict.fromkeys(ICO_categories, None)
    ending_stock_by_ICO_category = dict.fromkeys(ICO_categories, None)
    
    def __init__(self):
        self.production_data = pandas.read_csv("../datasets/capstone/coffee-production--USDA-FAS--psd_coffee.csv")
        self.production_data = self.production_data[["Country_Name", "Market_Year", "Attribute_Description", "Value"]]
        
        self.production_data.loc[:, "Value (60kg bags)"] = self.production_data.loc[:, "Value"] * 1000
        self.production_data.drop("Value", axis = 1, inplace = True)
        
        ICO_country_classifications = pandas.read_csv("../datasets/capstone/ICO composite indicator index country classification.csv")
        ICO_country_classifications.columns = ["Country", "Brazilian Naturals","Colombian Milds","Other Milds", "Robustas"]
        
        for category in self.ICO_categories:
            temp_dataframe = ICO_country_classifications[ICO_country_classifications[category]][["Country"]]
            temp_dataframe = temp_dataframe.merge(self.production_data, left_on = "Country", right_on = "Country_Name").drop("Country_Name", axis = 1)
            
            self.countries_in_ICO_category[category] = temp_dataframe["Country"].unique().tolist()
            self.ending_stock_by_ICO_category[category] = temp_dataframe[temp_dataframe["Attribute_Description"] == "Ending Stocks"].drop("Attribute_Description", axis = 1)
            if category == "Robustas":
                self.production_by_ICO_category[category] = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"].drop("Attribute_Description", axis = 1)
            else:
                self.production_by_ICO_category[category] = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"].drop("Attribute_Description", axis = 1)
        
    def find_ICO_category_of_country(self, country_name = None):
        categories = list()
        for ICO_category in self.countries_in_ICO_category.keys():
            if country_name in self.countries_in_ICO_category[ICO_category]:
                categories.append(ICO_category)
        return categories    
        
    def get_countries_in_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
    
    def get_production_data_for_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.production_by_ICO_category[ICO_category]
    
    def get_aggregate_production_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.production_by_ICO_category[ICO_category].groupby(by = "Market_Year")["Value (60kg bags)"].sum()
    
    def get_ending_stocks_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.ending_stock_by_ICO_category[ICO_category]
    
# ------------------------- Weather Data -------------------------
    
class Temperature_Data(object):
    temperature_data = pandas.DataFrame()
    ICO_categories = ["Brazilian Naturals", "Colombian Milds", "Other Milds", "Robustas"]
    countries_in_ICO_category = dict.fromkeys(ICO_categories, None)
    
    def __init__(self):
        # Import ICO classifications
        ICO_country_classifications = pandas.read_csv("../datasets/capstone/ICO composite indicator index country classification.csv")
        ICO_country_classifications.columns = ["Country", "Brazilian Naturals","Colombian Milds","Other Milds", "Robustas"]

        for category in self.ICO_categories:
            temp_dataframe = ICO_country_classifications[ICO_country_classifications[category]][["Country"]]
            self.countries_in_ICO_category[category] = temp_dataframe["Country"].unique().tolist()
        
        # Import temperature data
        self.temperature_data = pandas.read_csv("../datasets/capstone/temperature-in-coffee-growing-regions--from-berkeley-earth.csv")

        # Format temperature data
        self.temperature_data.sort_values(by = "Date")
        self.temperature_data["Date"] = pandas.to_datetime(self.temperature_data["Date"])
        self.temperature_data.set_index("Date", inplace = True)
        del self.temperature_data.index.name

    def find_ICO_category_of_country(self, country_name = None):
        categories = list()
        for ICO_category in self.countries_in_ICO_category.keys():
            if country_name in self.countries_in_ICO_category[ICO_category]:
                categories.append(ICO_category)
        return categories    
        
    def get_countries_in_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
        
    def temperature_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        results_dataframe = self.temperature_data[self.temperature_data["Country"].isin(self.countries_in_ICO_category[ICO_category])]
        if ICO_category != "Robustas":
            return results_dataframe[results_dataframe["Arabica Production"] == True]
        else:
            return results_dataframe[results_dataframe["Robusta Production"] == True]
    
    
# ------------------------- ICO Composite Indicator Data -------------------------

class ICO_Composite_Indicator_Index(object):
    ICO_indicator_data = pandas.DataFrame()
    ICO_categories = ["Brazilian Naturals", "Colombian Milds", "Other Milds", "Robustas"]
    countries_in_ICO_category = dict.fromkeys(ICO_categories, None)
    
    def __init__(self):
        # Import ICO classifications
        ICO_country_classifications = pandas.read_csv("../datasets/capstone/ICO composite indicator index country classification.csv")
        ICO_country_classifications.columns = ["Country", "Brazilian Naturals","Colombian Milds","Other Milds", "Robustas"]

        for category in self.ICO_categories:
            temp_dataframe = ICO_country_classifications[ICO_country_classifications[category]][["Country"]]
            self.countries_in_ICO_category[category] = temp_dataframe["Country"].unique().tolist()
        
        # Import temperature data
        self.ICO_indicator_data = pandas.read_csv("../datasets/capstone/ICO composite indicator prices 1990-2016.csv")

        # Format temperature data
        self.ICO_indicator_data["Unnamed: 0"] = pandas.to_datetime(self.ICO_indicator_data["Unnamed: 0"])
        self.ICO_indicator_data.set_index("Unnamed: 0", inplace = True)
        del self.ICO_indicator_data.index.name

        
    def find_ICO_category_of_country(self, country_name = None):
        categories = list()
        for ICO_category in self.countries_in_ICO_category.keys():
            if country_name in self.countries_in_ICO_category[ICO_category]:
                categories.append(ICO_category)
        return categories    
        
    def get_countries_in_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
    
    def ICO_indicator_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.ICO_indicator_data[ICO_category]
       
    def ICO_composite_indicator_data(self, ICO_category = "Brazilian Naturals"):
        return self.ICO_indicator_data[ICO_category]