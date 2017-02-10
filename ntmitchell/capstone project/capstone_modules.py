
# coding: utf-8

import pandas


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
        
            
        
    def get_countries_in_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
    def get_production_data(self, ICO_category = "Brazilian Naturals"):
        return self.production_by_ICO_category[ICO_category]
    
    def get_aggregate_production_data(self, ICO_category = "Brazilian Naturals"):
        return self.production_by_ICO_category[ICO_category].groupby(by = "Market_Year")["Value (60kg bags)"].sum()
    
    def get_ending_stocks(self, ICO_category = "Brazilian Naturals"):
        return self.ending_stock_by_ICO_category[ICO_category]
    