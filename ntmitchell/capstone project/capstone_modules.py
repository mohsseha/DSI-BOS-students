
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
        self.production_data.columns = ["Country", "Market_Year", "Attribute_Description", "Value"]
        
        self.production_data.loc[:, "Value (60kg bags)"] = self.production_data.loc[:, "Value"] * 1000
        self.production_data.drop("Value", axis = 1, inplace = True)
        
        ICO_country_classifications = pandas.read_csv("../datasets/capstone/ICO composite indicator index country classification.csv")
        ICO_country_classifications.columns = ["Country", "Brazilian Naturals","Colombian Milds","Other Milds", "Robustas"]
        
        for category in self.ICO_categories:
            temp_dataframe = ICO_country_classifications[ICO_country_classifications[category]][["Country"]]
            temp_dataframe = temp_dataframe.merge(self.production_data, on = "Country")
            
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
        
    def list_countries_in_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
    
    def get_production_data_for_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.production_by_ICO_category[ICO_category]
    
    def get_aggregate_production_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.production_by_ICO_category[ICO_category].groupby(by = "Market_Year")["Value (60kg bags)"].sum()
    
    def get_ending_stocks_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        temp_dataframe = self.ending_stock_by_ICO_category[ICO_category]
        for country in temp_dataframe["Country"].unique():
            temp_dataframe.ix[temp_dataframe["Country"] == country, "Percent Change"] = temp_dataframe[temp_dataframe["Country"] == country]["Value (60kg bags)"].pct_change()
        
        return temp_dataframe
    
    def get_production_share_by_country(self, country_or_category = "All"):
        
        total_arabica_production = pandas.pivot_table(data = self.production_data[self.production_data["Attribute_Description"] == "Arabica Production"], 
               values = ["Value (60kg bags)"], 
               index = ["Market_Year"], aggfunc = 'sum')["Value (60kg bags)"].to_dict()
        
        total_robusta_production = pandas.pivot_table(data = self.production_data[self.production_data["Attribute_Description"] == "Robusta Production"], 
               values = ["Value (60kg bags)"], 
               index = ["Market_Year"], aggfunc = 'sum')["Value (60kg bags)"].to_dict()
        
        if country_or_category in self.ICO_categories:
            results = []
            for country in self.countries_in_ICO_category[country_or_category]:
                temp_dataframe = self.production_data[(self.production_data["Country"] == country) 
                                                      & ((self.production_data["Attribute_Description"] == "Arabica Production") 
                                                         | (self.production_data["Attribute_Description"] == "Robusta Production"))]

                scaled_arabica_production = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"]["Value (60kg bags)"] / temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"]["Market_Year"].map(total_arabica_production)
                scaled_arabica_production_dataframe = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"].drop(["Attribute_Description", "Value (60kg bags)"], axis = 1 )
                scaled_arabica_production_dataframe = pandas.concat([scaled_arabica_production_dataframe, scaled_arabica_production], axis = 1)
                scaled_arabica_production_dataframe.columns = ["Country", "Market_Year", "Arabica Production Share"]

                scaled_robusta_production = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"]["Value (60kg bags)"] / temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"]["Market_Year"].map(total_robusta_production)
                scaled_robusta_production_dataframe = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"].drop(["Attribute_Description", "Value (60kg bags)"], axis = 1 )
                scaled_robusta_production_dataframe = pandas.concat([scaled_robusta_production_dataframe, scaled_robusta_production], axis = 1)
                scaled_robusta_production_dataframe.columns = ["Country", "Market_Year", "Robusta Production Share"]

                temp_dataframe = scaled_arabica_production_dataframe.merge(scaled_robusta_production_dataframe, on = ["Country", "Market_Year"])
                results.append(temp_dataframe)

            return(pandas.concat(results, ignore_index=True))

        elif country_or_category != "All":
            temp_dataframe = self.production_data[(self.production_data["Country_Name"] == country_or_category) 
                                                      & ((self.production_data["Attribute_Description"] == "Arabica Production") 
                                                         | (self.production_data["Attribute_Description"] == "Robusta Production"))]

            scaled_arabica_production = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"]["Value (60kg bags)"] / temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"]["Market_Year"].map(total_arabica_production)
            scaled_arabica_production_dataframe = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"].drop(["Attribute_Description", "Value (60kg bags)"], axis = 1 )
            scaled_arabica_production_dataframe = pandas.concat([scaled_arabica_production_dataframe, scaled_arabica_production], axis = 1)
            scaled_arabica_production_dataframe.columns = ["Country", "Market_Year", "Arabica Production Share"]

            scaled_robusta_production = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"]["Value (60kg bags)"] / temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"]["Market_Year"].map(total_robusta_production)
            scaled_robusta_production_dataframe = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"].drop(["Attribute_Description", "Value (60kg bags)"], axis = 1 )
            scaled_robusta_production_dataframe = pandas.concat([scaled_robusta_production_dataframe, scaled_robusta_production], axis = 1)
            scaled_robusta_production_dataframe.columns = ["Country", "Market_Year", "Robusta Production Share"]

            temp_dataframe = scaled_arabica_production_dataframe.merge(scaled_robusta_production_dataframe, on = ["Country", "Market_Year"])
            return temp_dataframe
        
        else:        
            results = []
            for country in self.production_data["Country"].unique():
                temp_dataframe = self.production_data[(self.production_data["Country"] == country) 
                                                      & ((self.production_data["Attribute_Description"] == "Arabica Production") 
                                                         | (self.production_data["Attribute_Description"] == "Robusta Production"))]

                scaled_arabica_production = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"]["Value (60kg bags)"] / temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"]["Market_Year"].map(total_arabica_production)
                scaled_arabica_production_dataframe = temp_dataframe[temp_dataframe["Attribute_Description"] == "Arabica Production"].drop(["Attribute_Description", "Value (60kg bags)"], axis = 1 )
                scaled_arabica_production_dataframe = pandas.concat([scaled_arabica_production_dataframe, scaled_arabica_production], axis = 1)
                scaled_arabica_production_dataframe.columns = ["Country", "Market_Year", "Arabica Production Share"]

                scaled_robusta_production = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"]["Value (60kg bags)"] / temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"]["Market_Year"].map(total_robusta_production)
                scaled_robusta_production_dataframe = temp_dataframe[temp_dataframe["Attribute_Description"] == "Robusta Production"].drop(["Attribute_Description", "Value (60kg bags)"], axis = 1 )
                scaled_robusta_production_dataframe = pandas.concat([scaled_robusta_production_dataframe, scaled_robusta_production], axis = 1)
                scaled_robusta_production_dataframe.columns = ["Country", "Market_Year", "Robusta Production Share"]

                temp_dataframe = scaled_arabica_production_dataframe.merge(scaled_robusta_production_dataframe, on = ["Country", "Market_Year"])
                results.append(temp_dataframe)

            return(pandas.concat(results, ignore_index=True))
    
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
        
    def list_countries_in_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
        
    def get_temperature_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        results_dataframe = self.temperature_data[self.temperature_data["Country"].isin(self.countries_in_ICO_category[ICO_category])]
        if ICO_category != "Robustas":
            return results_dataframe[results_dataframe["Arabica Production"] == True]
        else:
            return results_dataframe[results_dataframe["Robusta Production"] == True]
    
    
# ------------------------- ICO Composite Indicator Data -------------------------

class ICO_Composite_Indicator(object):
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
        
    def list_countries_in_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.countries_in_ICO_category[ICO_category]
    
    def get_ICO_indicator_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.ICO_indicator_data[ICO_category]
       
    def get_ICO_composite_indicator_data_by_ICO_category(self, ICO_category = "Brazilian Naturals"):
        return self.ICO_indicator_data[ICO_category]