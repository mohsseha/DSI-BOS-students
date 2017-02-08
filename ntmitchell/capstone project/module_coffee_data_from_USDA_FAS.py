
# coding: utf-8

# In[4]:

import pandas

# import matplotlib.pyplot as plt
# %matplotlib inline


# In[86]:

class Coffee_Data(object):
    production_data = None
    ICO_categories = ["Brazilian Naturals", "Colombian Milds", "Other Milds", "Robustas"]
    countries_in_ICO_category = dict.fromkeys(ICO_categories, None)
    production_by_ICO_category = dict.fromkeys(ICO_categories, None)
    ending_stock_by_ICO_category = dict.fromkeys(ICO_categories, None)
    
    def __init__(self):
        self.production_data = pandas.read_csv("../datasets/capstone/coffee-production-2016-1990--USDA-FAS--psd_coffee.csv")
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
    


# In[87]:

# data = Coffee_Data()
# data.get_aggregate_production_data("Robustas").head()
# data.get_countries_in_category("Robustas")[0:5]
# data.get_ending_stocks("Robustas").head()


# In[68]:

# data.get_aggregate_production_data("Brazilian Naturals").plot()
# data.get_aggregate_production_data("Colombian Milds").plot()
# data.get_aggregate_production_data("Other Milds").plot()
# data.get_aggregate_production_data("Robustas").plot()


# plt.title("Annual coffee production by ICO category")
# plt.legend(["Brazilian Naturals", "Colombian Milds", "Other Milds", "Robusta"], loc = 'best')
# plt.xlabel("Market year")
# plt.ylabel("Production (60 kg bags)")
# plt.show()


# In[27]:

# production_data[(production_data["Country_Name"] == "Bolivia") & (production_data["Attribute_Description"] == "Arabica Production") & (production_data["Market_Year"] > 1990)].plot(x = "Market_Year", y = "Value (60kg bags)")


# In[ ]:



