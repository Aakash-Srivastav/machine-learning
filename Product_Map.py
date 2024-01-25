import pandas as pd
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from IPython.display import display

flip_data = read_csv('flipkart_com-ecommerce_sample.csv') #reading flipkart csv file
flip_product_name = flip_data['product_name'].tolist() #storing flipkart product name in a list
flip_retail_price = flip_data['retail_price'].tolist() #storing flipkart retail price in a list
flip_discounted_price = flip_data['discounted_price'].tolist() #storing flipkart discounted price in a list
flip_product_name_unique = [*set(flip_product_name)] #storing unique flipkart product in a list

flip_product_category_tree = flip_data['product_category_tree'].tolist() #storing flpikart category tree in a list

#storing only initial/broad category into a list for flipkart
flip_product_category = []
for i in range(0,len(flip_product_category_tree)):
    category_flip = flip_product_category_tree[i][2:-2].split(" >>")[0]
    flip_product_category.append(category_flip)

#mapping product with initial/broad category of flipkart
flip_map = {}
for i in range(0,len(flip_product_name)):
    if flip_product_name[i] not in flip_map:
        flip_map[flip_product_name[i]]= flip_product_category[i]

flip_keys = list(flip_map.keys()) #stoing only keys from flip_map 
flip_val = list(flip_map.values()) #stoing only values from flip_map

amz_data = read_excel('amz_com-ecommerce_sample.xlsx') #reading amazon excel file
amz_product_name = amz_data['product_name'].tolist() #storing amazon product name in a list
amz_retail_price = amz_data['retail_price'].tolist() #storing amazon retail price in a list
amz_discounted_price = amz_data['discounted_price'].tolist() #storing amazon discounted price in a list
amz_product_name_unique = [*set(amz_product_name)] #storing unique amazon product in a list

amz_product_category_tree = amz_data['product_category_tree'].tolist() #storing amazon category tree in a list

#storing only initial/broad category into a list for amazon
amz_product_category = []
for i in range(0,len(amz_product_category_tree)):
    category_amz = amz_product_category_tree[i][2:-2].split(" >>")[0]
    amz_product_category.append(category_amz)

#mapping product with initial/broad category of amazon
amz_map = {}
for i in range(0,len(amz_product_name)):
    if amz_product_name[i] not in amz_map:
        amz_map[amz_product_name[i]]= amz_product_category[i]

amz_keys = list(amz_map.keys()) #stoing only keys from amaz_map
amz_val = list(amz_map.values()) #stoing only values from amaz_map

def product_matching():
    """
    1) Firstly it will take input of product
    2) Then it will check if the product is in flipkart set or not
    3) If the product is in flipkart set then it will check whether the product is in amazon set or not
        3.1) If the product is in amazon set as well then it will return a dataframe which includes names of product , retail price of product and discounted price of product on both ecommerce site
        3.2) If the product is not in amazon set then it will check which is the closest product within the same category which matches to that product and then it will return a datafrane contaning the above details
    4) If the product is not in flipkart dataset then it will return that input product does not exist
    """
    test = input('please enter product name : ')

    if test in flip_product_name_unique:
        if test in amz_product_name_unique:
            flip_position = flip_product_name.index(test)
            amz_position = amz_product_name.index(test)
            data = [[test, flip_retail_price[flip_position], flip_discounted_price[flip_position], test, amz_retail_price[amz_position], amz_discounted_price[amz_position]]]
            frame = pd.DataFrame(data,columns=['Product Name In Flipkart','Retail Price In Flipkart','Discounted Price In Flipkart','Product Name In Amazon','Retail Price In Amazon','Discounted Price In Amazon'])
            return (print(display(frame)))
        else:
            category = flip_map[test]
            keys  = [test]
            for i in range(0,len(amz_val)):
                if amz_val[i] == category:
                    keys.append(amz_keys[i])
            
            vectorizer = CountVectorizer()
            features = vectorizer.fit_transform(keys).todense()
            output= []
            for num in range(1,len(features)):
                output.append(euclidean_distances(features[0],features[num])[0][0])
            position = output.index(min(output))
            match_product = keys[int(position)+1]
            flip_position = flip_product_name.index(test)
            amz_position = amz_product_name.index(match_product)
            data = [[test, flip_retail_price[flip_position], flip_discounted_price[flip_position], match_product, amz_retail_price[match_product], amz_discounted_price[match_product]]]
            frame = pd.DataFrame(data,columns=['Product Name In Flipkart','Retail Price In Flipkart','Discounted Price In Flipkart','Product Name In Amazon','Retail Price In Amazon','Discounted Price In Amazon'])
            return (display(frame))
    else:
        return (f'{test} is not in flipkart dataset')


if "__main__" == __name__:
    product_matching()
