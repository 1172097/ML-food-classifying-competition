"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""
import pandas as pd
import numpy as np
import re
from collections import Counter

def process_q2(text):
    if pd.isnull(text):
        return None
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    numbers = [float(num) for num in numbers]
    filtered_numbers = [num for num in numbers if 1 <= num <= 10]
    if filtered_numbers:
        result = sum(filtered_numbers) / len(filtered_numbers)
    elif ',' in text:
        return text.count(',') + 1
    else:
        result = len(text) / 5
    return np.ceil(result)

def extract_numeric(value):
    if isinstance(value, str):
        value = value.lower().strip()
        numbers = re.findall(r'\d+', value)
        if numbers:
            return int(numbers[0])
        return "unknown"
    return value



def clean_drink(value):
    drink_map = {
        "coke": ["coke", "cola", "diet coke", "pepsi", "diet pepsi", "soda", "pop", "soft", "fanta", "sprite", "ginger"],
        "juice": ["juice", "nestea", "lemonade"],
        "tea": ["tea", "matcha"],
        "milk": ["ayran", "milk", "dairy"],
        "beer": ["beer", "wine", "alcoholic", "sake", "soju"],
        "water": ["water"],
        "soup": ["miso", "soup"]
    }
    if isinstance(value, str):
        value = re.sub(r'[^a-z0-9 ]', '', value.lower().strip())
        for category, keywords in drink_map.items():
            if any(keyword in value for keyword in keywords):
                return category
    return "unknown"

def one_hot_encode(df, column_name, categories):
    if column_name in df.columns:
        df[column_name] = df[column_name].str.lower().str.strip()
        df[column_name] = df[column_name].apply(lambda x: x.split(",") if isinstance(x, str) else [])
        for category in categories:
            df[f"{column_name}_{category}"] = df[column_name].apply(lambda x: 1 if category in x else 0)
        df.drop(columns=[column_name], inplace=True)
    return df

def clean_and_process_csv(file_path):
    df = pd.read_csv(file_path, dtype=str)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.drop_duplicates().fillna("unknown")

    df.rename(columns={
    'q1:_from_a_scale_1_to_5,_how_complex_is_it_to_make_this_food?_(where_1_is_the_most_simple,_and_5_is_the_most_complex)': 'q1',
    'q7:_when_you_think_about_this_food_item,_who_does_it_remind_you_of?':'q7',
    "q3:_in_what_setting_would_you_expect_this_food_to_be_served?_please_check_all_that_apply":'q3'
    }, inplace=True)
    
    # Process Q2
    q2_col = "q2:_how_many_ingredients_would_you_expect_this_food_item_to_contain?"
    if q2_col in df.columns:
        df['q2'] = df[q2_col].apply(process_q2)
        df.drop(columns=[q2_col], inplace=True)
    
    # Process Q4
    q4_col = "q4:_how_much_would_you_expect_to_pay_for_one_serving_of_this_food_item?"
    if q4_col in df.columns:
        df['q4'] = df[q4_col].apply(extract_numeric)
        df.drop(columns=[q4_col], inplace=True)
    
    # Process Q6
    q6_col = "q6:_what_drink_would_you_pair_with_this_food_item?"
    if q6_col in df.columns:
        df['q6'] = df[q6_col].apply(clean_drink)
        for cat in ["coke", "juice", "tea", "milk", "beer", "water", "soup"]:
            df[f'drink_{cat}'] = df['q6'].apply(lambda x: 1 if x == cat else 0)
        df.drop(columns=[q6_col, "q6"], inplace=True)

    # Process Q8
    q8_col = "q8:_how_much_hot_sauce_would_you_add_to_this_food_item?"

    def map_hot_sauce_response(text):
        text = text.lower().strip()
        if text == "unknown":
            return "none"
        elif text == "a little (mild)":
            return "mild"
        elif text == "a moderate amount (medium)":
            return "medium"
        elif text == "a lot (hot)":
            return "hot"
        elif text == "i will have some of this food item with my hot sauce":
            return "with_hot_sauce"
        else:
            return "unknown"

    print(df[q8_col].unique()[:10])  


    if q8_col in df.columns:
        df[q8_col] = df[q8_col].astype(str).apply(map_hot_sauce_response)
        for category in ["none", "mild", "medium", "hot", "with_hot_sauce"]:
            df[f'hot_sauce_{category}'] = df[q8_col].apply(lambda x: 1 if x == category else 0)
        df.drop(columns=[q8_col], inplace=True)  
        
    # Process Q7 and Q3 with one-hot encoding
    df = one_hot_encode(df, "q7", 
                        ["parents", "siblings", "friends", "teachers", "strangers"])
    df = one_hot_encode(df, "q3", 
                        ["week day lunch", "week day dinner", "weekend lunch", "weekend dinner", "at a party", "late night snack"])
    
    # Process Q5 - Bag of Words
    text_column = "q5:_what_movie_do_you_think_of_when_thinking_of_this_food_item?"
    word_counts = Counter()

    stop_words = set("i me my myself we our ours you your they them their what which who this that these those am is are was were be been being have has had do does did doing a an the and but if or because as until while of at by for with about between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y".split())

    # Helper to clean special characters
    def clean_special_symbols(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

    if text_column in df.columns:
        df[text_column] = df[text_column].apply(clean_special_symbols)
        df[text_column] = df[text_column].apply(
            lambda text: [word for word in text.strip().split() if word not in stop_words]
        )
        for words in df[text_column]:
            word_counts.update(words)
        
        unique_words = sorted(word_counts.keys())
        bow_df = pd.DataFrame(0, index=df.index, columns=unique_words)
        for idx, words in enumerate(df[text_column]):
            for word in words:
                bow_df.at[idx, word] = 1
        df.drop(columns=[text_column], inplace=True)
        df = pd.concat([df, bow_df], axis=1)


    """
        # Process Q5 - Bag of Words
    text_column = "q5:_what_movie_do_you_think_of_when_thinking_of_this_food_item?"
    word_counts = Counter()
    
    stop_words = set("i me my myself we our ours you your they them their what which who this that these those am is are was were be been being have has had do does did doing a an the and but if or because as until while of at by for with about between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y".split())
    if text_column in df.columns:
        df[text_column] = df[text_column].apply(lambda text: [word for word in text.lower().strip().split() if word not in stop_words] if isinstance(text, str) else [])
        for words in df[text_column]:
            word_counts.update(words)
        unique_words = sorted(word_counts.keys())
        bow_df = pd.DataFrame(0, index=df.index, columns=unique_words)
        for idx, words in enumerate(df[text_column]):
            for word in words:
                bow_df.at[idx, word] = 1
        df.drop(columns=[text_column], inplace=True)
        df = pd.concat([df, bow_df], axis=1)
    """

    
    for column in df.columns:
        if column != "label":
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
    print(df.columns)
    df.drop(columns=['id'], errors='ignore', inplace=True)


    output_path = "cleaned_" + file_path
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    return output_path

# clean_and_process_csv("cleaned_data_combined_modified.csv")



# Example usage:
# cleaned_file = clean_and_process_csv("cleaned_data_combined_modified.csv")
def load_forest_from_single_csv(filename="final_model.csv"):
    """Load multiple trees from a single CSV file."""
    df = pd.read_csv(filename)
    forest = [df[df["tree_id"] == i].reset_index(drop=True) for i in df["tree_id"].unique()]
    return forest

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    
    return y

def predict_single(tree_df, sample):
    """Traverse the tree manually to classify a single sample."""
    node_id = 0  # Start at the root

    while True:
        node = tree_df.loc[node_id]

        if node["feature"] == -2:  
            values = np.fromstring(node["values"][1:-1], sep=" ") 
            return np.argmax(values)  

        feature_idx = int(node["feature"])
        threshold = node["threshold"]
        
        if feature_idx >= len(sample):
            node_id = int(node["left_child"])
            continue
            
        if sample[feature_idx] <= threshold:
            node_id = int(node["left_child"])
        else:
            node_id = int(node["right_child"])

def predict_forest(forest, X):
    """Predict multiple samples using majority voting from multiple trees."""
    X_array = X.to_numpy()
    
    all_predictions = []
    for tree in forest:
        tree_predictions = []
        for x in X_array:
            pred = predict_single(tree, x)
            tree_predictions.append(pred)
        all_predictions.append(tree_predictions)
    
    predictions = np.array(all_predictions)
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    label_map = {
        0: 'Pizza',
        1: 'Shawarma', 
        2: 'Sushi'
    }
    
    cleaned_data = clean_and_process_csv(filename)
    data = pd.read_csv(cleaned_data)

    if 'label' in data.columns:
        data = data.drop(columns=['label'])

    forest_trees = load_forest_from_single_csv()
    predictions = predict_forest(forest_trees, data)
    
    preds = [label_map[p] for p in predictions]

    return preds




