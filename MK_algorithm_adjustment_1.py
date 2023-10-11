import os.path
import datetime
import pandas as pd
import pyodbc
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import time
import warnings
import argparse
import json
from datetime import datetime as dt



warnings.filterwarnings('ignore')

building_types_dict = {
    "кирпичный": ["кирпичный", "гиперпресованный кирпич", "жженый кирпич", "камень ракушечник", "керамзитовый кирпич",
                  "кирпич", "клинкерный кирпич", "ракушечник", "силикатный кирпич"],
    "панельный": ["панельный", "ж/б панели", "ж/б плиты", "крупнопанельные", "панель"],
    "монолитный": ["монолитный", "бетон", "бетонные блоки", "бетонокаркас", "блочный", "газобетон", "газоблок",
                   "газосиликатные блоки", "ж/б блок", "ж/б пеноблок", "железобетон", "керамзитобетон",
                   "керамзитобетонный блок", "керамзитовый блок", "крупноблочные", "монолит", "пенобетон", "пеноблочный",
                   "пенополистиролбетон", "пескоблок", "сплитерные блоки", "теплоблок", "эко-блок", "финблок"],
    "инное": ["инное","-", "бревенчатые", "брусчатый", "дерево", "дощатые", "ИЗОДОМ", "каркасно-насыпной",
              "каркасно-камышитовый обложенный кирпичом", "керамзитобетон", "керамзитобетонный блок", "крупноблочные",
              "ЛСТК", "монолит", "монолит шлаколитой", "пенобетон", "пеноблочный", "пенополистиролбетон",
              "пескоблок", "СИП панели", "сборно-щитовой", "сплиттерные блоки", "сруб", "сэндвич панели",
              "теплоблок", "шлак", "шлакобетон", "шлакоблок", "шлакозалитый", "шпальные", "эко-блок",
              "деревобетон (арболит)", "фибролитовые блоки", "фибролитовые плиты", "финблок", "каркасно-обшивные",
              "деревянные рубленные"] }

renovation_dict = {"хорошее": "хорошее",
                   "среднее": ["среднее", "-", ""],
                   "требует ремонта" : ["требуется ремонт", "требует ремонта"],
                   "свободная планировка": ["свободная планировка"],
                   "черновая отделка": "черновая отделка"}

furniture_map = {
    "полностью": "полностью",
    "частично": "частично",
    "без мебели": ["Квартира меблирована", "-", "", "без мебели",  "_", '-', '']
}

class RealEstateData:

    LOCAL_DATA_PATH = 'local_data.csv'

    def __init__(self, connection_str, fetch_data=True):
        self.connection_str = connection_str
        self.oldest_date_fetched = None

        if fetch_data:
            self.fetch_and_store_data()

    def fetch_and_store_data(self, force_fetch=False):
        # Check if file exists and was created today
        if not force_fetch and os.path.exists(self.LOCAL_DATA_PATH):
            file_timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(self.LOCAL_DATA_PATH))
            if file_timestamp.date() == datetime.datetime.today().date():
                return

        # Fetch data for the last 93 days directly
        query = """
        SELECT * FROM [RealEstateData].[dbo].[KrishaAdsFromUAS] 
        WHERE CAST([CreateDate] AS DATE) BETWEEN DATEADD(DAY, -93, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE)
        ORDER BY CreateDate DESC
        """
        try:
            with pyodbc.connect(self.connection_str) as conn:
                df = pd.read_sql(query, conn)
            print(f"Fetched {len(df)} rows from the database.")
            df.to_csv(self.LOCAL_DATA_PATH, index=False)
        except Exception as e:
            print(f"Error fetching and storing data: {e}")

    def preprocess_dataframe(self, df):
        df['Rooms'] = df['Rooms'].astype(int)
        df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()]
        df['Year'] = df['Year'].astype(int)
        df['floor_new'] = df.apply(self.calculate_floor_new, axis=1)
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        # Ensure the 'Price' and 'Square' columns are numeric
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Square'] = pd.to_numeric(df['Square'], errors='coerce')

        # Calculate price per square meter and store it in a new column
        df['price_per_1sqrm'] = df['Price'] / df['Square']
        return df

    def preprocess_floor_from_json(self, floor_new_value):
        # Convert the input value to lowercase for consistency if it's a string
        if isinstance(floor_new_value, str):
            floor_new_value = floor_new_value.lower()

        # Map text values to their corresponding numeric values
        floor_new_mapping = {
            "мансарда": 3,
            "пентхаус": 3,
            "цокольный": 1,
            "цоколь": 1
        }

        return floor_new_mapping.get(floor_new_value, floor_new_value)

    def calculate_floor_new(self, row):
        floor = row['Floor']
        try:
            n, x = [int(i) for i in floor.split(' из ')]
            if n == x:
                return 3
            elif n == 1:
                return 1
            else:
                return 2
        except ValueError:
            return None

    def calculate_distance(self, row, evaluated_apartment):
        apartment_loc = [radians(round(row['Latitude'], 4)), radians(round(row['Longitude'], 4))]
        evaluated_apartment_loc = [radians(round(evaluated_apartment['latitude'], 4)), radians(round(evaluated_apartment['longitude'], 4))]
        distance = haversine_distances([apartment_loc, evaluated_apartment_loc])[0,1] * 6371000
        return distance

    def filter_dataframe(self, df, evaluated_apartment, radius, building_types_dict, renovation_dict):
        # Filter by distance
        df['distance'] = df.apply(
            lambda row: self.calculate_distance(row, evaluated_apartment),
            axis=1, result_type='expand')
        within_radius_df = df[df['distance'] <= radius]
        print(f"Within radius DataFrame shape: {within_radius_df.shape}")

        # Filter by room count
        same_room_count_df = within_radius_df[within_radius_df['Rooms'] == evaluated_apartment['room_count']].copy()
        print(f"same_room_count_df DataFrame shape: {same_room_count_df.shape}")

        # Filter by total space
        same_room_count_df.loc[:, 'Square'] = pd.to_numeric(same_room_count_df['Square'], errors='coerce')
        min_space = evaluated_apartment['total_space'] * 0.9
        max_space = evaluated_apartment['total_space'] * 1.1
        similar_space_df = same_room_count_df[
            (same_room_count_df['Square'] >= min_space) & (same_room_count_df['Square'] <= max_space)
            ]
        print(f"similar_space_df shape: {similar_space_df.shape}")

        # Filter by construction year
        min_year = evaluated_apartment['year_of_construction'] - 5
        max_year = evaluated_apartment['year_of_construction'] + 5
        within_year_df = similar_space_df[
            (similar_space_df['Year'] >= min_year) & (similar_space_df['Year'] <= max_year)
            ]
        print(f"Within year DataFrame shape: {within_year_df.shape}")

        # Debugging: Print unique building types before filtering
        print("Unique building types in DataFrame before filtering:", within_year_df['Building'].str.lower().unique())

        building_type = evaluated_apartment['wall_material'].lower() if evaluated_apartment['wall_material'] else '-'

        # Debugging: Print the building_type for filtering
        print(f"Building type from user input: {building_type}")

        same_material_df = within_year_df[within_year_df['Building'].str.lower() == building_type]

        if same_material_df.empty:
            for key, building_list in building_types_dict.items():
                if building_type in map(str.lower, building_list):
                    alternative_building_types = [b.lower() for b in building_list]
                    same_material_df = within_year_df[
                        within_year_df['Building'].str.lower().isin(alternative_building_types)
                    ]
                    break
        print(f"same_material_df shape: {same_material_df.shape}")

        condition_value = evaluated_apartment['condition']
        if pd.isna(condition_value) or condition_value == '':
            condition_value = '-'

        # Similar to building types, look up the condition in the renovation_dict
        if condition_value in renovation_dict:
            renovation_values = renovation_dict[condition_value]
            if not isinstance(renovation_values, list):
                renovation_values = [renovation_values]
        else:
            renovation_values = [condition_value]

        same_condition_df = same_material_df[same_material_df['Renovation'].isin(renovation_values)]
        print(f"same_condition_df shape: {same_condition_df.shape}")

        same_floor_df = same_condition_df[same_condition_df['floor_new'] == evaluated_apartment['floor_new']]
        print(f"same_floor_df shape: {same_floor_df.shape}")  # Added print statement

        # Filter based on furniture
        furniture_value = evaluated_apartment['furniture']
        if pd.isna(furniture_value) or furniture_value == '':
            furniture_value = 'без мебели'

        # If the furniture value from evaluated apartment is in the furniture_map, get the corresponding DWH values
        if furniture_value in furniture_map:
            dwh_furniture_values = furniture_map[furniture_value]
            if not isinstance(dwh_furniture_values, list):  # if the mapped value is not a list, make it a list
                dwh_furniture_values = [dwh_furniture_values]
        else:
            dwh_furniture_values = [furniture_value]  # Use the provided value as it is if not in the furniture map
        print(f"dwh_furniture_values : {dwh_furniture_values} ")

        # Standardizing furniture values before filtering:
        same_floor_df['Furniture'] = same_floor_df['Furniture'].str.lower().str.strip()
        print(f"same_floor_df['Furniture'].dtypes : {same_floor_df['Furniture'].dtypes} ")
        dwh_furniture_values = [value.lower().strip() for value in dwh_furniture_values]
        matches = same_floor_df['Furniture'].isin(dwh_furniture_values)
        print(matches.any())

        # Now apply the filtering
        furniture_filtered_df = same_floor_df[same_floor_df['Furniture'].isin(dwh_furniture_values)]
        print(f"After initial filtering by furniture DataFrame shape: {furniture_filtered_df.shape}")

        if furniture_filtered_df.empty:
            print("No suitable apartments found after furniture filter, fetching more data...")
            return pd.DataFrame()  # Return empty DataFrame
        else:
            return furniture_filtered_df

    def alternative_filtering(self, last_30_days_df, evaluated_apartment, radius, building_types_dict, renovation_dict):

        # 1. Filter by distance
        last_30_days_df['distance'] = last_30_days_df.apply(
            lambda row: self.calculate_distance(row, evaluated_apartment),
            axis=1, result_type='expand')
        within_radius_df = last_30_days_df[last_30_days_df['distance'] <= 1000]

        print(f"Within 1000 meters radius DataFrame shape: {within_radius_df.shape}")

        if within_radius_df.shape[0] < 3:
            print("Expanding search to 3000 meters due to insufficient results...")

            # Filter for extended 3000 meters radius:
            extended_radius_df = last_30_days_df[last_30_days_df['distance'] <= 3000]

            # Apply price adjustments based on distances:
            extended_radius_df.loc[extended_radius_df['distance'] <= 1500, 'price_per_1sqrm'] *= 0.98
            extended_radius_df.loc[(extended_radius_df['distance'] > 1500) & (
                    extended_radius_df['distance'] <= 2000), 'price_per_1sqrm'] *= 0.96
            extended_radius_df.loc[(extended_radius_df['distance'] > 2000) & (
                    extended_radius_df['distance'] <= 3000), 'price_per_1sqrm'] *= 0.94

            # Update within_radius_df to use the extended and adjusted data:
            within_radius_df = extended_radius_df
            print(f"Within 3000 meters radius DataFrame shape: {within_radius_df.shape}")
            if within_radius_df.shape[0] < 3:
                print("Alternative filtering didn't fetch more data. Stopped at radius step.")
                return pd.DataFrame()


        # 2. Filter by room count
        room_adjustment_factors = {
            1: {2: 0.98, 3: 1.00, 4: 1.02, 5: 1.02, 6: 1.02, 7: 1.02},
            2: {1: 1.02, 3: 1.02, 4: 1.02, 5: 1.02, 6: 1.02, 7: 1.02},
            3: {1: 1, 2: 0.98, 3: 1.00, 4: 1.02, 5: 1.02, 6: 1.02, 7: 1.02},
            4: {1: 0.98, 2: 0.96, 3: 0.98, 4: 1.00, 5: 1.00, 6: 1.00, 7: 1.00}
        }

        room_count = evaluated_apartment['room_count']
        rooms_to_consider = {room_count}
        same_room_count_df = within_radius_df[within_radius_df['Rooms'].isin(rooms_to_consider)].copy()
        print(f"same_room_count_df: {same_room_count_df.shape[0]}")
        # Check if the primary filter yields less than 3 apartments
        if same_room_count_df.shape[0] < 3:
            rooms_to_consider = {1, 2, 3, 4, 5, 6, 7}  # Considering all room options
            same_room_count_df = within_radius_df[within_radius_df['Rooms'].isin(rooms_to_consider)].copy()

            # Apply room adjustments directly after selecting the similar apartments
            evaluated_room_count = evaluated_apartment['room_count']
            if evaluated_room_count in room_adjustment_factors:
                for room, factor in room_adjustment_factors[evaluated_room_count].items():
                    mask = same_room_count_df['Rooms'] == room
                    same_room_count_df.loc[mask, 'price_per_1sqrm'] *= factor

            print(f"same_room_count_df DataFrame shape AFTER the filter: {same_room_count_df.shape}")

            if same_room_count_df.shape[0] < 3:
                print("Alternative filtering didn't fetch more data. Stopped at room count step.")
                return pd.DataFrame()

        # 3. Filter by total space
        same_room_count_df.loc[:, 'Square'] = pd.to_numeric(same_room_count_df['Square'], errors='coerce')
        min_space = evaluated_apartment['total_space'] * 0.9
        max_space = evaluated_apartment['total_space'] * 1.1
        similar_space_df = same_room_count_df[
            (same_room_count_df['Square'] >= min_space) & (same_room_count_df['Square'] <= max_space)
            ]

        def get_square_adjustment_factor(difference):
            if -30 <= difference <= -16:
                return 0.94
            elif -15 <= difference <= -6:
                return 0.97
            elif -5 <= difference <= 5:
                return 1
            elif 6 <= difference <= 15:
                return 1.03
            elif 16 <= difference <= 30:
                return 1.06
            else:
                return 1  # default adjustment factor

        print(f"similar_space_df: {similar_space_df.shape[0]}")

        if similar_space_df.shape[0] < 3:
            # Calculate the square difference for all apartments
            all_space_df = same_room_count_df.copy()
            all_space_df['square_difference'] = all_space_df['Square'] - evaluated_apartment['total_space']

            # Remove apartments where the absolute 'square_difference' is greater than 30
            all_space_df = all_space_df[all_space_df['square_difference'].abs() <= 30]

            # Apply adjustments based on the square difference
            all_space_df['price_per_1sqrm'] *= all_space_df['square_difference'].apply(get_square_adjustment_factor)
            similar_space_df = all_space_df
            print(f"similar_space_df AFTER filtering: {similar_space_df.shape[0]}")

            if similar_space_df.shape[0] < 3:
                print("Alternative filtering didn't fetch more data. Stopped at space step. Going to year step.")
                rooms_to_consider = {1, 2, 3, 4, 5, 6, 7}
                similar_space_df = within_radius_df[within_radius_df['Rooms'].isin(rooms_to_consider)].copy()

                # Apply square adjustments
                similar_space_df['square_difference'] = similar_space_df['Square'] - evaluated_apartment['total_space']
                similar_space_df['price_per_1sqrm'] *= similar_space_df['square_difference'].apply(
                    get_square_adjustment_factor)

                # Apply room adjustments
                evaluated_room_count = evaluated_apartment['room_count']
                if evaluated_room_count in room_adjustment_factors:
                    for room, factor in room_adjustment_factors[evaluated_room_count].items():
                        mask = similar_space_df['Rooms'] == room
                        similar_space_df.loc[mask, 'price_per_1sqrm'] *= factor

                print(f"similar_space_df AFTER the filter and adjustments: {similar_space_df.shape[0]}")

        # 4. Filter by construction year
        min_year = evaluated_apartment['year_of_construction'] - 5
        max_year = evaluated_apartment['year_of_construction'] + 5
        within_year_df = similar_space_df[
            (similar_space_df['Year'] >= min_year) & (similar_space_df['Year'] <= max_year)
            ]
        print(f"Within year DataFrame shape: {within_year_df.shape}")

        if within_year_df.shape[0] < 3:
            min_year_expanded = evaluated_apartment['year_of_construction'] - 15
            max_year_expanded = evaluated_apartment['year_of_construction'] + 15
            within_expanded_year_df = similar_space_df[
                (similar_space_df['Year'] >= min_year_expanded) & (similar_space_df['Year'] <= max_year_expanded)
                ]
            print("Number of apartments before year filter:", similar_space_df.shape[0])
            print("Min expanded year:", min_year_expanded)
            print("Max expanded year:", max_year_expanded)

            # Apply the adjustments here so you can directly use them in the following filters.
            year_diff = within_expanded_year_df['Year'] - evaluated_apartment['year_of_construction']
            within_expanded_year_df.loc[year_diff.between(-15, -11), 'price_per_1sqrm'] *= 1.08
            within_expanded_year_df.loc[year_diff.between(-10, -6), 'price_per_1sqrm'] *= 1.04
            # No adjustment for -5 to +5 as it's a multiplier of 1
            within_expanded_year_df.loc[year_diff.between(6, 10), 'price_per_1sqrm'] *= 0.96
            within_expanded_year_df.loc[year_diff.between(11, 15), 'price_per_1sqrm'] *= 0.92
            # Assign the expanded dataframe to within_year_df for use in subsequent filters
            within_year_df = within_expanded_year_df.copy()
            print("Number of apartments AFTER year filter:", within_radius_df.shape[0])
            if within_year_df.shape[0] < 3:
                print("Alternative filtering didn't fetch more data. Stopped at year step. Going to total space step.")
                all_space_df = same_room_count_df.copy()
                all_space_df['square_difference'] = all_space_df['Square'] - evaluated_apartment['total_space']

                # Apply adjustments based on the square difference
                all_space_df['price_per_1sqrm'] *= all_space_df['square_difference'].apply(get_square_adjustment_factor)

                within_year_df = all_space_df
                print(f"similar_space_df shape AFTER FILTERS: {within_year_df.shape}")


        # 5. Filter by building type
        # Debugging: Print unique building types before filtering
        print("Unique building types in DataFrame BEFORE filtering:", within_year_df['Building'].str.lower().unique())
        building_type = evaluated_apartment['wall_material'].lower() if evaluated_apartment['wall_material'] else '-'

        # Debugging: Print the building_type for filtering
        print(f"Building type from user input: {building_type}")
        adjustment_matrix = {
            'кирпичный': {
                'кирпичный': 1,
                'панельный': 1.1,
                'монолитный': 1.05,
                'иное': 1.15
            },
            'панельный': {
                'кирпичный': 0.9,
                'панельный': 1,
                'монолитный': 0.95,
                'иное': 1.05
            },
            'монолитный': {
                'кирпичный': 0.95,
                'панельный': 1.05,
                'монолитный': 1,
                'иное': 1.1
            },
            'иное': {
                'кирпичный': 0.85,
                'панельный': 0.95,
                'монолитный': 0.9,
                'иное': 1
            }
        }

        same_material_df = within_year_df[within_year_df['Building'].str.lower() == building_type]

        if same_material_df.empty:
            for key, building_list in building_types_dict.items():
                if building_type in map(str.lower, building_list):
                    alternative_building_types = [b.lower() for b in building_list]
                    same_material_df = within_year_df[
                        within_year_df['Building'].str.lower().isin(alternative_building_types)
                    ]
                    break
        print(f"same_material_df shape: {same_material_df.shape}")

        if same_material_df.shape[0] < 3:
            # Fetching all available material types
            all_materials_df = within_year_df.copy()
            evaluated_building_type = evaluated_apartment['wall_material'].lower()

            # Apply adjustments
            for index, row in all_materials_df.iterrows():
                try:
                    adjustment_factor = adjustment_matrix[evaluated_building_type][row['Building'].lower()]
                except KeyError:  # If the material type doesn't exist in the matrix, no adjustment is made
                    adjustment_factor = 1

                all_materials_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

            same_material_df = all_materials_df
            print(f"AFTER adjusting by building type DataFrame shape: {same_material_df.shape}")
            if same_material_df.shape[0] < 3:
                min_year_expanded = evaluated_apartment['year_of_construction'] - 15
                max_year_expanded = evaluated_apartment['year_of_construction'] + 15
                within_expanded_year_df = similar_space_df[
                    (similar_space_df['Year'] >= min_year_expanded) & (similar_space_df['Year'] <= max_year_expanded)
                    ]
                print("Number of apartments before year filter:", similar_space_df.shape[0])
                print("Min expanded year:", min_year_expanded)
                print("Max expanded year:", max_year_expanded)

                # Apply the adjustments here so you can directly use them in the following filters.
                year_diff = within_expanded_year_df['Year'] - evaluated_apartment['year_of_construction']
                within_expanded_year_df.loc[year_diff.between(-15, -11), 'price_per_1sqrm'] *= 1.08
                within_expanded_year_df.loc[year_diff.between(-10, -6), 'price_per_1sqrm'] *= 1.04
                # No adjustment for -5 to +5 as it's a multiplier of 1
                within_expanded_year_df.loc[year_diff.between(6, 10), 'price_per_1sqrm'] *= 0.96
                within_expanded_year_df.loc[year_diff.between(11, 15), 'price_per_1sqrm'] *= 0.92
                # Assign the expanded dataframe to within_year_df for use in subsequent filters
                same_material_df = within_expanded_year_df.copy()
                print("Number of apartments AFTER year filter:", same_material_df.shape[0])

        # 6. Filter by condition
        condition_value = evaluated_apartment['condition']
        if pd.isna(condition_value) or condition_value == '':
            condition_value = '-'

        # Similar to building types, look up the condition in the renovation_dict
        if condition_value in renovation_dict:
            renovation_values = renovation_dict[condition_value]
            if not isinstance(renovation_values, list):
                renovation_values = [renovation_values]
        else:
            renovation_values = [condition_value]

        condition_adjustment_matrix = {
            'хорошее': {'хорошее': 1, 'среднее': 1.05, 'требует ремонта': 1.15, 'свободная планировка': 1.20, 'черновая отделка': 1.15},
            'среднее': {'хорошее': 0.95, 'среднее': 1, 'требует ремонта': 1.10, 'свободная планировка': 1.15, 'черновая отделка': 1.10},
            'требует ремонта': {'хорошее': 0.85, 'среднее': 0.90, 'требует ремонта' : 1.00 , 'свободная планировка': 1.05,  'черновая отделка': 1.00},
            'свободная планировка': {'хорошее': 0.80, 'среднее': 0.85, 'требует ремонта': 0.95, 'свободная планировка': 1.00, 'черновая отделка': 0.95},
            'черновая отделка': {'хорошее': 0.85, 'среднее': 0.90, 'требует ремонта': 1.00, 'свободная планировка': 1.05, 'черновая отделка': 1.00}
        }

        print(f'condition values before applying the filter', same_material_df['Renovation'].str.lower().unique())
        same_condition_df = same_material_df[same_material_df['Renovation'].isin(renovation_values)]
        print(f"same_condition_df shape: {same_condition_df.shape}")

        if same_condition_df.shape[0] < 3:
            # Identify which conditions to fetch based on evaluated_apartment condition
            condition_to_consider = evaluated_apartment['condition'].lower()
            if condition_to_consider in condition_adjustment_matrix:
                conditions_to_fetch = list(condition_adjustment_matrix[condition_to_consider].keys())

                expanded_conditions_df = same_material_df[
                    same_material_df['Renovation'].str.lower().isin(conditions_to_fetch)].copy()

                # Combine the apartments from the initial filter with the expanded filter
                expanded_conditions_df = pd.concat([same_condition_df, expanded_conditions_df]).drop_duplicates()
                print("Expanded conditions DataFrame shape:", expanded_conditions_df.shape)

                # Apply adjustments
                for index, row in expanded_conditions_df.iterrows():
                    try:
                        adjustment_factor = condition_adjustment_matrix[condition_to_consider][
                            row['Renovation'].lower()]
                    except KeyError:  # If the condition type doesn't exist in the matrix, no adjustment is made
                        adjustment_factor = 1

                    expanded_conditions_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

                same_condition_df = expanded_conditions_df

                if same_condition_df.shape[0] < 3:
                    # Fetching all available material types
                    all_materials_df = within_year_df.copy()
                    evaluated_building_type = evaluated_apartment['wall_material'].lower()

                    # Apply adjustments
                    for index, row in all_materials_df.iterrows():
                        try:
                            adjustment_factor = adjustment_matrix[evaluated_building_type][row['Building'].lower()]
                        except KeyError:  # If the material type doesn't exist in the matrix, no adjustment is made
                            adjustment_factor = 1

                        all_materials_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

                    same_condition_df = all_materials_df
                    print(f"AFTER adjusting by building type DataFrame shape: {same_condition_df.shape}")

        # 7. Filter by floor
        same_floor_df = same_condition_df[same_condition_df['floor_new'] == evaluated_apartment['floor_new']]
        print(f"same_floor_df shape: {same_floor_df.shape}")  # Added print statement

        floor_adjustment_matrix = {
            1: {1: 1, 2: 0.95, 3: 1.05},
            2: {1: 1.05, 2: 1, 3: 1.1},
            3: {1: 0.95, 2: 0.9, 3: 1}
        }
        if same_floor_df.shape[0] < 3:
            # Fetch all available floor types left after the 5th filter
            all_floors_df = same_condition_df.copy()
            evaluated_floor_type = evaluated_apartment['floor_new']

            # Apply adjustments
            for index, row in all_floors_df.iterrows():
                try:
                    adjustment_factor = floor_adjustment_matrix[evaluated_floor_type][row['floor_new']]
                except KeyError:  # If the floor type doesn't exist in the matrix, no adjustment is made
                    adjustment_factor = 1

                all_floors_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

            same_floor_df = all_floors_df
            if same_floor_df.shape[0] < 3:
                # Identify which conditions to fetch based on evaluated_apartment condition
                condition_to_consider = evaluated_apartment['condition'].lower()
                if condition_to_consider in condition_adjustment_matrix:
                    conditions_to_fetch = list(condition_adjustment_matrix[condition_to_consider].keys())

                    expanded_conditions_df = same_material_df[
                        same_material_df['Renovation'].str.lower().isin(conditions_to_fetch)].copy()

                    # Combine the apartments from the initial filter with the expanded filter
                    expanded_conditions_df = pd.concat([same_condition_df, expanded_conditions_df]).drop_duplicates()
                    print("Expanded conditions DataFrame shape:", expanded_conditions_df.shape)

                    # Apply adjustments
                    for index, row in expanded_conditions_df.iterrows():
                        try:
                            adjustment_factor = condition_adjustment_matrix[condition_to_consider][
                                row['Renovation'].lower()]
                        except KeyError:  # If the condition type doesn't exist in the matrix, no adjustment is made
                            adjustment_factor = 1

                        expanded_conditions_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

                    same_floor_df = expanded_conditions_df

        # 8. Filter by furniture

        # Filter based on furniture
        furniture_value = evaluated_apartment['furniture']
        print("Unique furniture types in DataFrame before filtering:", same_floor_df['Furniture'].str.lower().unique())
        if pd.isna(furniture_value) or furniture_value == '':
            furniture_value = 'без мебели'

        # If the furniture value from evaluated apartment is in the furniture_map, get the corresponding DWH values
        furniture_map_reverse = {v: k for k, vals in furniture_map.items() for v in
                                 (vals if isinstance(vals, list) else [vals])}

        furniture_adjustment_matrix = {
            'полностью': {'полностью': 0.95, 'частично': 0.97, 'без мебели': 1},
            'частично': {'полностью': 0.95, 'частично': 0.97, 'без мебели': 1},
            'без мебели': {'полностью': 0.95, 'частично': 0.97, 'без мебели': 1}
        }

        if furniture_value in furniture_map:
            dwh_furniture_values = furniture_map[furniture_value]
            if not isinstance(dwh_furniture_values, list):
                dwh_furniture_values = [dwh_furniture_values]
        else:
            dwh_furniture_values = [furniture_value]

        # Standardizing furniture values before filtering:
        same_floor_df['Furniture'] = same_floor_df['Furniture'].str.lower().str.strip()
        dwh_furniture_values = [value.lower().strip() for value in dwh_furniture_values]

        # Now apply the filtering
        furniture_filtered_df = same_floor_df[same_floor_df['Furniture'].isin(dwh_furniture_values)]
        print(f"After initial filtering by furniture DataFrame shape: {furniture_filtered_df.shape}")

        if furniture_filtered_df.shape[0] < 3:
            # Apply adjustments based on the matrix
            evaluated_furniture_type_standard = furniture_map_reverse.get(furniture_value, furniture_value)
            print(f"Evaluated furniture type (standardized): {evaluated_furniture_type_standard}")

            for index, row in same_floor_df.iterrows():
                current_furniture_standard = furniture_map_reverse.get(row['Furniture'], row['Furniture'])
                print(f"Current furniture type (standardized) for index {index}: {current_furniture_standard}")

                adjustment_factor = furniture_adjustment_matrix.get(evaluated_furniture_type_standard, {}).get(
                    current_furniture_standard, 1)
                print(f"Adjustment factor for index {index}: {adjustment_factor}")

                same_floor_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

            furniture_filtered_df = same_floor_df
            print(f"After adjusting by furniture DataFrame shape: {furniture_filtered_df.shape}")
            if furniture_filtered_df.shape[0] < 3:
                # Fetch all available floor types left after the 5th filter
                all_floors_df = same_condition_df.copy()
                evaluated_floor_type = evaluated_apartment['floor_new']

                # Apply adjustments
                for index, row in all_floors_df.iterrows():
                    try:
                        adjustment_factor = floor_adjustment_matrix[evaluated_floor_type][row['floor_new']]
                    except KeyError:  # If the floor type doesn't exist in the matrix, no adjustment is made
                        adjustment_factor = 1

                    all_floors_df.at[index, 'price_per_1sqrm'] *= adjustment_factor

                furniture_filtered_df = all_floors_df

        if furniture_filtered_df.empty:
            print("No suitable apartments found after furniture filter, fetching more data...")
            return pd.DataFrame()  # Return empty DataFrame
        else:
            return furniture_filtered_df

    def print_descriptive_statistics(self, top_similar_apartments_df):
        if top_similar_apartments_df.empty:
            print("No suitable apartments found for descriptive statistics.")
            return None, None  # Return None if no suitable apartments found

        average_price_per_sqrm = top_similar_apartments_df['price_per_1sqrm'].mean()

        # Create descriptive statistics for price_per_1sqrm
        desc_stats_df = top_similar_apartments_df['price_per_1sqrm'].describe().to_frame().T

        return average_price_per_sqrm, desc_stats_df

    def find_top_similar_apartments_from_local(self, evaluated_apartment_json_file, save_to_excel=True):
        # Set radius to 1000
        radius = 1000

        # Read evaluated_apartment from JSON file
        with open(evaluated_apartment_json_file, 'r', encoding='utf-8-sig') as f:
            evaluated_apartment = json.load(f)
            # Preprocess the floor_new value from the JSON
        evaluated_apartment["floor_new"] = self.preprocess_floor_from_json(evaluated_apartment["floor_new"])

        df = pd.read_csv('local_data.csv')
        print(f"Initial DataFrame shape: {df.shape}")

        df = self.preprocess_dataframe(df)
        if df.empty:
            print("DataFrame is empty. Exiting find_top_similar_apartments_from_local...")
            return

        most_recent_date = pd.to_datetime(df['CreateDate']).max()

        # 1. Filter for the last 30 days
        last_30_days_df = df[df['CreateDate'] >= (most_recent_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')]
        similar_apartments_df = self.filter_dataframe(last_30_days_df, evaluated_apartment, radius, building_types_dict,
                                                      renovation_dict)

        if similar_apartments_df.shape[0] >= 3:
            print("Enough data found in the last 30 days!")
        else:
            print(
                f"Only {similar_apartments_df.shape[0]} similar apartment(s) found from the last 30 days. Proceeding to the next 31 days...")

            # 2. Filter for the last 61 days if not enough apartments from the last 30 days
            last_61_days_df = df[df['CreateDate'] >= (most_recent_date - pd.Timedelta(days=61)).strftime('%Y-%m-%d')]
            similar_apartments_df = self.filter_dataframe(last_61_days_df, evaluated_apartment, radius,
                                                          building_types_dict, renovation_dict)

            if similar_apartments_df.shape[0] >= 3:
                print("Enough data found in the last 61 days!")
            else:
                print(
                    f"Only {similar_apartments_df.shape[0]} similar apartment(s) found from the last 61 days. Proceeding to use all data...")

                # 3. Use all data if still not enough apartments
                similar_apartments_df = self.filter_dataframe(df, evaluated_apartment, radius, building_types_dict,
                                                              renovation_dict)
                if similar_apartments_df.shape[0] < 3:
                    print("The 90 days data wasn't enough for analysis")
                    # Call the alternative filtering method using last_30_days_df
                    similar_apartments_df = self.alternative_filtering(last_30_days_df, evaluated_apartment, radius,
                                                                       building_types_dict, renovation_dict)

                    if not similar_apartments_df.empty:
                        average_price_per_sqrm, desc_stats_df = self.print_descriptive_statistics(similar_apartments_df)

                    else:
                        print("No apartments found even after alternative filtering.")
                        return
                else:
                    average_price_per_sqrm, desc_stats_df = self.print_descriptive_statistics(similar_apartments_df)

                    # Export to Excel (common for both)
                if save_to_excel:
                    try:
                        with pd.ExcelWriter(
                                'C:\\Users\\sarba\\Desktop\\Otbasy\MK\\similar_apartments_analysis.xlsx') as writer:
                            similar_apartments_df.to_excel(writer, sheet_name='Similar Apartments', index=False)
                            desc_stats_df.to_excel(writer, sheet_name='Descriptive Statistics', index=False)
                            print("File saved successfully!")
                    except Exception as e:
                        print("Error while saving file:", str(e))

                return average_price_per_sqrm


import os
os.getcwd()
os.chdir(r'C:\Users\sarba\Desktop\Otbasy\MK')
# Your JSON file path
json_file_path = 'C:\\Users\\sarba\\Desktop\\Otbasy\\MK\\evaluated_apartment.json'
# Initialize RealEstateAnalysis class with your connection string.
connection_string = "DRIVER={SQL Server};SERVER=10.10.2.92;DATABASE=RealEstateData;Trusted_Connection=yes"  # Replace with your actual connection string
real_estate_data = RealEstateData(connection_string)

real_estate_data.find_top_similar_apartments_from_local(json_file_path)



import os
os.getcwd()
df = pd.read_csv('local_data.csv')
recent_date = pd.to_datetime(df['CreateDate']).max()
df = df[df['CreateDate'] >= (recent_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')]

class ApartmentDistanceCalculator:
    def calculate_distance(self, row, evaluated_apartment):
        apartment_loc = [radians(round(row['Latitude'], 4)), radians(round(row['Longitude'], 4))]
        evaluated_apartment_loc = [radians(round(evaluated_apartment['latitude'], 4)), radians(round(evaluated_apartment['longitude'], 4))]
        distance = haversine_distances([apartment_loc, evaluated_apartment_loc])[0,1] * 6371000
        return distance

# Assuming df is your DataFrame
calculator = ApartmentDistanceCalculator()
evaluated_apartment_1 = {
    "latitude": 43.181436,
    "longitude": 76.838718
}
df['distance'] = df.apply(lambda row: calculator.calculate_distance(row, evaluated_apartment_1), axis=1)
df['distance'].describe()

df = df[ (df['distance'] <= 1000)]
df.columns

df = df[df['Rooms'] == 1]
df.shape
df['Square'].describe()

df = df[(df['Year'] >= 2018) & (df['Year'] <= 2028)]
df['Building'].value_counts()

df[df['Building'] == '-']['Renovation'].value_counts()
df = df[df['Building'] == '-']
df.shape
df = df[df['Renovation'] == 'хорошее']



import pandas as pd
class FloorProcessor:
    def calculate_floor_new(self, row):
        floor = row['Floor']
        try:
            n, x = [int(i) for i in floor.split(' из ')]
            if n == x:
                return 3
            elif n == 1:
                return 1
            else:
                return 2
        except ValueError:
            return None

processor = FloorProcessor()

# Apply the function to the DataFrame
df['floor_new'] = df.apply(lambda row: processor.calculate_floor_new(row), axis=1)

df['floor_new']
df = df[df['floor_new'] == 1]
df.shape
df['Furniture'].value_counts()

df['Square'].describe()


