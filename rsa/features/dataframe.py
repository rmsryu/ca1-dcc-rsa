from ..constants import enums, dictionaries
from pandas import Series

# Set total casutalties
def set_total_casualties(row :Series):
    """Calculate total number of casualties

    Args:
        row (Series): A record from the RSA casualties dataset

    Returns:
        int: total number of causalties
    """    
    
    return row[enums.feature_names.NO_FATAL] + row[enums.feature_names.NO_SERIOUS] + row[enums.feature_names.NO_MINOR]


# Calculate dependant variable:
def set_calculated_outcome(row :Series):
    """Calculate outcome

    Args:
        row (Series): A record from the RSA casualties dataset

    Returns:
        int: enums.outcome
    """    

    if row[enums.feature_names.NO_FATAL] > 0:
        return enums.outcome.FATAL
    if row[enums.feature_names.NO_SERIOUS] > 0:
        return enums.outcome.SERIOUS
    if row[enums.feature_names.NO_MINOR] > 0:
        return enums.outcome.MINOR
    if row[enums.feature_names.NO_NOTINJURED] > 0:
        return enums.outcome.NOT_INJURED
    if row[enums.feature_names.NO_UNKONWN] > 0:
        return enums.outcome.UNKNOWN


# Calculate severity:
def set_severity(row :Series):
    """Calculate outcome

    Args:
        row (Series): A record from the RSA casualties dataset

    Returns:
        int: enums.severity
    """    
    high = [enums.outcome.FATAL,enums.outcome.SERIOUS]
    low = [enums.outcome.MINOR, enums.outcome.NOT_INJURED]

    if row[enums.feature_names.OUTCOME_CALCULATED] in high :
        return enums.severity.HIGH
    elif row[enums.feature_names.OUTCOME_CALCULATED] in low :
        return enums.severity.LOW
    else:
        return None

def map_unknown_vehicle_type(vehicle_type):
    """Map unknown vehicle type to Other 

    Args:
        vehicle_type (string): Value of the feature vehicle_type on the rsa dataset

    Returns:
        string: Mapped vehicle type
    """    
    if vehicle_type in dictionaries.VEHICLE_TYPES:
        return vehicle_type
    else:
        return enums.vehicle_type.OTHER

def convert_df_labels(df):
    '''
    Convert rsa dataset to category values

    Args:
        df (_type_): rsa dataset

    Returns:
        DataFrame: Converted Dataframe
    '''
    df_return = df.copy(deep=True)
    columns = df_return.columns
    if enums.feature_names.WEEKDAY in columns:
        df_return.weekday = df_return.weekday.apply(lambda x: f"{x} - {dictionaries.DAY_OF_WEEK.get(x)}")
    if enums.feature_names.HOUR in columns:
        df_return.hour = df_return.hour.apply(lambda x: dictionaries.HOURS.get(x))
    if enums.feature_names.COUNTY in columns:
        df_return.county = df_return.county.apply(lambda x: dictionaries.COUNTIES.get(x))
    if enums.feature_names.OUTCOME in columns:
        df_return.outcome = df_return.outcome.apply(lambda x: dictionaries.OUTCOME_TYPES.get(x))
    if enums.feature_names.OUTCOME_CALCULATED in columns:
        df_return.outcome_calculated = df_return.outcome_calculated.apply(lambda x: dictionaries.OUTCOME_TYPES.get(x))
    if enums.feature_names.VEHICLE_TYPE in columns:
        df_return.vehicle_type = df_return.vehicle_type.apply(lambda x: dictionaries.VEHICLE_TYPES.get(x))
        df_return.vehicle_type.fillna("Other", inplace=True)
    if enums.feature_names.CIRCUMSTANCES in columns:
        df_return.circumstances = df_return.circumstances.apply(lambda x: dictionaries.CIRCUMSTANCE_TYPES.get(x))
        df_return.circumstances.fillna("Other", inplace=True)
    if enums.feature_names.AGE in columns:
        df_return.age = df_return.age.apply(lambda x: dictionaries.AGE_GROUPS.get(x))
    if enums.feature_names.GENDER in columns:
        df_return.gender = df_return.gender.apply(lambda x: dictionaries.GENDER.get(x))
    if enums.feature_names.SEVERITY in columns:
        df_return.severity = df_return.severity.apply(lambda x: dictionaries.SEVERITY.get(x))
    if enums.feature_names.FBS_STATION in columns:
        df_return.severity = df_return.severity.apply(lambda x: dictionaries.FIREBRIGADE_STATION.get(x))
        
    return df_return

__all__ = ['convert_df_labels', 'set_total_casualties', 'set_calculated_outcome', 'set_severity', 'map_unknown_vehicle_type']