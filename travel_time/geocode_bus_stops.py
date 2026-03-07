from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.location import Location
import time

# Read the CSV file
script_path = Path(__file__).parent
df = pd.read_csv(script_path / "./data/stops.csv")

# Initialize the geocoder
geolocator = Nominatim(user_agent="framingham_bus_stops")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


# Function to geocode addresses
def get_coordinates(address: str) -> pd.Series:
    # try to geocode as if bus stop name is address
    coords = pd.Series({"lat": None, "lon": None})
    try:
        coords = get_address_coordinates(address)
    except Exception as e1:
        print(f"Error geocoding address {address}: {e1}")
        # try to geocode as if bus stop name is an intersection
        try:
            street_names = address.split(" & ")
            if len(street_names) == 2:
                coords = get_intersection_coordinates(street_names[0], street_names[1])
        except Exception as e2:
            print(f"Error geocoding intersection {address}: {e2}")
            try:
                street_names = address.split(" @ ")
                if len(street_names) == 2:
                    coords = get_intersection_coordinates(
                        street_names[0], street_names[1]
                    )
            except Exception as e3:
                print(f"Error geocoding intersection {address}: {e3}")
                pass

    return coords


def get_address_coordinates(address: str):
    # Add "Framingham, Massachusetts" to improve geocoding accuracy
    full_address = f"{address.strip()}, Framingham, Massachusetts"
    location: Location = geocode(full_address)
    if location:
        print(f"Geocoded: {address} -> ({location.latitude}, {location.longitude})")
        return pd.Series({"lat": location.latitude, "lon": location.longitude})
    else:
        raise ValueError(f"Could not geocode: {address}")


def get_intersection_coordinates(street_1: str, street_2: str):
    # Add "Framingham, Massachusetts" to improve geocoding accuracy
    street_1 = street_1.strip()
    street_2 = street_2.strip()
    location_1: Location = geocode(f"{street_1}, Framingham, Massachusetts")
    location_2: Location = geocode(f"{street_2}, Framingham, Massachusetts")
    if location_1 and location_2:
        # get street line between the two locations and return the midpoint as the intersection coordinates
        midpoint_lat = (location_1.latitude + location_2.latitude) / 2
        midpoint_lon = (location_1.longitude + location_2.longitude) / 2
        print(f"Geocoded: {street_1} & {street_2} -> ({midpoint_lat}, {midpoint_lon})")
        return pd.Series({"lat": midpoint_lat, "lon": midpoint_lon})
    else:
        raise ValueError(f"Could not geocode: {street_1} & {street_2}")


# Apply geocoding to each address in the 'id' column
print("Starting geocoding process...")
df[["lat", "lon"]] = df["id"].apply(get_coordinates)

# Display results
print("\nGeocoding complete!")
print(f"Successfully geocoded: {df['lat'].notna().sum()} out of {len(df)} stops")
print("\nFirst few rows:")
print(df.head())

# Save the updated dataframe
df.to_csv(script_path / "./data/stops-new.csv", index=False)
print("\nUpdated data saved to ./data/stops-new.csv")
