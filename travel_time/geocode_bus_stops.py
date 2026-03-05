import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Read the CSV file
df = pd.read_csv("./data/stops.csv")

# Initialize the geocoder
geolocator = Nominatim(user_agent="framingham_bus_stops")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


# Function to geocode addresses
def get_coordinates(address):
    try:
        # Add "Framingham, Massachusetts" to improve geocoding accuracy
        full_address = f"{address}, Framingham, Massachusetts"
        location = geocode(full_address)
        if location:
            print(f"Geocoded: {address} -> ({location.latitude}, {location.longitude})")
            return pd.Series({"lat": location.latitude, "lon": location.longitude})
        else:
            print(f"Could not geocode: {address}")
            return pd.Series({"lat": None, "lon": None})
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        return pd.Series({"lat": None, "lon": None})


# Apply geocoding to each address in the 'id' column
print("Starting geocoding process...")
df[["lat", "lon"]] = df["id"].apply(get_coordinates)

# Display results
print("\nGeocoding complete!")
print(f"Successfully geocoded: {df['lat'].notna().sum()} out of {len(df)} stops")
print("\nFirst few rows:")
print(df.head())

# Save the updated dataframe
df.to_csv("./data/stops-new.csv", index=False)
print("\nUpdated data saved to ./data/stops-new.csv")
