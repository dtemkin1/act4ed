"""
Script to grab relevant GTFS and LODES OD pairs for transit analysis.

Raises:
    ValueError: If TRANSITLAND_API_KEY environment variable is not set.
    ValueError: If no feed versions are found for the specified feed key.
"""

import pathlib
import os

import pandas
import requests

from pygris.data import get_lodes
from dotenv import load_dotenv

load_dotenv()

STATE = "MA"
FEED_ID = "f-drt0-metrowestregionaltransitauthority"

TRANSITLAND_API_KEY = os.getenv("TRANSITLAND_API_KEY")
BASE_API_URL = "https://transit.land/api/v2/rest"
FOLDER = pathlib.Path(__file__).parent.resolve() / "data"


def main():
    """
    Runs all other functions to download relevant data.
    """
    print("Grabbing GTFS data...")
    scrape_gtfs(FEED_ID)

    print("Grabbing archived GTFS data...")
    scrape_gtfs(FEED_ID, latest=False)

    print("Grabbing LODES data from latest year available...")
    scrape_lodes(STATE)

    print("Grabbing LODES data from 2021...")
    scrape_lodes(STATE, year=2021)

    print("All done!")


def scrape_gtfs(feed_key: str, latest: bool = True):
    """
    Grabs GTFS zip file and downloads it to data folder. Relies on transit.land feed keys.

    Args:
        feed_key (str): transit.land feed lookup key.
        latest (bool, optional): Whether to download the latest feed version. Defaults to True.

    Raises:
        ValueError: If TRANSITLAND_API_KEY environment variable is not set.
        ValueError: If no feed versions are found for the specified feed key.
    """

    if not TRANSITLAND_API_KEY:
        raise ValueError("TRANSITLAND_API_KEY environment variable not set.")

    feed_url: str
    if latest:
        feed_url = f"{BASE_API_URL}/feeds/{feed_key}/download_latest_feed_version"
    else:
        # Get the feed versions to find an archived one
        versions_url = f"{BASE_API_URL}/feeds/{feed_key}/feed_versions"
        versions_response = requests.get(
            versions_url, headers={"apikey": f"{TRANSITLAND_API_KEY}"}, timeout=10
        )
        versions_response.raise_for_status()
        versions_data = versions_response.json()

        # Get the oldest feed version
        feed_versions = versions_data.get("feed_versions", [])
        if not feed_versions:
            raise ValueError("No feed versions found for the specified feed key.")

        if len(feed_versions) < 12:
            feed_version_key = feed_versions[-1].get("sha1", "")
        else:
            feed_version_key = feed_versions[12].get("sha1", "")

        # Get the archived feed version download URL using the feed version key
        feed_url = f"{BASE_API_URL}/feed_versions/{feed_version_key}/download"

    response = requests.get(
        feed_url, headers={"apikey": f"{TRANSITLAND_API_KEY}"}, timeout=10
    )
    response.raise_for_status()

    with open(
        FOLDER / f"gtfs_{feed_key}_{"latest" if latest else "archived"}.zip", "wb"
    ) as f:
        f.write(response.content)


def scrape_lodes(state: str, year: int = 2022):
    """
    Scrapes LODES OD pair data from Census and saves it as a (very large) csv.

    .. TIP::
        Check out https://lehd.ces.census.gov/data/ for more information on LODES data.

    Args:
        state (str): The state to scrape data for.
        year (int, optional): The year to scrape data for. Defaults to 2022.
    """
    lodes_od = get_lodes(state=state, year=year, lodes_type="od")
    assert isinstance(lodes_od, pandas.DataFrame)

    with open(
        FOLDER / f"lodes_od_{state.lower()}_{year}.csv", "w", encoding="utf-8"
    ) as f:
        f.write(lodes_od.to_csv(encoding="utf-8"))


if __name__ == "__main__":
    main()
