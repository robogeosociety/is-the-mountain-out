import requests
import tomllib
from pathlib import Path


def generate_map():
    # Load config
    with open("mountain.toml", "rb") as f:
        config = tomllib.load(f)

    # Load key
    with open("mapbox.key", "r") as f:
        token = f.read().strip()

    # Coordinates
    mtn = config["mountain"]
    cam = config["webcam"]
    weather = config["weather"]

    # Marker colors: Rainier (Red), Webcam (Blue), Weather (Green)
    # Using Maki icons as reliable high-quality substitutes for the requested emojis
    # mountain -> 🏔️, cinema -> 🎥, airport -> 🛬
    markers = [
        f"pin-s-mountain+f44336({mtn['longitude']},{mtn['latitude']})",
        f"pin-s-cinema+2196f3({cam['longitude']},{cam['latitude']})",
        f"pin-s-airport+4caf50({weather['longitude']},{weather['latitude']})",
    ]

    overlay = ",".join(markers)
    style = "mapbox/outdoors-v12"
    width, height = 800, 600

    # Calculate bounds to add padding (Mapbox 'auto' is often too tight)
    # Center is approximately halfway between the furthest points
    avg_lon = (mtn["longitude"] + cam["longitude"]) / 2
    avg_lat = (mtn["latitude"] + cam["latitude"]) / 2

    # Zoom 8 provides a better balance for the Puget Sound region coverage
    zoom = 8

    url = f"https://api.mapbox.com/styles/v1/{style}/static/{overlay}/{avg_lon},{avg_lat},{zoom}/{width}x{height}@2x?access_token={token}"

    print(f"Requesting map from Mapbox (Zoom: {zoom})...")
    headers = {"Referer": "https://tommyroar.github.io"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        Path("assets").mkdir(exist_ok=True)
        with open("assets/map.png", "wb") as f:
            f.write(response.content)
        print("Map successfully saved to assets/map.png")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    generate_map()
