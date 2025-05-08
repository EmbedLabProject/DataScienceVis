import requests

def get_address_from_latlong(lat, lon, api_key):
    url = "https://api.longdo.com/map/services/address"
    params = {
        "key": api_key,
        "lat": lat,
        "lon": lon
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # extract info
        district = data.get('district', '').replace('เขต', '').strip()
        subdistrict = data.get('subdistrict', '').replace('แขวง', '').replace('ตำบล', '').strip()

        return district, subdistrict

    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None, None

# api_key = "dad93076eb9a905a9122a00806c70616"
# lat = 13.72427
# lon = 100.53726

# district, subdistrict = get_address_from_latlong(lat, lon, api_key)
# print("District:", district)
# print("Subdistrict:", subdistrict)
