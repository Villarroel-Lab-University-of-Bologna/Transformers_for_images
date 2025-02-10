<<<<<<< Updated upstream
import requests
import knime.extension as knext
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Flickr Image Downloader",
    node_type=knext.NodeType.SOURCE,
    category=knext.category(
        path="/community/demo",
        level_id="demoimg",
        name="Flickr Image Downloader",
        description="Node for downloading image URLs from Flickr",
        icon = "icons/icon.png"
    ),
    id="img-downloader",
)
@knext.output_table(
    name="Image URLs",
    description="Table containing the URLs of the images downloaded from Flickr."
)
class FlickrImageDownloader:
    """
    Node for downloading images from Flickr using a search term, an API key, and the desired number of images.
    """

    api_key = knext.StringParameter(
        label="API Key",
        description="Your Flickr API key to authenticate requests.",
        default_value="",
    )

    search_term = knext.StringParameter(
        label="Search Term",
        description="Search term for the images to download.",
        default_value="",
    )

    no_images = knext.IntParameter(
        label="Number of Images",
        description="Number of images to retrieve from Flickr.",
        default_value=10,
        min_value=1,
        max_value=500,
    )

    def configure(self):
        return knext.Schema.from_columns([
            knext.Column(name="Image URL", ktype=knext.string())
        ])

    def execute(self, exec_context):
        base_url = "https://www.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",
            "api_key": self.api_key,
            "text": self.search_term,
            "per_page": self.no_images,
            "format": "json",
            "nojsoncallback": 1,
        }

        LOGGER.info("Sending request to Flickr API...")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data from Flickr API: {response.text}")

        data = response.json()

        if "photos" not in data or "photo" not in data["photos"]:
            raise ValueError("Unexpected API response format.")

        photo_list = data["photos"]["photo"]
        
        # Construct image URLs
        urls = []
        for photo in photo_list:
            photo_url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
            urls.append(photo_url)

        LOGGER.info(f"Retrieved {len(urls)} image URLs from Flickr.")

        # Create a DataFrame with the image URLs
        result_df = pd.DataFrame({"Image URL": urls})

        # Report progress
        exec_context.set_progress(1.0, "Image retrieval complete.")

=======
import requests
import knime.extension as knext
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Flickr Image Downloader",
    node_type=knext.NodeType.SOURCE,
    category=knext.category(
        path="/community/demo",
        level_id="demoimg",
        name="Flickr Image Downloader",
        description="Node for downloading image URLs from Flickr",
        icon = "icons/icon.png"
    ),
    id="img-downloader",
)
@knext.output_table(
    name="Image URLs",
    description="Table containing the URLs of the images downloaded from Flickr."
)
class FlickrImageDownloader:
    """
    Node for downloading images from Flickr using a search term, an API key, and the desired number of images.
    """

    api_key = knext.StringParameter(
        label="API Key",
        description="Your Flickr API key to authenticate requests.",
        default_value="",
    )

    search_term = knext.StringParameter(
        label="Search Term",
        description="Search term for the images to download.",
        default_value="",
    )

    no_images = knext.IntParameter(
        label="Number of Images",
        description="Number of images to retrieve from Flickr.",
        default_value=10,
        min_value=1,
        max_value=500,
    )

    def configure(self):
        return knext.Schema.from_columns([
            knext.Column(name="Image URL", ktype=knext.string())
        ])

    def execute(self, exec_context):
        base_url = "https://www.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",
            "api_key": self.api_key,
            "text": self.search_term,
            "per_page": self.no_images,
            "format": "json",
            "nojsoncallback": 1,
        }

        LOGGER.info("Sending request to Flickr API...")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data from Flickr API: {response.text}")

        data = response.json()

        if "photos" not in data or "photo" not in data["photos"]:
            raise ValueError("Unexpected API response format.")

        photo_list = data["photos"]["photo"]
        
        # Construct image URLs
        urls = []
        for photo in photo_list:
            photo_url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
            urls.append(photo_url)

        LOGGER.info(f"Retrieved {len(urls)} image URLs from Flickr.")

        # Create a DataFrame with the image URLs
        result_df = pd.DataFrame({"Image URL": urls})

        # Report progress
        exec_context.set_progress(1.0, "Image retrieval complete.")

>>>>>>> Stashed changes
        return knext.Table.from_pandas(result_df)