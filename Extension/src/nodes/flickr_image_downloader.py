import requests
import knime.extension as knext
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

# Define sub-category
image_category=knext.category(
        path="/community/vit_ft",
        level_id="flickrimg",
        name="Flickr Image Downloader",
        description="Node for downloading image URLs from Flickr",
        icon = "icons/icon.png"
    )


@knext.node(
    name="Flickr Image Downloader",
    node_type=knext.NodeType.SOURCE,
    icon_path="icons/icon.png",
    category=image_category,
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

    credential_param = knext.StringParameter(
        label="Flickr API Key",
        description="Choose one of the connected credentials (Pass key through credential config password field)",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a)
    )
    
    search_term = knext.StringParameter(
        label="Search Term",
        description="Search term for the images to download.",
        default_value="",
    )

#The maximum nuumber of images that can be downloaded for every page is 500
    no_images = knext.IntParameter(
        label="Number of Images",
        description="Number of images to retrieve from Flickr.",
        default_value=10,
        min_value=1,
        max_value=500,
    )


    def configure(self, ctx: knext.ConfigurationContext):
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("No credentials provided.")
        if not self.credential_param:
            raise knext.InvalidParametersError("Credentials not selected.")
        return knext.Schema.from_columns([
            knext.Column(name="Image URL", ktype=knext.string())
        ])


    def execute(self, ctx: knext.ExecutionContext):
        # Fetch credentials from KNIME node
        credentials = ctx.get_credentials(self.credential_param)
        self.api_key = credentials.password # Use the password field to store the API key
        base_url = "https://www.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",   # Method to search for photos
            "api_key": self.api_key,            # API key
            "text": self.search_term,           # Search term
            "per_page": self.no_images,         # Number of images to retrieve
            "format": "json",                   # Format of the response
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
        ctx.set_progress(1.0, "Image retrieval complete.")
        return knext.Table.from_pandas(result_df)