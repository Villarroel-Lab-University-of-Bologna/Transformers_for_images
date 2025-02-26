import requests
import knime.extension as knext
import pandas as pd
import logging
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Define sub-category
image_category = knext.category(
    path="/community/vit_ft",
    level_id="flickrimg",
    name="Flickr Image Downloader",
    description="Node for downloading image URLs from Flickr",
    icon="icons/icon.png",
)


@knext.node(
    name="Flickr Image Downloader",
    node_type=knext.NodeType.SOURCE,
    icon_path="icons/icon.png",
    category=image_category,
    id="img-downloader",
)
@knext.output_table(
    name="Fetched Images",
    description="Table containing the URLs of the images downloaded from Flickr.",
)
class FlickrImageDownloader:
    """
    Node for downloading images from Flickr using a search term, an API key, and the desired number of images.
    """

    credential_param = knext.StringParameter(
        label="Flickr API Key",
        description="Choose one of the connected credentials (Pass key through credential config password field)",
        choices=lambda a: knext.DialogCreationContext.get_credential_names(a),
    )

    search_term = knext.StringParameter(
        label="Search Term",
        description="Search term for the images to download.",
        default_value="",
    )

    # The maximum nuumber of images that can be downloaded for every page is 500
    no_images = knext.IntParameter(
        label="Number of Images",
        description="Number of images to retrieve from Flickr.",
        default_value=10,
        min_value=1,
        max_value=20000,
    )

    def configure(self, ctx: knext.ConfigurationContext):
        if not ctx.get_credential_names():
            raise knext.InvalidParametersError("No credentials provided.")
        if not self.credential_param:
            raise knext.InvalidParametersError("Credentials not selected.")
        return knext.Column(name="Image", ktype=knext.logical(Image.Image))

    def execute(self, ctx: knext.ExecutionContext):
        # Fetch credentials from KNIME node
        credentials = ctx.get_credentials(self.credential_param)
        self.api_key = (
            credentials.password
        )  # Use the password field to store the API key
        base_url = "https://www.flickr.com/services/rest/"

        max_per_page = 500  # Flickr's per-page limit
        total_images = self.no_images  # User-requested total images
        urls = []
        page = 1

        while len(urls) < total_images:
            remaining_images = total_images - len(urls)
            per_page = min(remaining_images, max_per_page)  # Limit per request

            params = {
                "method": "flickr.photos.search",
                "api_key": self.api_key,
                "text": self.search_term,
                "per_page": per_page,
                "page": page,  # Update page number
                "format": "json",
                "nojsoncallback": 1,
            }

            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch data from Flickr API: {response.text}"
                )

            data = response.json()

            if "photos" not in data or "photo" not in data["photos"]:
                raise ValueError("Unexpected API response format.")

            photo_list = data["photos"]["photo"]

            if not photo_list:
                LOGGER.warning("No more images available. Stopping early.")
                break  # Stop if no more images available

            for photo in photo_list:
                # Validate metadata
                if (
                    "farm" not in photo
                    or photo["farm"] == 0
                    or "server" not in photo
                    or not photo["server"]
                    or "id" not in photo
                    or not photo["id"]
                    or "secret" not in photo
                    or not photo["secret"]
                ):
                    LOGGER.warning(f"Skipping invalid photo: {photo}")
                    continue  # Skip invalid entries

                photo_url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"

                if photo_url not in urls:  # Prevent duplicates
                    urls.append(photo_url)

                if len(urls) >= total_images:
                    break  # Stop if we have enough images

            page += 1  # Move to the next page

        LOGGER.info(f"Retrieved {len(urls)} unique image URLs from Flickr.")

        # Create a DataFrame with the image URLs
        result_df = pd.DataFrame()
        result_df["Image"] = [self.__open_image_from_url(i) for i in urls]

        # Report progress
        ctx.set_progress(0.9, "Image retrieval complete.")
        return knext.Table.from_pandas(result_df)

    def __open_image_from_url(self, url):
        import io

        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            return Image.open(buffer)
        else:
            raise ValueError(f"Failed to fetch image from URL: {url}")
