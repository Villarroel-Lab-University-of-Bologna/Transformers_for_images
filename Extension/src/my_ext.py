import knime.extension as knext


main_category = knext.category(
    path="/community/",
    level_id="vit_ft",
    name="Vision Transformers and Flickr Images Retrieval",
    description="Nodes for Vision Transformers and Flickr Images Retrieval",
    icon="icons/icon.png",
)

from nodes import vit
from nodes import flickr_image_downloader