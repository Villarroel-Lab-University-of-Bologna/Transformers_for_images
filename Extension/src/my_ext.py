import knime.extension as knext


main_category = knext.category(
    path="/community/",
    level_id="demo",
    name="Demo Extension",
    description="Python Nodes for Image & Text Processing",
    icon="icons/icon.png"
)

from nodes import vit_learner
