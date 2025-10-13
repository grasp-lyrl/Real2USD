import asyncio
import usd_search_client
from usd_search_client.rest import ApiException
from pprint import pprint

# from PIL import Image
import cv2
import io, os
import base64
import numpy as np
from ipdb import set_trace as st

"""
author: Christopher D. Hsu
email: chsu8@seas.upenn.edu
created: 11-25-2024
updated: 1-20-2024

usd_search_client is a python package that you can just install into your env:
https://github.com/NVIDIA-Omniverse/usdsearch-client/tree/main
pip install git+https://github.com/NVIDIA-Omniverse/usdsearch-client
"""


class USDSearch:
    def __init__(self):
        # See configuration.py for a list of all supported configuration parameters.
        self.configuration = usd_search_client.Configuration(
            host="http://localhost:30517"
        )
        # The client may configure the authentication and authorization parameters
        # in accordance with the API server security policy.
        # Examples for each auth method are provided below, use the example that
        # satisfies your auth use case.

        # Configure API key authorization: APIKeyHeader
        # self.configuration.api_key['APIKeyHeader'] = os.environ["API_KEY"]

        # Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
        # self.configuration.api_key_prefix['APIKeyHeader'] = 'Bearer'

        # Configure HTTP basic authorization: HTTPBasic
        # self.configuration = usd_search_client.Configuration(
        # username = os.environ["USERNAME"],
        # password = os.environ["PASSWORD"]
        # )

        # Configure Bearer authorization: HTTPBearer
        # self.configuration = usd_search_client.Configuration(
        #     access_token = os.environ["BEARER_TOKEN"]
        # )

    def process_image(self, image, formated=True) -> str:
        """
        Used for image_similarity_search, image must be shrunken and encoded
        Resize image, encode as jpeg to shrink size then convert to b64 for upload
        In: cv2 image
        Out: Base64 encoded image with format prefix (str) or
             Base64 encoded image (str)
        """
        height, width = image.shape[:2]

        # Resize only if the image is larger than 336x336
        if height > 336 or width > 336:
            image_resized = cv2.resize(image, (336, 336))
        else:
            image_resized = image

        # Encode the image as JPEG
        _, buffer = cv2.imencode(".jpg", image_resized)

        # Convert the JPEG buffer to base64
        image_b64 = base64.b64encode(buffer).decode()
        assert (
            len(image_b64) < 180000
        ), "Image too large to upload."  # ensure image is small enough

        if formated:
            return f"data:image/jpeg;base64,{image_b64}"
        else:
            return image_b64

    async def call_search_post_api(self, desciption, image_string, search_path='SimReadyAssets', exclude_file_name='', limit=5):
        """
        description = 'pallet' # str | Conduct text-based searches powered by AI (optional) (default to '')
        image_similarity_search = ['[]']#[image_string] # List[str] | Perform similarity searches based on a list of images (optional)
        file_name = '' # str | Filter results by asset file name, allowing partial matches. Use wildcards: `*` for any number of characters, `?` for a single character. Separate terms with `,` for OR and `;` for AND. (optional) (default to '')
        exclude_file_name = '' # str | Exclude results by asset file name, allowing partial matches. Use wildcards: `*` for any number of characters, `?` for a single character. Separate terms with `,` for OR and `;` for AND. (optional) (default to '')
        file_extension_include = '' # str | Filter results by file extension. Use wildcards: `*` for any number of characters, `?` for a single character. Separate terms with `,` for OR and `;` for AND. (optional) (default to '')
        file_extension_exclude = '' # str | Exclude results by file extension. Use wildcards: `*` for any number of characters, `?` for a single character. Separate terms with `,` for OR and `;` for AND. (optional) (default to '')
        created_after = '' # str | Filter results to only include assets created after a specified date (optional) (default to '')
        created_before = '' # str | Filter results to only include assets created before a specified date (optional) (default to '')
        modified_after = '' # str | Filter results to only include assets modified after a specified date (optional) (default to '')
        modified_before = '' # str | Filter results to only include assets modified before a specified date (optional) (default to '')
        file_size_greater_than = '' # str | Filter results to only include files larger than a specific size (optional) (default to '')
        file_size_less_than = '' # str | Filter results to only include files smaller than a specific size (optional) (default to '')
        created_by = '' # str | Filter results to only include assets created by a specific user. In case AWS S3 bucket is used as a storage backend, this field corresponds to the owner's ID. In case of an Omniverse Nucleus server, this field may depend on the configuration, but typically corresponds to user email. (optional) (default to '')
        exclude_created_by = '' # str | Exclude assets created by a specific user from the results (optional) (default to '')
        modified_by = '' # str | Filter results to only include assets modified by a specific user. In the case, when AWS S3 bucket is used as a storage backend, this field corresponds to the owner's ID. In case of an Omniverse Nucleus server, this field may depend on the configuration, but typically corresponds to user email. (optional) (default to '')
        exclude_modified_by = '' # str | Exclude assets modified by a specific user from the results (optional) (default to '')
        similarity_threshold = 0 # float | Set the similarity threshold for embedding-based searches. This functionality allows filterring duplicates and returning only those results that are different from each other. Assets are considered to be duplicates if the cosine distance betwen the embeddings a smaller than the similarity_threshold value, which could be in the [0, 2] range. (optional)
        cutoff_threshold = 0 # float | Set the cutoff threshold for embedding-based searches (optional)
        search_path = '' # str | Specify the search path within the storage backend. This path should not contain the storage backend URL, just the asset path on the storage backend. Use wildcards: `*` for any number of characters, `?` for a single character. Separate terms with `,` for OR and `;` for AND. (optional) (default to '')
        exclude_search_path = '' # str | Specify the search path within the storage backend. This path should not contain the storage backend URL, just the asset path on the storage backend. Use wildcards: `*` for any number of characters, `?` for a single character. Separate terms with `,` for OR and `;` for AND. (optional) (default to '')
        search_in_scene = '' # str | Conduct the search within a specific scene. Provide the full URL for the asset including the storage backend URL prefix. (optional) (default to '')
        filter_by_properties = '' # str | Filter assets by USD attributes where at least one root prim matches (note: only supported for a subset of attributes indexed). Format: `attribute1=abc,attribute2=456` (optional) (default to '')
        min_bbox_x = 3.4 # float | Filter by minimum X axis dimension of the asset's bounding box (optional)
        min_bbox_y = 3.4 # float | Filter by minimum Y axis dimension of the asset's bounding box (optional)
        min_bbox_z = 3.4 # float | Filter by minimum Z axis dimension of the asset's bounding box (optional)
        max_bbox_x = 3.4 # float | Filter by maximum X axis dimension of the asset's bounding box (optional)
        max_bbox_y = 3.4 # float | Filter by maximum Y axis dimension of the asset's bounding box (optional)
        max_bbox_z = 3.4 # float | Filter by maximum Z axis dimension of the asset's bounding box (optional)
        return_images = False # bool | Return images if set to True (optional) (default to False)
        return_metadata = False # bool | Return metadata if set to True (optional) (default to False)
        return_root_prims = False # bool | Return root prims if set to True (optional) (default to False)
        return_default_prims = False # bool | Return default prims if set to True (optional) (default to False)
        return_predictions = False # bool | Return predictions if set to True (optional) (default to False)
        return_in_scene_instances_prims = False # bool | [in-scene search only] Return prims of instances of objects found in the scene (optional) (default to False)
        embedding_knn_search_method = usd_search_client.SearchMethod("approximate") # SearchMethod | Search method, approximate should be faster but is less accurate. Default is exact (optional)
        limit = 56 # int | Set the maximum number of results to return from the search, default is 32 (optional)
        vision_metadata = 'vision_metadata_example' # str | Uses a keyword match query on metadata fields that were generated using Vision Language Models (optional)
        return_vision_generated_metadata = False # bool | Returns the metadata fields that were generated using Vision Language Models (optional) (default to False)
        """

        # Enter a context with an instance of the API client
        async with usd_search_client.ApiClient(self.configuration) as api_client:
            # Create an instance of the API class
            api_instance = usd_search_client.AISearchApi(api_client)
            deep_search_request = usd_search_client.DeepSearchSearchRequest(
                description=desciption,
                image_similarity_search=image_string,
                search_path=search_path,
                # exclude_search_path="*/Textures/*",
                exclude_file_name=exclude_file_name,
                file_extension_include="usd",
                # file_extension_exclude="png",
                limit=limit,
                return_images=True,
            )
            try:
                # Search Post
                # i added the asyncio.wait_for to set a timeout
                api_response = await asyncio.wait_for(
                    api_instance.search_post_v2_deepsearch_search_post(
                        deep_search_request
                    ),
                    timeout=1.0,
                )
                url = []
                scores = []
                images = []
                for item in api_response:
                    url.append(item.url)
                    scores.append(item.score)

                    # # decode b64 image
                    image = base64.b64decode(item.image)
                    image = np.frombuffer(image, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)


                return url, scores, images

            except Exception as e:
                print(
                    "Exception when calling AISearchApi->search_post_v2_deepsearch_search_post: ",
                    e,
                )
                return None, None, None


if __name__ == "__main__":
    usd_search = USDSearch()
    image_str = "/data/tests/seg-chair.png"
    image_str1 = "/data/tests/seg-chair.png"
    image_string = usd_search.process_image(image_str)
    url, scores = asyncio.run(
        usd_search.call_search_post_api("", [image_string])
    )
    print("image search", url, scores)

    # url, scores = asyncio.run(
    #     usd_search.call_search_post_api("Barrel", "")
    # )
    # print("text search", url, scores)
