from google import genai
import time
import asyncio
import base64
import cv2
from config.gemini_key import GEMINI_API_KEY

class GeminiUSDSelector:
    def __init__(self):

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Rate limiting variables
        self.last_gemini_call_time = 0
        self.min_call_interval = 1.0  # Minimum seconds between calls
        self.max_retries = 3
        self.retry_delay = 2.0  # Seconds to wait between retries
        
        # Define the schema for Gemini response
        self.image_comparison_schema = {
            "type": "object",
            "properties": {
                "best_match_index": {
                    "type": "integer",
                    "description": "The index (0-4) of the image that best matches the query image"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0 and 1"
                }
            },
            "required": ["best_match_index", "confidence"]
        }
        
        # Define the context text for Gemini
        self.context_text = """
        You are an expert in comparing a real cropped image of a object and 3 reference images. 
        You will be given a query image and 3 reference images.
        Your task is to determine which of the 3 reference images most closely matches the query image.
        The query image and the reference images might be have different poses, lighting, conditions, and background.
        Consider factors like:
        - Overall shape and form
        - Style and design
        - Material appearance
        - Functionality
        - Size and proportions
        
        Return only the index (0-2) of the best matching image and your confidence score.
        """

    def gemini_query(self, query_image, reference_images):
        # Convert images to base64 for Gemini
        def image_to_base64(img):
            _, buffer = cv2.imencode('.jpg', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        # Create Gemini content with images
        contents = [
            self.context_text,
            {
                "role": "user",
                "parts": [
                    {"text": "Query image:"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_to_base64(query_image)}}
                ]
            }
        ]
        
        # Add reference images
        for idx in reversed(range(len(reference_images))):
            contents.append({
                "role": "user",
                "parts": [
                    {"text": f"Reference image {idx}:"},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_to_base64(reference_images[idx])}}
                ]
            })
        
        # Call Gemini API with retry logic
        response = asyncio.run(self.call_gemini_with_retry(contents))
        return response

    async def call_gemini_with_retry(self, contents):
        """Helper function to call Gemini API with retry logic and rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_gemini_call_time
        
        # Enforce minimum time between calls
        if time_since_last_call < self.min_call_interval:
            await asyncio.sleep(self.min_call_interval - time_since_last_call)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash-lite',
                    contents=contents,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': self.image_comparison_schema
                    }
                )
                self.last_gemini_call_time = time.time()
                return response
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries - 1:
                    # self.get_logger().warning(f"Rate limit hit, retrying in {self.retry_delay} seconds...")
                    print(f"Rate limit hit, retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise e