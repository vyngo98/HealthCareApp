# pip install langfuse 'smolagents[telemetry]' openinference-instrumentation-smolagents datasets 'smolagents[gradio]' gradio --upgrade

import os
# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-594082a2-a09f-4a49-ae20-a8df66afc79f"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-e453aeee-8757-4fb4-b78a-f3f8b9736bc0"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com" # 🇪🇺 EU region
# os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com" # 🇺🇸 US region

os.environ["HF_TOKEN"] = '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'

from langfuse import get_client

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")