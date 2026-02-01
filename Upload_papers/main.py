
from supabase import create_client
import os
from Preprocessing.chunk_and_embed import Document_processing

# Create the supabase client. Make sure to export the env variables.
supabase_client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)


# The folder where the docuemnts to be processed are stored
docs_dir = "documents"
docs_paths = list(map(lambda x: os.path.join(docs_dir, x), os.listdir(docs_dir)))

doc_process = Document_processing(docs_paths, supabase_client)
# doc_process.process_and_store_docs()