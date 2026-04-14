import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load .env file
load_dotenv()

SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
POSTGRES_URI  = os.getenv("POSTGRES_URI")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_KEY. "
        "Check your .env file locally, or 'Repository Secrets' if running on GitHub Actions."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
