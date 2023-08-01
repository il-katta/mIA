import sqlite3
from document_loaders import ThreadedUnstructuredURLLoader
import json
import config

with sqlite3.connect(str(config.DATA_DIR / 'places.sqlite')) as conn:
    cur = conn.cursor()
    res = cur.execute("""
        SELECT url
        FROM moz_bookmarks as mb
        LEFT JOIN moz_places as mp ON mb.fk = mp.id
        WHERE type = 1
        and url is not null
        and url not like 'javascript:%'
        and url not like 'place:%'
        and url not like 'view-source:%'
        and url not like 'about:%'
        and url not like 'http%.lan/%'
        and url not like 'http%.lan:%'
        and url not like 'http%.local:%'
        and url not like 'http%.local/%'
        GROUP BY url
    """)

    urls = set()
    for (b_url,) in res:
        urls.add(b_url)
    cur.close()

u_loader = ThreadedUnstructuredURLLoader(
    urls=list(urls),
    max_workers=100,
    mode="single",
    strategy="fast",
    show_progress_bar=True
)

docs = u_loader.load()

with open(config.DATA_DIR / 'bookmarks.json', "w") as f:
    json.dump([dict(doc) for doc in docs], f)
