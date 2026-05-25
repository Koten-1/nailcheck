from icrawler.builtin import BingImageCrawler
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from config import RAW_DIR

SEARCH_QUERIES = {
    "healthy": [
        "healthy fingernails close up",
        "normal fingernails macro photo",
        "clean pink fingernails close up",
        "healthy nail bed color",
    ],
    "iron_deficiency": [
        "koilonychia spoon nails iron deficiency",
        "pale nail beds anemia fingernails",
        "spoon shaped nails medical",
        "iron deficiency anemia fingernail signs",
    ],
    "fungal": [
        "onychomycosis fungal nail infection",
        "nail fungus discoloration close up",
        "thickened yellow toenail fungus",
        "nail fungus early stage fingers",
    ],
    "nutrient_deficiency": [
        "leukonychia white spots fingernails",
        "beau lines nails nutrient deficiency",
        "brittle nails zinc deficiency",
        "nail ridges vitamin deficiency close up",
    ],
}

IMAGES_PER_QUERY = 80


def scrape_all():
    for class_name, queries in SEARCH_QUERIES.items():
        for i, query in enumerate(queries):
            # each query gets its own subfolder — no filename clashes
            query_dir = RAW_DIR / class_name / f"batch_{i}"
            query_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[SCRAPING] {class_name} batch_{i} — '{query}'")
            crawler = BingImageCrawler(
                storage={"root_dir": str(query_dir)},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
            )
            crawler.crawl(
                keyword=query,
                max_num=IMAGES_PER_QUERY,
                min_size=(100, 100),
            )
        print(f"[DONE] {class_name}")


if __name__ == "__main__":
    scrape_all()