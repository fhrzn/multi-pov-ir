"""
Task 2: Map instance tags to 6 categories:
  tower, bridge, palace_castle, arch_gate, monument_statue, other

Each unique instance tag from the resolved dataset is explicitly mapped.
"""

import argparse
import pandas as pd
import ast
import json
from pathlib import Path

_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "phd-research" / "dataset" / "geotir"
_DEFAULT_INPUT = _DEFAULT_DIR / "geotir_jp_gldv_wiki_resolved.csv"
_DEFAULT_OUTPUT = _DEFAULT_DIR / "geotir_jp_gldv_wiki_mapped.csv"

# ============================================================
# EXPLICIT TAG -> CATEGORY MAPPING
# Categories: tower, bridge, palace_castle, arch_gate, monument_statue, other
# ============================================================

TAG_TO_CATEGORY = {
    # ===================== TOWER =====================
    "tower": "tower",
    "tower block": "tower",
    "twin towers": "tower",
    "clock tower": "tower",
    "communication tower": "tower",
    "television tower": "tower",
    "transmitter mast": "tower",
    "observation tower": "tower",
    "lattice tower": "tower",
    "elevator test tower": "tower",
    "water tower": "tower",
    "residential tower": "tower",
    "skyscraper": "tower",
    "skyscraper complex": "tower",
    "hyperboloid structure": "tower",  # typically tower-shaped structures
    "lighthouse": "tower",  # lighthouse is a type of tower
    "headframe": "tower",  # mining tower structure

    # ===================== BRIDGE =====================
    "bridge": "bridge",
    "arch bridge": "bridge",
    "stone arch bridge": "bridge",
    "concrete arch bridge": "bridge",
    "beam bridge": "bridge",
    "girder bridge": "bridge",
    "truss bridge": "bridge",
    "suspension bridge": "bridge",
    "self-anchored suspension bridge": "bridge",
    "cable-stayed bridge": "bridge",
    "cantilever bridge": "bridge",
    "bascule bridge": "bridge",
    "moveable bridge": "bridge",
    "vertical-lift bridge": "bridge",
    "pontoon bridge": "bridge",
    "wooden bridge": "bridge",
    "steel bridge": "bridge",
    "footbridge": "bridge",
    "road bridge": "bridge",
    "railway bridge": "bridge",
    "cross-road bridge": "bridge",
    "cross-sea bridge": "bridge",
    "pipeline bridge": "bridge",
    "road-rail bridge": "bridge",
    "double-decker bridge": "bridge",
    "multi-way bridge": "bridge",
    "toll bridge": "bridge",
    "rigid-frame bridge": "bridge",
    "twin bridges": "bridge",
    "spiral bridge": "bridge",
    "spectacles bridge": "bridge",  # Megane-bashi - famous stone arch bridge
    "aqueduct": "bridge",
    "navigable aqueduct": "bridge",
    "railway viaduct": "bridge",

    # ===================== PALACE / CASTLE =====================
    "palace": "palace_castle",
    "royal palace": "palace_castle",
    "castle": "palace_castle",
    "Japanese castle": "palace_castle",
    "star fort": "palace_castle",
    "fortress": "palace_castle",
    "gusuku": "palace_castle",  # Okinawan castle
    "hirajiro": "palace_castle",  # flatland Japanese castle
    "yamajiro": "palace_castle",  # mountain Japanese castle
    "tenshu": "palace_castle",  # castle tower/keep
    "castle park": "palace_castle",
    "water castle": "palace_castle",
    "ancient mountain fortress in Japan": "palace_castle",
    "Jōsaku castle": "palace_castle",
    "detached palace": "palace_castle",
    "harem": "palace_castle",  # part of palace complex
    "official residence": "palace_castle",  # government residences / palaces
    "royal villa": "palace_castle",
    "Japanese Imperial Facilities": "palace_castle",

    # ===================== ARCH / GATE =====================
    "gate": "arch_gate",
    "mon": "arch_gate",  # Japanese gate
    "sanmon": "arch_gate",  # temple gate
    "sōmon": "arch_gate",  # outer gate
    "Niōmon": "arch_gate",  # temple guardian gate
    "Kōraimon": "arch_gate",  # Korean-style gate
    "karamete-mon": "arch_gate",  # rear gate of a castle
    "Hanzō-mon": "arch_gate",  # specific gate name but instance is gate type
    "natural arch": "arch_gate",
    "sandō": "arch_gate",  # approach road with torii gates

    # ===================== MONUMENT / STATUE =====================
    "monument": "monument_statue",
    "statue": "monument_statue",
    "colossal statue": "monument_statue",
    "Buddhist statue in Tōdai-ji": "monument_statue",
    "stone Buddha statue": "monument_statue",
    "daibutsu": "monument_statue",  # great Buddha statue
    "stele": "monument_statue",
    "obelisk": "monument_statue",
    "cenotaph": "monument_statue",
    "war memorial": "monument_statue",
    "memorial": "monument_statue",
    "tomb of the unknown soldier": "monument_statue",
    "sculpture": "monument_statue",
    "sculpture series": "monument_statue",
    "group of sculptures": "monument_statue",
    "work of art": "monument_statue",
    "sand sculpture": "monument_statue",
    "kubizuka": "monument_statue",  # burial mound for heads, memorial
    "nose tomb": "monument_statue",  # Mimizuka - war memorial mound
    "magaibutsu": "monument_statue",  # rock-carved Buddha
    "Kuyō-tō": "monument_statue",  # memorial/memorial tower
    "fountain": "monument_statue",  # decorative monument
    "floral clock": "monument_statue",  # public monument
    "meridian marker": "monument_statue",  # geographic monument
    "meoto iwa": "monument_statue",  # sacred rocks, spiritual monument
    "fixed construction": "monument_statue",  # e.g. flood memorial stone monument

    # ===================== OTHER (everything else) =====================
    # The rest are explicitly mapped to "other" below
}

# All remaining tags go to "other". We list them explicitly for documentation.
OTHER_TAGS = [
    # --- Nature / Geography ---
    "48 waterfalls", "V-shaped valley", "active volcano", "alluvial plain",
    "archipelago", "basin", "bay", "beach", "caldera", "canyon", "cape",
    "cave", "channel", "cinder cone", "cirque", "cliff", "coast",
    "coastal landform", "col", "complex volcano", "drowned valley", "dune",
    "explosion crater", "extinct volcano", "fault", "fen", "forest",
    "gendarme", "gorge", "grassland", "headland", "highland", "hill",
    "island", "island group", "islet", "karst", "lagoon", "lake",
    "lake island", "landslide-dammed lake", "lateral volcano", "lava dome",
    "lava tube", "marsh", "mountain", "mountain pass", "mountain range",
    "mudflat", "peninsula", "plain", "plateau", "point", "pond",
    "pyramidal peak", "raised bog", "ravine", "reef", "ria", "ridge",
    "river", "river island", "rock", "rock formation", "rock shelter",
    "sea cave", "sea relic lake", "shoal", "skerry", "snowfield",
    "solutional cave", "spit", "spring", "still waters", "strait",
    "stratovolcano", "stream", "sulfur spring", "summit", "tectonic lake",
    "tholus", "tidal island", "tied island", "tombolo", "valley",
    "volcanic crater lake", "volcanic group", "volcanic island",
    "volcanic landform", "volcanic plug", "volcanic rock", "volcano",
    "waterfall", "waterhole", "wetland", "whirlpool",
    "special-looking rocks", "urami waterfall",

    # --- Trees / Plants ---
    "bamboo grove", "giant tree", "hibakujumoku", "lone cherry tree",
    "old-growth forest", "pine forest", "pine tree", "plum grove",
    "remarkable tree", "tree stump",

    # --- Parks / Gardens ---
    "arboretum", "botanical garden", "Chinese garden", "daimyō garden",
    "English garden", "exhibition park", "flower garden", "forest park",
    "garden", "general park (Japan)", "herb garden", "historical park",
    "Japanese garden", "Japanese strolling garden", "Japanese temple garden",
    "linear park", "memorial park", "National Garden",
    "National Government Park", "Olympic Park", "park", "roof garden",
    "roof terrace", "rose garden", "sacred grove", "scenic park",
    "science park", "sculpture garden", "seaside park", "sports park",
    "Tokyo metropolitan park", "traffic park", "urban park",
    "wild flower park", "archaeological park", "floral park",

    # --- Religious buildings ---
    "Buddhist temple", "Buddhist nunnery", "Shinto shrine", "cathedral",
    "Catholic cathedral", "Catholic church building", "Catholic parish church",
    "Anglican or Episcopal cathedral", "Eastern Orthodox cathedral",
    "Eastern Orthodox church building", "church building", "mosque",
    "Taoist temple", "temple", "temple of Confucius", "minor basilica",
    "monastery", "Trappist monastery", "Orthodox chapel",
    "co-cathedral", "religious building", "structure of worship",
    "provincial temple", "provincial nunnery", "bodaiji", "enkiri-dera",
    "hydrangea temple", "jingū", "jingū-ji", "taisha",
    "Shinto shrines by tradition",

    # --- Shrines (specific shrine types) ---
    "Akagi Shrine (worship)", "Akiha shrine", "Aso shrine",
    "Atago shrine (worship)", "Awashima shrine", "Dairokuten shrine",
    "Ebisu shrine", "Futaarayama Shrine", "Gion shrine",
    "Hachidai Ryūō Shrines", "Hachiman shrine", "Hakusan shrine",
    "Hie shrine", "Hikawa shrine", "Himuro Shrines",
    "Hitokotonushi shrine", "Hiyoshi shrine", "Inari shrine",
    "Itsukushima shrine", "Kamo shrine", "Kashima Shrine (worship)",
    "Kasuga shrine (worship)", "Katori shrine", "Kifune shrine",
    "Komagata shrine", "Kotohira shrine", "Kumano shrine",
    "Matsuo shrine", "Mishima shrine", "Mitake shrine",
    "Mitsumine shrine", "Miwa shrines", "Mononobe shrine",
    "Munakata shrine", "Nogi shrine", "Seimei Shrine",
    "Sengen shrine", "Shimogamo Shrine", "Shinmei shrine",
    "Shiogama-jinja (worship)", "Shirahige-jinja (worship)",
    "Suga shrine", "Suitengū", "Sumiyoshi shrine",
    "Susanoo shrine", "Suwa Shrine", "Taga shrine",
    "Tenmangū (worship)", "Tsushima shrine", "Yakumo shrine",
    "Yasaka shrine", "Ōyamazumi shrine", "Sannō shrine",
    "onsen shrine", "gokoku shrine", "shrine dedicated to Empress Jingū",
    "Kokubu-Hachimangu",

    # --- Shrine classifications ---
    "Betsu-gū", "Kanpei-sha", "Kokuhei-sha", "Shokan-sha",
    "Shikinaisha", "Shikinai Ronsha", "shikigesha",
    "Disputed Shikinaisha or Shikigeisha",
    "Shrine receiving Hoe and Quiver", "Shrine receiving Hoe offering",
    "Shrines receiving Tsukinami-sai and Niiname-sai and Ainame-sai offerings",
    "Shrines receiving Tsukinami-sai and Niiname-sai offerings",
    "chokusaisha", "Regional Sōja",
    "Kongōbu-ji Temple", "Enryaku-ji Temple", "Tōshō-gū",
    "Kōtai Jingū", "Toyōke Daijingū",

    # --- Buddhist temple sub-types ---
    "tatchū", "Amida-dō", "dō", "hokora", "honden", "kaisan-dō",
    "kyōzō", "massha", "setsumatsusha", "keidai-sessha", "keidai-sha",
    "keigai-sessha", "okumiya", "otabisho", "moto-Ise", "shukubō",
    "shōrō", "Taishi-dō",

    # --- Pilgrimage ---
    "Shikoku Pilgrimage",
    "Pilgrimage of the 20 exceptional temples of Shikoku",
    "pilgrimage site", "pilgrims' way",

    # --- Museums ---
    "museum", "museum building", "art museum", "art gallery",
    "history museum", "science museum", "natural history museum",
    "archaeological museum", "ethnographic museum", "folk museum",
    "local museum", "city museum", "national museum",
    "prefectural museum", "private museum", "university museum",
    "contemporary art museum", "museum of modern art",
    "museum of Asian art", "museum of broadcasting", "museum of culture",
    "architectural museum", "aviation museum", "automobile museum",
    "astronautical museum", "astronomical museum", "bank museum",
    "beer museum", "biographical museum", "calligraphy museum",
    "ceramics museum", "Christian museum", "coffee museum",
    "comics museum", "corporate museum", "craft museum",
    "customs museum", "dance museum", "doll museum",
    "education museum", "energy museum", "fashion museum",
    "film museum", "firefighting museum", "fishery museum",
    "food museum", "geological museum", "harbor museum",
    "Holocaust museum", "horological museum", "industry museum",
    "language museum", "literary museum", "maritime museum",
    "matsuri float museum", "medical museum", "meteorological museum",
    "military museum", "mining museum", "music museum",
    "naval museum", "numismatic museum", "open-air museum",
    "paleontological museum", "paper museum", "peace museum",
    "philatelic museum", "photography museum", "picture book museum",
    "printing museum", "prison museum", "railway museum",
    "religious museum", "sake museum", "sculpture museum",
    "sewerage museum", "sports museum", "textile museum",
    "theatre museum", "toy museum", "tramway museum",
    "transport museum", "waterworks museum", "whisky distillery",
    "Ainu museum", "agricultural museum", "alcoholic drinks museum",
    "Fudoki no oka", "historic house museum",
    "buried cultural property center", "children's museum",
    "defunct museum",

    # --- Libraries ---
    "library", "library network", "academic library", "children's library",
    "national library", "parliamentary library", "prefectural library of Japan",
    "public library", "private library", "private library of Japan",
    "religious library", "research library", "subscription library",
    "bunko", "municipal library of Japan", "main library",

    # --- Archive ---
    "archives", "house archive", "municipal archive", "public archive",

    # --- Buildings (general) ---
    "building", "building complex", "building of public administration",
    "historic building", "public building", "industrial building",
    "office building", "office", "station building",
    "A-bombed building", "Western-style building",
    "architectural ensemble", "architectural heritage monument",
    "architectural landmark", "architectural structure",
    "nonbuilding structure", "destroyed building or structure",
    "commercial building", "commercial complex",
    "prefectural office building", "government building",
    "government offices", "city hall", "Rathaus",
    "parliament building", "post office",
    "combined facility", "facility",

    # --- Houses / Residences ---
    "house", "single-family detached home", "villa",
    "historic house", "buke yashiki", "kominka", "machiya",
    "minka", "minka-en", "nagaya", "ijinkan", "guest house",
    "ryokan", "hotel", "hotel building", "casino hotel",
    "classic hotel", "railway hotel", "multifamily residential",
    "condominium", "vacation property",

    # --- Historic post towns / roads ---
    "69 Stations of the Nakasendō", "station of the Tōkaidō",
    "shukuba", "honjin", "Hatago", "kaidō", "historic road",
    "jōkamachi", "monzenmachi", "machinami",

    # --- Entertainment / Leisure ---
    "amusement park", "theme park", "amusement ride", "amusement arcade",
    "roller coaster", "4th-dimension roller coaster", "hypercoaster",
    "launched roller coaster", "steel roller coaster", "dark ride",
    "Ferris wheel", "Legoland", "miniature park",
    "family entertainment center", "leisure center", "resort",

    # --- Cultural / Performing arts ---
    "theatre building", "theatre company", "opera house",
    "national theatre", "cinema", "communal cinema", "movie theater",
    "performing arts building", "performing arts center",
    "concert hall", "music venue", "noh stage", "arts center",
    "arts venue", "cultural center", "cultural institution",
    "puppetry company", "dance hall",
    "media library", "television studio",

    # --- Sports ---
    "stadium", "Olympic stadium", "domed stadium", "arena",
    "indoor arena", "velodrome", "keirin racing track",
    "motorsport racing track", "kart circuit", "racing circuit",
    "motocross track", "horse racing venue", "kyōteijō",
    "association football pitch", "association football venue",
    "baseball field", "baseball venue", "ball game venue",
    "rugby union venue", "sports venue", "sports complex",
    "sports facility", "multi-purpose sports venue",
    "shooting range", "equestrian facility", "gym",
    "swimming pool", "swimming center", "indoor swimming pool",
    "ice rink", "ski resort", "ski jumping hill",
    "golf course", "golf club", "rowing and canoeing venue",
    "recreation center", "budōkan", "pitch",
    "defunct sports venue",

    # --- Water infrastructure ---
    "dam", "arch dam", "arch-gravity dam", "gravity dam",
    "embankment dam", "rock-fill dam", "earth-fill dam",
    "combined dam", "hydroelectric dam",
    "reservoir", "irrigation reservoir", "retention basin",
    "water reservoir", "canal", "canal lock", "lock", "sluice",
    "waterway", "waterworks", "pumping station",
    "hydroelectric power station", "pumped-storage power station",

    # --- Transport ---
    "railway station", "dead-end railway station", "junction station",
    "interchange station", "last station", "monorail station",
    "over-track railway station", "tram stop", "union station",
    "former railway station", "abandoned railway station",
    "railway line", "railway tunnel", "rail yard",
    "heritage railway", "abandoned railway",
    "suspension railway", "funicular", "aerial lift", "aerial tramway",
    "former bus station",
    "aerodrome", "road", "road junction", "road tunnel",
    "toll road", "controlled-access highway", "interchange",
    "tunnel", "undersea tunnel",
    "port", "port city", "dock", "pier", "breakwater",
    "marina", "passenger ship terminal", "fishing port",
    "cargo ship", "ocean liner", "steamship",
    "four-masted barque", "preserved watercraft",
    "lighthouse tender", "museum ship", "train ferry",
    "Meitetsu Nagoya Main Line", "Honshi-Bisan Line",
    "Higashi-Kyushu Expressway", "Iida Line",
    "Fukuoka Expressway Circular Route",
    "Japan National Route 354",
    "service area", "rest area", "roadside station",

    # --- Commercial / Shopping ---
    "shopping center", "shopping street", "shopping district",
    "shopping arcade in Japan", "shōtengai", "outlet mall",
    "market", "morning market", "food market", "fish market",
    "central wholesale market", "Chinatown",
    "commercial district", "central business district",
    "downtown", "prosperous area",

    # --- Food / Drink ---
    "restaurant", "Japanese restaurant", "Chinese restaurant", "café",
    "winery", "tea house", "chashitsu", "ochaya",

    # --- Accommodation ---
    # (covered in Houses / Residences)

    # --- Education ---
    "school", "elementary school in Japan",
    "elementary school in the Empire of Japan", "former school building",
    "agricultural school", "college", "university", "private university",
    "academic institution", "educational institution", "military academy",
    "Rangaku school",

    # --- Administrative divisions ---
    "city", "city of Japan", "city block", "ward of Japan",
    "ward area of Tokyo", "special ward of Japan",
    "town of Japan", "village of Japan", "village hall",
    "prefecture of Japan", "prefectural capital of Japan",
    "subprefecture of Japan", "province of Japan",
    "second-level administrative division", "chōchō",
    "capital of Japan", "ōaza",
    "human settlement", "neighborhood", "planned community",
    "college town", "ancient city", "old town", "holy city",
    "historic district", "spa town",

    # --- Regions / Historical ---
    "region of Japan", "historical region", "territory",
    "disputed territory", "Chūgoku region", "Kyushu", "Shikoku",
    "Asuka", "Heian-kyō", "Nagaoka-kyō", "Nikkō-shi", "Tsuruga",
    "Murone", "Shōnai-machi",

    # --- Provinces ---
    "Awa Province", "Awaji Province", "Bizen Province", "Bungo Province",
    "Dewa Province", "Hōki Province", "Iga Province", "Izumo Province",
    "Kawachi Province", "Kii Province", "Ōmi Province", "Owari Province",
    "Sanuki Province", "Satsuma Province", "Shimotsuke Province",
    "Suō Province", "Oshima Province",

    # --- Prefectures ---
    "Aomori Prefecture", "Ehime Prefecture", "Iwate Prefecture",
    "Kagoshima Prefecture", "Nagasaki Prefecture", "Okinawa Prefecture",
    "Ōita Prefecture",

    # --- Islands ---
    "artificial island", "Awaji Island", "Hirado Island",
    "Ishigaki Island", "Itsukushima", "Rishiri Island",
    "Yakushima", "former island", "uninhabited island",
    "Ōsumi Islands",

    # --- Countries ---
    "Japan", "North Korea", "South Korea", "Taiwan",
    "People's Republic of China", "United Nations",

    # --- Cemeteries / Tombs ---
    "cemetery", "buddhist cemetery", "foreign cemetery in Japan",
    "war cemetery", "reien",
    "Commonwealth War Graves Commission maintained cemetery",
    "tomb", "mausoleum", "Tokugawa mausoleum",
    "Imperial mausoleum or tomb", "imperial mausoleum",
    "imperial mausoleum/tomb reference site", "royal tomb",
    "kofun", "kofungun", "circular kofun", "tumulus",
    "cave tomb", "horizontal stone chamber",
    "stone circle",

    # --- Archaeological ---
    "archaeological site", "Shell midden",
    "group of archaeological sites",

    # --- Industrial ---
    "factory", "ironworks", "blast furnace", "reverberatory furnace",
    "silk mill", "shipyard", "mine", "mine building", "coal mine",
    "copper mine", "gold mine", "silver mine",
    "industrial heritage site", "charcoal clamp",

    # --- Power / Energy ---
    "coal-fired power station", "oil-fired power station",
    "nuclear power plant", "wind farm",

    # --- Military ---
    "military base", "military installation",
    "United States military base in Okinawa",
    "prison", "defunct prison", "prisoner-of-war camp",
    "detention house in Japan",

    # --- Science / Observatory ---
    "astronomical observatory", "radio telescope",
    "solar observatory", "solar telescope", "planetarium",
    "space center",

    # --- Government ---
    "consulate", "consulate general", "embassy", "chancery",
    "government agency", "local government",
    "ministry of culture",
    "District Public Prosecutors Office",

    # --- Events / Festivals ---
    "Japanese festival", "Tanabata", "kunchi", "nebuta", "matsuri",
    "lights festival", "winter festival", "rock festival",
    "championship", "marathon", "trade fair", "recurring event",
    "luminaria",

    # --- Streets / Roads / Paths ---
    "street", "avenue", "alley", "sidewalk", "path",
    "public staircase", "steep road", "boardwalk", "promenade",
    "numbered street", "Gojō Street", "Sanjō Street",
    "kiridōshi",

    # --- Misc structures ---
    "warehouse", "red brick warehouse", "former warehouse",
    "stable", "greenhouse", "information centre",
    "tourism office", "visitor center", "showroom",
    "bathhouse", "sentō", "communal bath",
    "one-day onsen facility", "onsen", "onsenkyō",
    "outdoor hot spring bath",

    # --- Organizations / Companies ---
    "organization", "defunct organization", "kabushiki gaisha",
    "broadcaster", "publishing house", "academic publisher",
    "brand", "business", "home electronics retailer",
    "transport company", "baseball team",
    "Central Japan Railway Company",

    # --- Wikimedia / Meta ---
    "Wikimedia category", "Wikimedia disambiguation page",
    "Category:Baseball venues in Japan", "database",

    # --- Pagoda ---
    "Japanese pagoda", "pagoda", "five-storied pagoda",
    "step pyramid",

    # --- Misc / Hard to classify ---
    "monolith", "stone", "ice cave", "icicle", "rock",
    "scenic viewpoint", "observation deck",
    "waterfront", "beach", "diorama",
    "model railroad layout", "Minato Oasis",
    "Roadside station Utazu Rinkaikoen",
    "National Treasure of Japan", "cultural heritage",
    "heritage site", "natural monument",
    "architectural landmark",
    "tourist attraction",
    "UNESCO Global Geopark",
    "Wildlife Protection Areas in Japan",
    "protected area",
    "prefectural natural parks of Japan",
    "quasi-national park of Japan",
    "national park",
    "highest point",
    "sacred mountain",
    "hometown Fuji",
    "fumoto",
    "kokushi genzaisha", "kokufu",

    # --- Hanamachi / Red light ---
    "hanamachi", "hanamachi in Osaka", "hanamachi in Kanazawa",
    "Kagai of Kyoto", "red-light district", "yūkaku",

    # --- Numerical groups ---
    "dyad", "triad", "tetrad", "pentad", "hexad", "heptad",
    "octad", "monad", "group",

    # --- Food / Agriculture ---
    "farm", "ranch", "stud farm", "horse stud farm",
    "bear park", "bird park", "bird sanctuary",
    "insectarium", "herpetarium",
    "public aquarium", "zoo", "safari park",
    "wildlife refuge",

    # --- Other specific ---
    "Japonic", "mythical object",
    "pawnshop", "bank building",
    "conference hall", "convention center",
    "function hall", "hall", "multi-purpose hall",
    "event venue", "venue",
    "business park", "campus",
    "urban renewal", "ground station",
    "escalator",
    "green roof", "terrace",
    "fangsheng pond", "wet moat",
    "windbreak",
    "itinerary",
    "jin'ya", "bansho", "daikansho", "sekisho",
    "gōshi",
    "utaki",
    "sōan",
    "polling place",
    "Osamu Dazai", "Mitsukuri Genpo", "Tomitaro Makino",
    "herbarium", "research institute",
    "artificial hill", "artificial lake", "artificial pond",
    "wedding chapel",
    "water park",
    "Nabari River",
    "flood bypass", "drainage tunnel",
    "water clock",
    "tour boat ride",

    # --- Unresolved Q-IDs (these are tags that remained as Wikidata IDs) ---
    "Q11366207", "Q11408323", "Q11429117", "Q11548891",
    "Q11558636", "Q11578893", "Q11660666", "Q121072018",
    "Q121844395", "Q123025722", "Q1230677", "Q134986946",
    "Q16694867", "Q17208880", "Q17209521", "Q20044722",
    "Q42311907",

    # --- Densely inhabited district ---
    "densely inhabited district in Japan",
    "urbanization promotion area",

    # --- Colony ---
    "colony",

    # --- Misc already covered ---
    "age", "former courthouse",

    # Additions for completeness
    "herbarium", "snapshot",
    "Korean Cultural Center branch office",
    "Tokutei daisanshu gyokō",
    "clubhouse", "community center", "film set",
    "flat-bottomed ferry", "former Buddhist temple",
    "former lake", "former theater", "group of lakes",
    "historic site", "imperial garden", "intersection",
    "joint", "main hall", "pre-dreadnought battleship",
    "railway park", "revenue house", "show cave",
    "technology museum", "underground city",
]

# Build the full mapping: add "other" for all OTHER_TAGS
for tag in OTHER_TAGS:
    if tag not in TAG_TO_CATEGORY:
        TAG_TO_CATEGORY[tag] = "other"


def get_category(instance_tags: list[str]) -> str:
    """
    Given a list of instance tags, determine the primary category.
    Priority: tower > bridge > palace_castle > arch_gate > monument_statue > other

    If multiple tags map to different non-other categories, use priority order.
    """
    categories_found = set()
    for tag in instance_tags:
        cat = TAG_TO_CATEGORY.get(tag, "other")
        categories_found.add(cat)

    priority = ["tower", "bridge", "palace_castle", "arch_gate", "monument_statue"]
    for cat in priority:
        if cat in categories_found:
            return cat
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map instance_tag values to categories: tower, bridge, palace_castle, arch_gate, monument_statue, other."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=_DEFAULT_INPUT,
        metavar="CSV",
        help=f"Input CSV file (default: {_DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        metavar="CSV",
        help="Output CSV file (default: <input>_mapped.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path.with_stem(input_path.stem + "_mapped")

    print(f"Reading: {input_path}")
    df = pd.read_csv(input_path)

    # First, check for any unmapped tags
    all_tags = set()
    for tags_str in df["instance_tag"].dropna():
        try:
            parsed = ast.literal_eval(tags_str)
            for t in parsed:
                all_tags.add(t)
        except:
            pass

    unmapped = [t for t in all_tags if t not in TAG_TO_CATEGORY]
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} unmapped tags (will default to 'other'):")
        for t in sorted(unmapped):
            print(f"  - {t}")

    # Apply mapping
    categories = []
    for _, row in df.iterrows():
        tags_str = row["instance_tag"]
        if pd.isna(tags_str) or tags_str == "" or tags_str == "[]":
            categories.append("other")
            continue
        try:
            tags = ast.literal_eval(tags_str)
            cat = get_category(tags)
            categories.append(cat)
        except:
            categories.append("other")

    df["category"] = categories

    # Summary
    print(f"\n--- Category distribution ---")
    counts = df["category"].value_counts()
    for cat, count in counts.items():
        print(f"  {cat}: {count}")

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Show examples for each non-other category
    for cat in ["tower", "bridge", "palace_castle", "arch_gate", "monument_statue"]:
        subset = df[df["category"] == cat]
        print(f"\n--- {cat} examples ({len(subset)} total) ---")
        for _, row in subset.head(10).iterrows():
            print(f"  {row['poi_name']}: {row['instance_tag']}")


if __name__ == "__main__":
    main()
