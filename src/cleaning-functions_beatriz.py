#Handling nulls safely
#Lowercases and strips whitespace
#Strips accents and special characters (é → e, ç → c, etc.)
#Removes anything that isn't a letter, number or space
#Collapses multiple spaces into one

def normalize(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


#normalize() cleans the text. find_variety() was the first attempt at extracting variety from it — simple, readable, but naive (no word boundaries, order-dependent). The regex pattern was the upgrade that fixed those problems by matching whole words only and prioritising longer variety names first.
def find_variety(name):
    for v in common_varieties:
        if v in name:
            return v
    return "Other"

#

variety_pattern = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in valid_varieties) + r")\b"
)