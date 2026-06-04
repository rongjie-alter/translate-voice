#!/usr/bin/env python3
"""Client for batch voice transcription and translation.

Usage:
    voice-client transcribe --voice-dir ./voices --voice-index VoiceIndex.json
    voice-client translate  --voice-index VoiceIndex.json [--target-lang en]

Environment variables:
    TRANSCRIBE_SERVER_URL   default: http://localhost:8001
    TRANSLATE_SERVER_URL    default: http://localhost:8002
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

TRANSCRIBE_SERVER_URL = os.environ.get("TRANSCRIBE_SERVER_URL", "http://localhost:8001")
TRANSLATE_SERVER_URL = os.environ.get("TRANSLATE_SERVER_URL", "http://localhost:8002")

# BCP-47 language code → full name (moved here from the translategemma chat_template)
LANGUAGES: dict[str, str] = {
    "aa": "Afar", "aa-DJ": "Afar", "aa-ER": "Afar", "ab": "Abkhazian",
    "af": "Afrikaans", "af-NA": "Afrikaans", "ak": "Akan", "am": "Amharic",
    "an": "Aragonese", "ar": "Arabic", "ar-AE": "Arabic", "ar-BH": "Arabic",
    "ar-DJ": "Arabic", "ar-DZ": "Arabic", "ar-EG": "Arabic", "ar-EH": "Arabic",
    "ar-ER": "Arabic", "ar-IL": "Arabic", "ar-IQ": "Arabic", "ar-JO": "Arabic",
    "ar-KM": "Arabic", "ar-KW": "Arabic", "ar-LB": "Arabic", "ar-LY": "Arabic",
    "ar-MA": "Arabic", "ar-MR": "Arabic", "ar-OM": "Arabic", "ar-PS": "Arabic",
    "ar-QA": "Arabic", "ar-SA": "Arabic", "ar-SD": "Arabic", "ar-SO": "Arabic",
    "ar-SS": "Arabic", "ar-SY": "Arabic", "ar-TD": "Arabic", "ar-TN": "Arabic",
    "ar-YE": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "az-Arab": "Azerbaijani", "az-Arab-IQ": "Azerbaijani",
    "az-Arab-TR": "Azerbaijani", "az-Cyrl": "Azerbaijani",
    "az-Latn": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian",
    "be-tarask": "Belarusian", "bg": "Bulgarian", "bg-BG": "Bulgarian",
    "bm": "Bambara", "bm-Nkoo": "Bambara", "bn": "Bengali", "bn-IN": "Bengali",
    "bo": "Tibetan", "bo-IN": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "bs-Cyrl": "Bosnian", "bs-Latn": "Bosnian", "ca": "Catalan",
    "ca-AD": "Catalan", "ca-ES": "Catalan", "ca-FR": "Catalan",
    "ca-IT": "Catalan", "ce": "Chechen", "co": "Corsican", "cs": "Czech",
    "cs-CZ": "Czech", "cv": "Chuvash", "cy": "Welsh", "da": "Danish",
    "da-DK": "Danish", "da-GL": "Danish", "de": "German", "de-AT": "German",
    "de-BE": "German", "de-CH": "German", "de-DE": "German", "de-IT": "German",
    "de-LI": "German", "de-LU": "German", "dv": "Divehi", "dz": "Dzongkha",
    "ee": "Ewe", "ee-TG": "Ewe", "el": "Greek", "el-CY": "Greek",
    "el-GR": "Greek", "el-polyton": "Greek", "en": "English",
    "en-AE": "English", "en-AG": "English", "en-AI": "English",
    "en-AS": "English", "en-AT": "English", "en-AU": "English",
    "en-BB": "English", "en-BE": "English", "en-BI": "English",
    "en-BM": "English", "en-BS": "English", "en-BW": "English",
    "en-BZ": "English", "en-CA": "English", "en-CC": "English",
    "en-CH": "English", "en-CK": "English", "en-CM": "English",
    "en-CX": "English", "en-CY": "English", "en-CZ": "English",
    "en-DE": "English", "en-DG": "English", "en-DK": "English",
    "en-DM": "English", "en-ER": "English", "en-ES": "English",
    "en-FI": "English", "en-FJ": "English", "en-FK": "English",
    "en-FM": "English", "en-FR": "English", "en-GB": "English",
    "en-GD": "English", "en-GG": "English", "en-GH": "English",
    "en-GI": "English", "en-GM": "English", "en-GS": "English",
    "en-GU": "English", "en-GY": "English", "en-HK": "English",
    "en-HU": "English", "en-ID": "English", "en-IE": "English",
    "en-IL": "English", "en-IM": "English", "en-IN": "English",
    "en-IO": "English", "en-IT": "English", "en-JE": "English",
    "en-JM": "English", "en-KE": "English", "en-KI": "English",
    "en-KN": "English", "en-KY": "English", "en-LC": "English",
    "en-LR": "English", "en-LS": "English", "en-MG": "English",
    "en-MH": "English", "en-MO": "English", "en-MP": "English",
    "en-MS": "English", "en-MT": "English", "en-MU": "English",
    "en-MV": "English", "en-MW": "English", "en-MY": "English",
    "en-NA": "English", "en-NF": "English", "en-NG": "English",
    "en-NL": "English", "en-NO": "English", "en-NR": "English",
    "en-NU": "English", "en-NZ": "English", "en-PG": "English",
    "en-PH": "English", "en-PK": "English", "en-PL": "English",
    "en-PN": "English", "en-PR": "English", "en-PT": "English",
    "en-PW": "English", "en-RO": "English", "en-RW": "English",
    "en-SB": "English", "en-SC": "English", "en-SD": "English",
    "en-SE": "English", "en-SG": "English", "en-SH": "English",
    "en-SI": "English", "en-SK": "English", "en-SL": "English",
    "en-SS": "English", "en-SX": "English", "en-SZ": "English",
    "en-TC": "English", "en-TK": "English", "en-TO": "English",
    "en-TT": "English", "en-TV": "English", "en-TZ": "English",
    "en-UG": "English", "en-UM": "English", "en-VC": "English",
    "en-VG": "English", "en-VI": "English", "en-VU": "English",
    "en-WS": "English", "en-ZA": "English", "en-ZM": "English",
    "en-ZW": "English", "eo": "Esperanto", "es": "Spanish",
    "es-AR": "Spanish", "es-BO": "Spanish", "es-BR": "Spanish",
    "es-BZ": "Spanish", "es-CL": "Spanish", "es-CO": "Spanish",
    "es-CR": "Spanish", "es-CU": "Spanish", "es-DO": "Spanish",
    "es-EA": "Spanish", "es-EC": "Spanish", "es-ES": "Spanish",
    "es-GQ": "Spanish", "es-GT": "Spanish", "es-HN": "Spanish",
    "es-IC": "Spanish", "es-MX": "Spanish", "es-NI": "Spanish",
    "es-PA": "Spanish", "es-PE": "Spanish", "es-PH": "Spanish",
    "es-PR": "Spanish", "es-PY": "Spanish", "es-SV": "Spanish",
    "es-US": "Spanish", "es-UY": "Spanish", "es-VE": "Spanish",
    "et": "Estonian", "et-EE": "Estonian", "eu": "Basque", "fa": "Persian",
    "fa-AF": "Persian", "fa-IR": "Persian", "ff": "Fulah",
    "ff-Adlm": "Fulah", "ff-Adlm-BF": "Fulah", "ff-Adlm-CM": "Fulah",
    "ff-Adlm-GH": "Fulah", "ff-Adlm-GM": "Fulah", "ff-Adlm-GW": "Fulah",
    "ff-Adlm-LR": "Fulah", "ff-Adlm-MR": "Fulah", "ff-Adlm-NE": "Fulah",
    "ff-Adlm-NG": "Fulah", "ff-Adlm-SL": "Fulah", "ff-Adlm-SN": "Fulah",
    "ff-Latn": "Fulah", "ff-Latn-BF": "Fulah", "ff-Latn-CM": "Fulah",
    "ff-Latn-GH": "Fulah", "ff-Latn-GM": "Fulah", "ff-Latn-GN": "Fulah",
    "ff-Latn-GW": "Fulah", "ff-Latn-LR": "Fulah", "ff-Latn-MR": "Fulah",
    "ff-Latn-NE": "Fulah", "ff-Latn-NG": "Fulah", "ff-Latn-SL": "Fulah",
    "fi": "Finnish", "fi-FI": "Finnish", "fil-PH": "Filipino",
    "fo": "Faroese", "fo-DK": "Faroese", "fr": "French", "fr-BE": "French",
    "fr-BF": "French", "fr-BI": "French", "fr-BJ": "French",
    "fr-BL": "French", "fr-CA": "French", "fr-CD": "French",
    "fr-CF": "French", "fr-CG": "French", "fr-CH": "French",
    "fr-CI": "French", "fr-CM": "French", "fr-DJ": "French",
    "fr-DZ": "French", "fr-FR": "French", "fr-GA": "French",
    "fr-GF": "French", "fr-GN": "French", "fr-GP": "French",
    "fr-GQ": "French", "fr-HT": "French", "fr-KM": "French",
    "fr-LU": "French", "fr-MA": "French", "fr-MC": "French",
    "fr-MF": "French", "fr-MG": "French", "fr-ML": "French",
    "fr-MQ": "French", "fr-MR": "French", "fr-MU": "French",
    "fr-NC": "French", "fr-NE": "French", "fr-PF": "French",
    "fr-PM": "French", "fr-RE": "French", "fr-RW": "French",
    "fr-SC": "French", "fr-SN": "French", "fr-SY": "French",
    "fr-TD": "French", "fr-TG": "French", "fr-TN": "French",
    "fr-VU": "French", "fr-WF": "French", "fr-YT": "French",
    "fy": "Western Frisian", "ga": "Irish", "ga-GB": "Irish",
    "gd": "Scottish Gaelic", "gl": "Galician", "gn": "Guarani",
    "gu": "Gujarati", "gu-IN": "Gujarati", "gv": "Manx", "ha": "Hausa",
    "ha-Arab": "Hausa", "ha-Arab-SD": "Hausa", "ha-GH": "Hausa",
    "ha-NE": "Hausa", "he": "Hebrew", "he-IL": "Hebrew", "hi": "Hindi",
    "hi-IN": "Hindi", "hi-Latn": "Hindi", "hr": "Croatian",
    "hr-BA": "Croatian", "hr-HR": "Croatian", "ht": "Haitian",
    "hu": "Hungarian", "hu-HU": "Hungarian", "hy": "Armenian",
    "ia": "Interlingua", "id": "Indonesian", "id-ID": "Indonesian",
    "ie": "Interlingue", "ig": "Igbo", "ii": "Sichuan Yi", "ik": "Inupiaq",
    "io": "Ido", "is": "Icelandic", "it": "Italian", "it-CH": "Italian",
    "it-IT": "Italian", "it-SM": "Italian", "it-VA": "Italian",
    "iu": "Inuktitut", "iu-Latn": "Inuktitut", "ja": "Japanese",
    "ja-JP": "Japanese", "jv": "Javanese", "ka": "Georgian", "ki": "Kikuyu",
    "kk": "Kazakh", "kk-Arab": "Kazakh", "kk-Cyrl": "Kazakh",
    "kk-KZ": "Kazakh", "kl": "Kalaallisut", "km": "Central Khmer",
    "kn": "Kannada", "kn-IN": "Kannada", "ko": "Korean", "ko-CN": "Korean",
    "ko-KP": "Korean", "ko-KR": "Korean", "ks": "Kashmiri",
    "ks-Arab": "Kashmiri", "ks-Deva": "Kashmiri", "ku": "Kurdish",
    "kw": "Cornish", "ky": "Kyrgyz", "la": "Latin", "lb": "Luxembourgish",
    "lg": "Ganda", "ln": "Lingala", "ln-AO": "Lingala", "ln-CF": "Lingala",
    "ln-CG": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lt-LT": "Lithuanian",
    "lu": "Luba-Katanga", "lv": "Latvian", "lv-LV": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "ml-IN": "Malayalam", "mn": "Mongolian", "mn-Mong": "Mongolian",
    "mn-Mong-MN": "Mongolian", "mr": "Marathi", "mr-IN": "Marathi",
    "ms": "Malay", "ms-Arab": "Malay", "ms-Arab-BN": "Malay", "ms-BN": "Malay",
    "ms-ID": "Malay", "ms-SG": "Malay", "mt": "Maltese", "my": "Burmese",
    "nb": "Norwegian Bokmål", "nb-SJ": "Norwegian Bokmål",
    "nd": "North Ndebele", "ne": "Nepali", "ne-IN": "Nepali", "nl": "Dutch",
    "nl-AW": "Dutch", "nl-BE": "Dutch", "nl-BQ": "Dutch", "nl-CW": "Dutch",
    "nl-NL": "Dutch", "nl-SR": "Dutch", "nl-SX": "Dutch",
    "nn": "Norwegian Nynorsk", "no": "Norwegian", "no-NO": "Norwegian",
    "nr": "South Ndebele", "nv": "Navajo", "ny": "Chichewa", "oc": "Occitan",
    "oc-ES": "Occitan", "om": "Oromo", "om-KE": "Oromo", "or": "Oriya",
    "os": "Ossetian", "os-RU": "Ossetian", "pa": "Punjabi", "pa-IN": "Punjabi",
    "pa-Arab": "Punjabi", "pa-Guru": "Punjabi", "pl": "Polish",
    "pl-PL": "Polish", "ps": "Pashto", "ps-PK": "Pashto", "pt": "Portuguese",
    "pt-AO": "Portuguese", "pt-BR": "Portuguese", "pt-CH": "Portuguese",
    "pt-CV": "Portuguese", "pt-GQ": "Portuguese", "pt-GW": "Portuguese",
    "pt-LU": "Portuguese", "pt-MO": "Portuguese", "pt-MZ": "Portuguese",
    "pt-PT": "Portuguese", "pt-ST": "Portuguese", "pt-TL": "Portuguese",
    "qu": "Quechua", "qu-BO": "Quechua", "qu-EC": "Quechua", "rm": "Romansh",
    "rn": "Rundi", "ro": "Romanian", "ro-MD": "Romanian", "ro-RO": "Romanian",
    "ru": "Russian", "ru-BY": "Russian", "ru-KG": "Russian",
    "ru-KZ": "Russian", "ru-MD": "Russian", "ru-RU": "Russian",
    "ru-UA": "Russian", "rw": "Kinyarwanda", "sa": "Sanskrit",
    "sc": "Sardinian", "sd": "Sindhi", "sd-Arab": "Sindhi",
    "sd-Deva": "Sindhi", "se": "Northern Sami", "se-FI": "Northern Sami",
    "se-SE": "Northern Sami", "sg": "Sango", "si": "Sinhala", "sk": "Slovak",
    "sk-SK": "Slovak", "sl": "Slovenian", "sl-SI": "Slovenian", "sn": "Shona",
    "so": "Somali", "so-DJ": "Somali", "so-ET": "Somali", "so-KE": "Somali",
    "sq": "Albanian", "sq-MK": "Albanian", "sq-XK": "Albanian",
    "sr": "Serbian", "sr-RS": "Serbian", "sr-Cyrl": "Serbian",
    "sr-Cyrl-BA": "Serbian", "sr-Cyrl-ME": "Serbian", "sr-Cyrl-XK": "Serbian",
    "sr-Latn": "Serbian", "sr-Latn-BA": "Serbian", "sr-Latn-ME": "Serbian",
    "sr-Latn-XK": "Serbian", "ss": "Swati", "ss-SZ": "Swati",
    "st": "Southern Sotho", "st-LS": "Southern Sotho", "su": "Sundanese",
    "su-Latn": "Sundanese", "sv": "Swedish", "sv-AX": "Swedish",
    "sv-FI": "Swedish", "sv-SE": "Swedish", "sw": "Swahili",
    "sw-CD": "Swahili", "sw-KE": "Swahili", "sw-TZ": "Swahili",
    "sw-UG": "Swahili", "ta": "Tamil", "ta-IN": "Tamil", "ta-LK": "Tamil",
    "ta-MY": "Tamil", "ta-SG": "Tamil", "te": "Telugu", "te-IN": "Telugu",
    "tg": "Tajik", "th": "Thai", "th-TH": "Thai", "ti": "Tigrinya",
    "ti-ER": "Tigrinya", "tk": "Turkmen", "tl": "Tagalog", "tn": "Tswana",
    "tn-BW": "Tswana", "to": "Tonga", "tr": "Turkish", "tr-CY": "Turkish",
    "tr-TR": "Turkish", "ts": "Tsonga", "tt": "Tatar", "ug": "Uyghur",
    "uk": "Ukrainian", "uk-UA": "Ukrainian", "ur": "Urdu", "ur-IN": "Urdu",
    "ur-PK": "Urdu", "uz": "Uzbek", "uz-Arab": "Uzbek", "uz-Cyrl": "Uzbek",
    "uz-Latn": "Uzbek", "ve": "Venda", "vi": "Vietnamese",
    "vi-VN": "Vietnamese", "vo": "Volapük", "wa": "Walloon", "wo": "Wolof",
    "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba", "yo-BJ": "Yoruba",
    "za": "Zhuang", "zh": "Chinese", "zh-CH": "Chinese", "zh-TW": "Chinese",
    "zh-Hans": "Chinese", "zh-Hans-HK": "Chinese", "zh-Hans-MO": "Chinese",
    "zh-Hans-MY": "Chinese", "zh-Hans-SG": "Chinese", "zh-Hant": "Chinese",
    "zh-Hant-HK": "Chinese", "zh-Hant-MO": "Chinese", "zh-Hant-MY": "Chinese",
    "zh-Latn": "Chinese", "zu": "Zulu", "zu-ZA": "Zulu",
}


def _load_voice_index(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_voice_index(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(data.items())), f, ensure_ascii=False, indent="\t")


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_transcribe(args: argparse.Namespace) -> None:
    voice_dir = Path(args.voice_dir)
    index_path = Path(args.voice_index)

    if not voice_dir.is_dir():
        print(f"Error: {voice_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    voice_index = _load_voice_index(index_path)

    ogg_files = sorted(voice_dir.glob("voice_*.ogg"))
    pending = [f for f in ogg_files if not voice_index.get(f.stem, {}).get("jp")]

    if not pending:
        print("All files already transcribed.")
        return

    with tqdm(pending, desc="Transcribing", unit="file") as pbar:
        for ogg_file in pbar:
            pbar.set_postfix(file=ogg_file.name)
            try:
                with open(ogg_file, "rb") as f:
                    resp = httpx.post(
                        f"{TRANSCRIBE_SERVER_URL}/transcribe",
                        files={"audio": (ogg_file.name, f, "audio/ogg")},
                        timeout=120.0,
                    )
                resp.raise_for_status()
                text: str = resp.json()["result"]["text"]
                voice_index.setdefault(ogg_file.stem, {})["jp"] = text
                _save_voice_index(index_path, voice_index)
            except Exception as exc:
                tqdm.write(f"  ERROR {ogg_file.name}: {exc}")


def cmd_translate(args: argparse.Namespace) -> None:
    target_lang: str = args.target_lang
    index_path = Path(args.voice_index)

    if target_lang not in LANGUAGES:
        print(
            f"Warning: '{target_lang}' not in the known language map; proceeding anyway.",
            file=sys.stderr,
        )

    voice_index = _load_voice_index(index_path)

    glossary: dict | None = None
    if args.glossary_file:
        with open(args.glossary_file, encoding="utf-8") as f:
            glossary = json.load(f)

    pending = [
        (key, entry["jp"])
        for key, entry in sorted(voice_index.items())
        if entry.get("jp") and not entry.get(target_lang)
    ]

    if not pending:
        print(f"Nothing to translate to '{target_lang}'.")
        return

    payload_base: dict = {"source_lang": "ja", "target_lang": target_lang}
    if glossary:
        payload_base["custom_glossary"] = glossary
    if args.context:
        payload_base["context"] = args.context

    with tqdm(pending, desc=f"Translating → {target_lang}", unit="entry") as pbar:
        for key, jp_text in pbar:
            pbar.set_postfix(key=key)
            try:
                resp = httpx.post(
                    f"{TRANSLATE_SERVER_URL}/translate",
                    json={**payload_base, "text": jp_text},
                    timeout=300.0,
                )
                resp.raise_for_status()
                translated: str = resp.json()["result"]["text"]
                voice_index[key][target_lang] = translated
                _save_voice_index(index_path, voice_index)
            except Exception as exc:
                tqdm.write(f"  ERROR '{key}': {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-transcribe and translate voice_*.ogg files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="Transcribe audio files → Japanese text")
    p_tr.add_argument("--voice-dir", required=True, metavar="DIR",
                      help="Directory containing voice_*.ogg files")
    p_tr.add_argument("--voice-index", default="VoiceIndex.json", metavar="FILE",
                      help="Path to VoiceIndex.json (default: VoiceIndex.json)")

    p_xl = sub.add_parser("translate", help="Translate Japanese text entries")
    p_xl.add_argument("--voice-index", default="VoiceIndex.json", metavar="FILE",
                      help="Path to VoiceIndex.json (default: VoiceIndex.json)")
    p_xl.add_argument("--target-lang", default="en", metavar="CODE",
                      help="BCP-47 target language code (default: en)")
    p_xl.add_argument("--glossary-file", metavar="FILE",
                      help='JSON file mapping source terms to translations, e.g. {"勇者":"Hero"}')
    p_xl.add_argument("--context", metavar="TEXT",
                      help="Free-text context hint sent to the translator")

    args = parser.parse_args()
    if args.command == "transcribe":
        cmd_transcribe(args)
    else:
        cmd_translate(args)


if __name__ == "__main__":
    main()
