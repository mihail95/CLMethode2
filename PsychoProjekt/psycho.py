import pandas as pd
import spacy
from spacy.tokens import Doc
from scipy.stats import pearsonr


def count_konjunktiv(item: Doc) -> int:
    """Counts up the amount of times a Konjunktiv sentence is used in a spacy document."""
    konj_count = 0
    for sent in item.sents:
        # Mood = 'Sub' soll Konjunktiv sein
        moods = [token.morph.get("Mood") for token in sent]
        konj_count += int(['Sub'] in moods)

    return konj_count

def count_weak_language(text: str) -> int:
    """Counts up the number of times a 'weak' word appears in the given text."""
    unschaerfeindikatoren = [
        "vielleicht", "möglicherweise", "wahrscheinlich", "eventuell", "ziemlich",
        "relativ", "im Allgemeinen", "in der Regel", "tendenziell", "mehr oder weniger",
        "ein bisschen", "fast", "annähernd", "circa", "im Wesentlichen",
        "oft", "gelegentlich", "manchmal", "oftmals", "häufig",
        "irgendwie", "sozusagen", "in gewisser Weise", "unter Umständen", "denkbar",
        "anscheinend", "scheinbar", "vermutlich", "potenziell", "im Großen und Ganzen",
        "prinzipiell", "hauptsächlich", "so gut wie", "weitgehend", "gewissermaßen",
        "theoretisch", "größtenteils", "überwiegend", "gegebenenfalls", "unter Umständen",
        "vage", "unzureichend", "ungefähr", "grob geschätzt", "eher",
        "kaum", "nicht ganz", "praktisch", "trotzdem", "theoretisch gesehen",
        "beinahe", "so gut wie", "weitestgehend", "einigermaßen", "partiell",
        "weitgehend", "prinzipiell", "halbwegs", "durchaus", "tendenziell",
        "oberflächlich betrachtet", "mehr oder minder", "annäherungsweise", "im Allgemeinen", "gewöhnlich",
        "bis zu einem gewissen Grad", "mit Vorbehalt", "eher weniger", "in etwa", "fast immer",
        "vorwiegend", "selten", "in den meisten Fällen", "hin und wieder", "ab und zu",
        "vereinzelt", "zeitweise", "sporadisch", "fallweise", "im Prinzip",
        "gewöhnlich", "hauptsächlich", "teilweise", "größtenteils", "maßgeblich",
        "überwiegend", "im Wesentlichen", "in gewissem Umfang", "zu einem gewissen Grad", "einigermaßen",
        "zu einem bestimmten Maß", "im Großen und Ganzen", "so gut wie", "weitestgehend", "fast"
    ]

    weak_words_count = {}
    for indikator in unschaerfeindikatoren:
        weak_words_count[indikator] = text.count(indikator)

    return sum(weak_words_count.values())

def tokens_dep_tagging(document: Doc) -> list:
    """Diese Funktion nimmt einen Document als Input und liefert eine Liste von Sätzen, die ebenfalls Listen von
    Tupeln mit einem Token als erstes und einem POS-Tag als zweites Element darstellen"""

    sent_with_toks = []

    for sent in document.sents:
        sentence = []
        for token in sent:
            sentence.append((token, token.pos_))
        sent_with_toks.append(sentence)
    return sent_with_toks

def passiv_finder(sent_with_toks):
    """Diese Funktion nimmt eine Liste von Sätzen mit Tupeln (Token als erstes Element und den POS-Tag
     als zweites Element) als Input und liefert ein Passiv-Score dazu als Output"""

    passiv_score = 0
    for sent in sent_with_toks:
        sent_lemma = list()
        morph_infos = ""
        for tok in sent:
            sent_lemma.append(tok[0].lemma_)
        for tok in sent:
            morph_infos += str(tok[0].morph)
        if "werden" in sent_lemma and "VerbForm=Part" in morph_infos:
            passiv_score += 1
        #elif "lassen" in sent_lemma and "VerbForm=Inf" in morph_infos:
        #    passiv_score +=1

    return passiv_score


############# MAIN SCRIPT ##############
nlp = spacy.load("de_core_news_sm")
config = {"punct_chars": None}
nlp.add_pipe("sentencizer", config=config)

df = pd.read_csv("train.csv", sep=",")
# df = pd.read_csv("trial.csv", sep=",")

mindset_df = df[["BR01_01", "wahrg_Mindset"]].copy()
mindset_df["weak_word_count"] = 0
mindset_df["konjunktiv_count"] = 0
tokens_mit_dep_tags = []

for i in mindset_df.index:
    item_doc = nlp(mindset_df["BR01_01"][i])
    item_konjunktiv_count = count_konjunktiv(item_doc)
    item_weak_word_count = count_weak_language(mindset_df["BR01_01"][i])
    mindset_df.loc[i, "weak_word_count"] = item_weak_word_count
    mindset_df.loc[i, "konjunktiv_count"] = item_konjunktiv_count
    tokens_mit_dep_tags.append(tokens_dep_tagging(item_doc))


print("------------------------ RESULTS ------------------------------------------------------------------")
########### UNSCHÄRFEINDIKATOREN #####################
x = mindset_df["wahrg_Mindset"]
y = mindset_df["weak_word_count"]

x_array = x.to_numpy()
y_array = y.to_numpy()

# Keine signifikante Korrelation vorhanden
result_weak = pearsonr(x_array, y_array)
print(f"Unschärfeindikatoren Korrelation (Pearson's R): {result_weak.statistic} with p-value({result_weak.pvalue})\n")

##########  KONJUNKTIV  ####################
# Keine signifikante Korrelation vorhanden
result_konj = pearsonr(mindset_df["wahrg_Mindset"], mindset_df["konjunktiv_count"])
print(f"Anzahl Konjunktivsätze Korrelation (Pearson's R): {result_konj.statistic} with p-value({result_konj.pvalue})\n")

########### PASSIV #####################
passiv_scores = list()

for letter in tokens_mit_dep_tags:
    passiv_scores.append(passiv_finder(letter))

mindset_df["Anzahl Passivsätze"] = passiv_scores
passiv_saetze_corr_test = pearsonr(mindset_df["wahrg_Mindset"], mindset_df["Anzahl Passivsätze"])

# geringe negative korrelation
print(f"Passivkonstruktionen Korrelation (Pearson's R): {passiv_saetze_corr_test.statistic} with p-value({passiv_saetze_corr_test.pvalue})\n")

mindset_df.to_csv('mindset_analysis.csv')