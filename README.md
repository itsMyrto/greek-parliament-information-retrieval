# Greek parliament information retrieval

## Προαπαιτήσεις
- Python 3.x

## Εγκατάσταση
- Πλοηγηθείτε στον φάκελο του project
- Δημιουργήστε ένα virtual environment Python: `python -m venv venv`
- Ενεργοποιήστε το virtual environment:
    - Σε Windows: `.\venv\Scripts\activate`
    - Σε macOS/Linux: `source venv/bin/activate`
- Εγκαταστήστε τις βιβλιοθήκες του project: `pip install -r requirements.txt`
- Εγκαταστήστε την επέκταση για την ελληνική γλώσσα του Spacy: `python -m spacy download el_core_news_sm`

## Εκτέλεση του project

Όλα τα μέρη μπορούν να εκτελεστούν ανεξάρτητα μεταξύ τους

### Μέρος 1
- Αρχεία μέρους 1: `dataCleanupPart1.py`, `inverse_index.py` @itsMyrto προσθέστε τα υπόλοιπα

### Μέρος 2 (λέξεις-κλειδιά)
- Αρχεία μέρους 2: `part3.py`, `cacheAndSaved/` `speeches.db` (generated) `helpers/`
    
    Το μέρος 2 υποστηρίζει όλα τα ζητούμενα της εκφώνησης και τα εκτελέι με command line arguments `python part2.py --help` για πληροφορίες.

    - `python part2.py --process-keywords` Υπολογίζει τα keywords ανά ομιλία, ομιλητή και κόμμα και λόγω όγκου τα αποθηκεύει στη βάση αντί να τα εμφανίζει.

    - `python parth2.py --gather-politician-keywords "μητσοτακης κωνσταντινου κυριακος"` Υπολογίζει τα keywords ανά χρονιά για το άτομο στα εισαγωγικά. Το όνομα δίνεται σε μη τονισμένη τυπική μορφή, όπως στο παράδειγμα.

    - `python parth2.py --gather-party-keywords "νεα δημοκρατια"` Υπολογίζει τα keywords ανά χρονιά για το κόμμα στα εισαγωγικά. Το όνομα δίνεται σε μη τονισμένη τυπική μορφή, όπως στο παράδειγμα.

    - `python part2.py --gather-total-keywords` Υπολογίζει τα keywords ανά ομιλία, και εμφανίζει ανά χρονιά την εξέλιξή τους.

    - `python part2.py --demo` Τρέχει ένα demo που εμφανίζει τη χρονική εξέλιξη των keywords για τον κ. Κυριάκο Μητσοτάκη και τη Νέα Δημοκρατία

    Επίσης παρέχονται οι παράμετροι `--max-rows {number}` και `--output {filename.csv}` ώστε η εκτέλεση να πραγματοποιηθεί με άνω φράγμα στο πλήθος των ομιλιών ή να διοχετευτεί το output σε ένα csv αρχείο.

### Μέρος 3 (Ομοιότητες)
- Αρχεία μέρους 3: `part3.py`, `cacheAndSaved/` `speeches.db` (generated) `helpers/`
    
    Για το μέρος 3, αρκεί να τρέξει `python part3.py` που by-default θα εμφανίσει τα 5 πιο όμοια ζεύγη.

    Επίσης παρέχονται οι παράμετροι `--top-k {number}` και `--output {filename.csv}` ώστε να επιστρέψει επιθυμητό πλήθος ζευγών ή να διοχετευτεί το output σε ένα csv αρχείο.

### Μέρος 4

### Μέρος 5

### Μέρος 6
- Αρχεία μέρους 6: `Θα βάλω ένα pre-flight για να κάνει τη βάση και θα το γράψω`
