# Projektni prijedlog — Prepoznavanje pješaka pomoću CNN-ova


## 1. Ime projekta
Prepoznavanje pješaka pomoću konvolucijskih neuronskih mreža


## 2. Tema i kratki opis projekta
Cilj projekta je izraditi model dubokog učenja arhitekture konvolucijske neuronske mreže (CNN) koji automatski prepoznaje (detektira) pješake na slikama (ili u videozapisima). Model će se trenirati na javno dostupnim skupovima podataka te evaluirati pomoću standardnih metrika točnosti detekcije. Ovisno o preostalom vremenu, provest će se eksperimenti s augmentiranim podacima i naprednija evaluacija u svrhu dobivanja boljeg uvida u rezultate.


## 3. Razrada projekta u zadatke
### 1. Prikupljanje i priprema podataka
    - Preuzimanje i analiza javnih podatkovnih skupova s označenim pješacima
        - primjeri: Caltech (https://data.caltech.edu/records/f6rph-90m20), EuroCity Persons Dataset (https://eurocity-dataset.tudelft.nl/), drugi (https://github.com/ViswanathaReddyGajjala/Datasets)
    - Formatiranje podatkovnog skupa u format pogodan za ulaz konvolucijske neuronske mreže
    - Normalizacija i augmentacija podataka radi poboljšanja generalizacije modela
### 2. Izrada i treniranje CNN-a
    - Dizajn i implementacija jednostavne arhitekture CNN-a (npr. u PyTorchu)
    - Eksperimentiranje s različitim arhitekturama ili gotovim modelima
    - Odabir konačne arhitekture i optimizacija hiperparametara (learning rate, batch size, broj epoha)
### 3. Evaluacija modela
    - Korištenje različitih metrika (precision, recall, F1-score, mAP)
    - Analiza pogrešaka i vizualizacija rezultata detekcije
### 4. Implementacija demonstracijskog sustava
    - Izrada jednostavnog sučelja u komandnoj liniji koje prikazuje rezultate prepoznavanja na slikama ili videu
### 5. Dokumentacija i prezentacija
    - Izrada izvještaja
    - Priprema materijala za predaju i obranu projekta


## 4. Ishodi projekta
- Implementiran, naučen i evaluiran CNN model sposoban detektirati pješake s određenom razinom točnosti
- Jednostavna demonstracijska aplikacija koja prikazuje rezultate
- Projektni izvještaj / dokumentacija


## 5. Dodjela poslova članovima tima
- **Mate Papak:**
    - implementacija osnovne arhitekture CNN-a
    - usporedba alternativnih arhitektura
    - izrada jednostavnog prezentacijskog alata
- **Filip Aleksić:** 
    - istraživanje, prikupljanje i priprema podataka
    - istraživanje, implementacija i usporedba alternativnih arhitektura
- **Luka Miličević:**
    - istraživanje, prikupljanje i priprema podataka
    - implementacija evaluacijskih metrika
    - izrada izvještaja
- **Roko Peran:**
    - istraživanje metoda augmentacije podataka i evaluacijskih metrika
    - istraživanje, implementacija i usporedba alternativnih arhitektura
    - izrada izvještaja
- **Vice Sladoljev:**
    - implementacija i optimizacija konačnog modela
    - analiza rezultata
    - izrada izvještaja


## 6. Okvirni vremenski plan rada
| Faza          | Aktivnost                                                      | Trajanje |
|---------------|----------------------------------------------------------------|----------|
| 1             | Prikupljanje podataka i analiza podatkovnih skupova            | 1 tjedan |
| 2             | Priprema podataka i izrada inicijalnog modela                  | 2 tjedna |
| 3             | Dizajn, implementacija i treniranje različitih modela          | 3 tjedna |
| 4             | Evaluacija i analiza rezultata                                 | 2 tjedan |
| 5             | Dokumentacija i završna priprema projekta                      | 1 tjedan |

Ukupno trajanje: **9 tjedana**