Čia turi būti dedami kompiuterinės tomografijos failai, parsisiųsti iš:
https://luna16.grand-challenge.org/download/

Svarbiausi failai - subset0 : subset9 ir candidates_V2.csv

Kompiuterinės tomografijos duomenims apdoroti naudojamas skriptas prep-pics.py, esantis code\parse-pics\ kataloge.
Iš kompiuterinės tomografijos duomenų suformuojamos dvimatės nuotraukos .tiff formatu, kurios saugomos data\pics\ kataloge.

! Dėmesio:

Komp. tomografijos failai yra virš 100 GB dydžio.
Apdorojant visus duomenis (~750.000 kandidatų) skripto veikimas gali užtrukti kelias dienas.

Jei skriptas lūžta (sakoma, kad negalima paskirti reikiamo atminties dydžio 
skaitant kažkurį .mhd failą), reikia padidinti virtualios atminties dydį operacinėje sistemoje.
