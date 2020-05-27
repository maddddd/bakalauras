# Bakalaurinis darbas. Tema - "Neuroninių tinklų naudojimas plaučių ligų analizei"

Parengė Vidmantas Bakštys, 2020

Darbas parengtas naudojant Python (versija 3.8) programavimo kalbą, naudojant jos distribuciją Anaconda (versija 2020.2).
Kuriant darbą naudota Windows10 operacinė sistema, tačiau jis turėtų veikti ir ant kitų operacinių sistemų (tačiau tai nėra ištestuota).


Pasiruošimo instrukcijos:

1)  Nuklonuojam repozitoriją.
2)  Parsisiunčiam Anaconda v2020.2 ir ją suinstaliuojam.
3)  (Nebūtina) Parsisiunčiam Nvidia Cuda v10.2 ir ją suinstaliuojam (tinka tik Nvidia grafinėms kortoms).
4)  Atidarome "Anaconda prompt" konsolę ir sukuriame naują conda aplinką:
		conda create --name bakalauras		- bus sukurta nauja aplinka, vardu "bakalauras"
5)  Aktyvuojame aplinką "Anaconda prompt" konsolėje:
		activate bakalauras					- bus aktyvuota aplinka, vardu "bakalauras"
6)  Suinstaliuojame PyTorch:
		conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
	Dauguma kitų reikalingų paketų, kaip pvz. numpy ar PIL yra suinstaliuojami kartu su PyTorch paketu arba yra iškart įtraukti į Anaconda
	distribuciją, todėl jų papildomai instaliuoti nebereikia.
7a) Jei nenorime patys išgavinėti nuotraukų iš kompiuterinės tomografijos duomenų:
	Įeiname į katalogą "data", kuriame yra "pics.7z" failas ir jį ten pat išarchyvuojame.
7b)	Jei norime patys išgauti nuotraukas iš kompiuterinės tomografijos duomenų (gali užtrukti ilgai):
	"Anaconda prompt" konsolėje, aktyvavę 4 žingsnyje sukurtą aplinką, rašome:
		pip install SimpleITK
8b)	Tada parsisiunčiame kompiuterinės tomografijos duomenis iš https://luna16.grand-challenge.org/Download/
	Reikalingi failai - subset0.zip : subset9.zip bei candidates_v2.csv failai
	Candidates_v2.csv failą tiesiog įdedame į "data" katalogą, o subset0.zip : subset9.zip failus išarchyvuojame "data" kataloge.
9b) Paleidžiame prep-pics.py skriptą, esantį "code\parse-pics\" kataloge ir laukiame, kol nuotraukos bus išgautos.
	Jei skriptas lūžta, reikia padidinti operacinės sistemos virtualią atmintį.
	
	
Naudojimo instrukcijos:

Norint naudoti tinklus, kviečiantysis kodas privalo patikrinti, ar jis yra pagrindinė programos gija, t.y. kodas turi būti apgaubtas konstrukcijos:
	if __name__ == "__main__":

1)  Norint naudoti neuroninį tinklą, pirma reikia jį sukurti ir ištreniruoti. Tai yra daroma tiesiai neuroninių tinklų failuose kataloge "code",
	pvz. parašome or_cnn.py failo gale:
		cnn = cnn = CNN(0.001, 50, 32, 'train', 40)		- bus sukurtas tinklas su į atmintį užkrautais treniravimo duomenimis.
2)	Treniruojame tinklą:
		cnn.train()										- treniravimo eiga aprašoma konsolėje. Tinklo elementų svoriai bus apmokyti.
3)	Išsaugom tinklą (svarbu, jei norime jį vėliau testuoti):
		tools.save_model(cnn, 'or_cnn')					- tinklo svoriai bus išsaugoti "trained_nets" kataloge.
4)	Įkeliame tinklą iš naujo iš failo (prasminga, jei norime pakeisti duomenų aibę, pvz. norime dirbti su 'test' tipo duomenimis)
		path = tools.get_model_path_in_hdd(cnn, 'or_cnn')
		cnn = tools.load_model(path, 'or_cnn', 'test', 50)
5)	Testuojame tinklą - testavimas ne tik ištestuoja tinklą, bet ir įrašo testavimo rezultatų vidurkius į failą "params.txt", esantį kataloge "trained_nets".
	Šie duomenys yra svarbūs, jei norime tinklą naudoti drauge su kitais tinklais (konsensuse).

Tinklų konsensuso testavimas:

Konsensuso tinklas pats savaime neegzistuoja, jis tiesiog naudoja kitų, jau ištreniruotų tinklų svorius ir testavimo nustatytų tikslumų rezultatus.
Norint testuoti tinklus drauge, reikia consensus.py failo gale, po apgaubiančia "if __name__ == "__main__":" konstrukcija, inicializuoti
klasės Consensus objektą. Ši klasė priima neuroninių tinklų objektus ir jų tikslumus kaip konstruktoriaus parametrus.

Patogiausia šiuos parametrus gauti funkcijos load_networks_from_paths() pagalba:
		nets, accs, pos, negs = load_networks_from_paths()	
Metodas parašo, kurie tinklai diske yra tinkami pasirinkimui, kaip dalyviai. 
Metodas yra interaktyvus, t.y. jis paprašo vartotojo įvesti norimų tinklų numerius. Eiliškumas tas, pagal kurį jie buvo išvardinti anksčiau.
Numeruojama nuo nulio. Pasirinkimo konsolėje pavyzdys:
	0 1 3 <Enter>				- bus pasirinkti 3 tinklai iš išvardintų tinklų sąrašo.

Konsensuso objekto inicializavimo pavyzdys:
    cons_net = Consensus(nets, accs, pos, negs, 'weighted', 0.001, 1, 32)

Testavimas naudojant konsensusą:
	cons_net.predict()			- bus atspausdinti testavimo rezultatai konsolėje.