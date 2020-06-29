Projekt dotyczy pracy magisterkiej o tytule
# Wpływowość użytkowników w portalach społecznościowych w kontekście struktury ich sąsiedztw #

## Baza danych ##

Oprócz plików zawartych w repozytorium należy posiadać lokalną bazę danych zawierającą informacje z portalu Salon24.pl. 
Plik, który należy zaimportować w celu utworzenia lokalnej bazy danych zapisany w formacie sql znajduje się w folderze salon24_data.
W tym folderze znajdują się także wszystkie informacje niezbędne do korzystania z tej bazy danych.

## Generowane pliki ##
Wyniki działania programu przeznaczone dla użytkownika są zapisywane w folderze oputput. W folderze graphs znajdują się pliki związane z grafami, które są generowane w trakcie działania programu poprzez serializację. Przy kolejnych uruchomieniach są one deserializowane w celu przyspieszenia obliczeń (bez nich każdorazowo należałoby przetwarzać wszystkie relacje z bazy). Dodatkowo w wyniku predykcji tworzy się folder temp zawierający modele i zbiór danych wykorzystywanych do predykcji. 

## Wymagania ##
Do działania programu niezbędne jest zainstalowanie pakietów:

•	psycopg2

•	NetworkX

•	Pandas

•	NumPy

•	Matplotlib

•	Scikit-learn

•	Pickle

Dodatkowo w celu korzystania z prostego interfejsu użytkownika należy posiadać pakiet PyQt5. Korzystanie z graficznego interfejsu nie jest jednak wymagane.

## Uruchomienie programu ##
Zalecane jest uruchomienie programu z użyciem oprogramowania PyCharm. Uruchomienie obliczeń (programu) odbywa się poprzez klasę Manager. W konstruktorze tej klasy należy podać parametry do połączenia bazą danych. Jeśli baza nie zawiera wyliczonych wartości miar należy przed ich użyciem stworzyć odpowiednie kolumny poprzez wywołanie metody calculate. W tym celu wykorzystywane są grafy tworzone automatycznie na podstawie danych z bazy, które następnie są zapisywane w folderze graph, co umożliwia ich późniejsze użycie bez potrzeby ponownego przetwarzania relacji. 
	Możliwe jest korzystanie z prostego interfejsu graficznego stworzonego w ramach systemu lub wywoływanie funkcji bezpośrednio poprzez klasę Manager. Graficzny interfejs użytkownika jest realizowany w ramach klasy App.
Główną funkcjonalność programu realizują następujące funkcje klasy App (wywołujące odpowiednie metody klasy Manager):

a)	calculate – wyliczanie wartości nowych (jeszcze nie uwzględnionych w bazie) miar

b)	display – wypisywanie danych użytkowników spełniających określone warunki

c)	histogram – tworzenie histogramu dla wybranych danych

d)	statistics – wypisywanie statystyk dla wybranych danych

e)	correlation – wyliczanie korelacji dla wybranych miar

f)	ranking – tworzenie rankingu użytkowników w oparciu o podaną miarę

g)	table – tworzenie tabeli zawierającej wartości wybranych miar dla użytkowników do tabeli (z uwzględnieniem wszystkich lub tylko wybranych użytkowników)

h)	clustering – klastrowanie metodą k-means wraz z rysowaniem wykresów

i)	prediction – przeprowadzenie predykcji przyszłej wartości stopnia wejściowego


## Pliki wyjściowe ##
Wynik działania funkcji składowych programu to często wykres wizualizujący dane lub przebieg procesu związanego z ich przetwarzaniem i analizą. Dodatkowo, w przypadku większości funkcji możliwe jest zapisanie wyniku działania (np. tabelę z wartościami miar dla poszczególnych użytkowników) do pliku. Pliki te są zapisywane w folderze output.
Pliki wynikowe zapisywane są w formacie CSV. Ich szczegółowa zawartość zależy od funkcji, w której następuje zapis. Są to jednak tabele, które zawierają nazwy kolumn i jeśli jest taka potrzeba nazwy wierszy, które umożliwiają ich łatwą interpretację. Forma, w jakiej zapisane są wyniki umożliwia zaimportowanie ich w prosty sposób do programu Excel.
