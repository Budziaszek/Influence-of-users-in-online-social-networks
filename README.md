The project concerns a master's thesis
# Influence of users on social networks in the context of the structure of their neighborhoods #

## Database ##

In addition to the files contained in the repository, you must have a local database containing information from the Salon24.pl portal.
The file that needs to be imported to create a local database, saved in sql format, is located in the salon24_data folder.
This folder also contains all the information necessary to use this database.

## Generated files ##
The program's results for the user are saved in the oputput folder. In the graphs folder there are files related to graphs, which are generated during the program's operation by serialization. At subsequent launches, they are deserialized to speed up the calculations (without them, all relations from the database should be processed each time). Additionally, as a result of prediction, a temp folder is created containing models and a set of data used for prediction.

## Requirements ##
For the program to work, it is necessary to install the following packages:

• psycopg2

• NetworkX

• Pandas

• NumPy

• Matplotlib

• Scikit-learn

• Pickle

Additionally, in order to use the simple user interface, you must have the PyQt5 package. However, the use of a graphical interface is not required.

## Running the program ##
It is recommended to run the program with PyCharm software. The calculation (program) is started through the Manager class. In the constructor of this class, parameters for database connection should be provided. If the database does not contain the calculated values ​​of measures, you should create appropriate columns before using them by calling the calculate method. For this purpose, graphs created automatically on the basis of data from the database are used, which are then saved in the graph folder, which allows them to be used later without the need to reprocess the relationship.
It is possible to use the simple graphical interface created within the system or call functions directly through the Manager class. The graphical user interface is implemented within the App class.
The main functionality of the program is performed by the following functions of the App class (calling appropriate methods of the Manager class):

a) calculate - calculation of new (not yet included in the database) measures

b) display - displaying data of users meeting certain conditions

c) histogram - creating a histogram for selected data

d) statistics - printing out statistics for selected data

e) correlation - calculating correlation for selected measures

f) ranking - creating a ranking of users based on a given measure

g) table - creating a table containing the values ​​of selected measures for users to a table (including all or only selected users)

h) clustering - k-means clustering with drawing charts

i) prediction - making a prediction of the future value of the input stage


## Output files ##
The result of operation of program component functions is often a graph visualizing data or the course of a process related to their processing and analysis. Additionally, in the case of most functions, it is possible to save the result of an operation (e.g. a table with measurement values ​​for individual users) to a file. These files are saved in the output folder.
The resulting files are saved in CSV format. Their detailed content depends on the function in which the write takes place. However, these are tables that contain column names and, if necessary, row names that allow easy interpretation. The form in which the results are saved makes it easy to import them into Excel.
