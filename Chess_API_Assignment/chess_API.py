import json
import requests
import matplotlib.pyplot as plt
import datetime
import urllib.request
import numpy as np
from scipy.optimize import curve_fit
import timeit

print('Libraries Imported') #Output for user.

#####################################################################################################
# Section 1
'''
Import libraries Define functions for working with json files, API urls and trend fitting etc.
'''

def create_file(name):
    with open(name, 'w') as f:
        pass
    

def fill_file(url, name, mode):       #Name: 'name_of_file.json', Mode: w = overwrite, a = append
    
    find_url = requests.get(url)
    data = find_url.json()
    
    with open(name, mode) as f:
        json.dump(data, f)     #Saves data into it

        
def open_file(file_name):

    with open(file_name) as f: #Opens file for use
        data = json.load(f)
    
    return(data)


def url_prep(url):
    response = urllib.request.urlopen(url).read()  #Opening up URLs
    json_obj = str(response, 'utf-8')  #Converting for json loader
    data = json.loads(json_obj)    #Loading json data
    return(data)


def players(num_players, time_class):   #Searches leaderboards for the top 'x' amount of players for a certain time class
                                        #e.g. top 50 bullet chess players
    username = []

    for item in data[time_class]:

        if item['rank'] <= num_players:
            username.append(item['username'])

    return(username)


def merge_dict(dic2, dic1): #Function for merging two dictionaries.
    for key in dic1:
        dic2.setdefault(key, []).append(dic1[key])
        

def func(x,a,b,c): #Function for logarithmic fitting.
    if c == 1:
        pass
    return a*np.log(x)+ b


#####################################################################################################
# Section 2
'''
Create and fill leaderboard file from API url to be used later
'''

create_file('chess_leaderboards.json')
fill_file('https://api.chess.com/pub/leaderboards', 'chess_leaderboards.json', 'w')


#####################################################################################################
# Section 3
'''
Locate top 'x' (less than or equal to 50) amount of players for specified time control
e.g. bullet (2 minute game: 60 seconds for each player), from leaderboard file created.
'''

data = open_file('chess_leaderboards.json')
username = players(50, 'live_bullet')


#####################################################################################################
# Section 4
'''
** If zip file contains player data this section need not be run (other than for grading). **

Data Collection: Save game records for 'x' amount of players specified, for specified time frame (e.g. 2007 - 2020),
to appropriate files. Each month is a different API and as such must be merged together to create a seperate all
encompassing file for each player. If a player doesn't have game records for a certain year, it is simply saved as
an empty list within the final merged dictionary.
'''


years = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', 
         '2014', '2015', '2016', '2017', '2018', '2019', '2020'] #Years to be looped through (Chess.com founded in 2007).

for i in username: #Runs through all users.
                
        print('Creating game archive file for', i, '.') #Output for the user to monitor progress.
        
        file_name = '{username}_games.json'.format(username=i) #Creates a file for each player.
        create_file(file_name)
        
        
        d3 = {}
        
        
        for j in years:          #Runs through all years preselected.
            
            print('Year:', j)    #Output for the user to monitor progress.
            
            if j == '2020':      #Exception for 3 months of 2020, can be changed a more months elapse.
                months = ['01', '02', '03']

            else:
                months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

            for k in months: #Runs through all months in the year selected.
                
                #Url is filled out as required and a function was created to convert it into useable json data.
                url = 'https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}'.format(username=i, YYYY=j, MM=k)
                games = url_prep(url)

                merge_dict(d3, games)   #Uses function to add multiple months of game data together 
                                        #(merging information from multiple APIs).

        with open(file_name, 'a') as f: #Opens created file.
             json.dump(d3, f)           #Saves merged monthly dictionaries into it, as this is done on a loop, all data will 
                                        #be saved within one final merged dictionary

        print('Game archive file created for {user}.\n'.format(user=i)) #Output for the user to monitor progress.

##Code that was used to measure time of code execution:

#start = timeit.default_timer() #Placed at start of code
# stop = timeit.default_timer() #Placed at end of code
# execution_time = stop - start
# print("Program Executed in ", execution_time, 'seconds')

# ~ 2.5+ hrs to run this section for top 50 players


#####################################################################################################
# Section 5
'''
Data Preparation and Analysis. Run through player files already created, counting number of games,
number of bullet games and change in rating (and appending). Once fully completed, a graph displaying
accumulative games vs change in rating is produced and a logarithmic fit overlayed.

Exception cases have been made to filter out players that don't fit the criteria of this investigation. 
Inactive players that don't have very many games and so aren't representative of players learning from Chess.com. 
Also players that joined the site already high rated have been filtered as they don't show any growth 
related to the website and therefore skew the data from looking at players that have evolved from lower 
ratings through the site.
'''

rank = 0

total_rating_dif = []
total_accum_bullet = []
rank_list = []

#Lists for graphing, to investigate the relationship between a player's rank and number of games played.
final_rating = []
final_games = []
final_games_bull = []

#Two lines of code for colour gradient used in final plot.
n = len(username)
colors = plt.cm.autumn(np.linspace(0,1,n))

for i in username: #Runs through all players.

    rank = rank + 1 #Keeps track of each player's rank from 1 - 50.
    
    game_data = open_file('{username}_games.json'.format(username=i)) #Opens each player file to be used.
    
    num_games = 0
    num_bullet_games = 0
    rating = []
    
    accum_bullet = []
    rating_dif = []

    joined_high_rated = False #Players are assumed to not have joined the website high rated until data says otherwise.

    for month in game_data['games']: #Goes through months.

        num_games = num_games + len(month)
        
        if joined_high_rated == False:
            #Will change to 'True' if a player rating takes a huge jump,
            #this only happens when a player joins and starts at a rating 
            #much lower than there actual rating.

            for game in month: #Goes through games in month.
                

                if game['time_class'] == 'bullet'and game['rules'] == 'chess':
                    #Searches all games that fit the bullet time frame (60 seconds for each player)
                    #and follow the standard chess rules.

                    num_bullet_games = num_bullet_games + 1 #Keeps count of the number of bullet games played.


                    #if/elif to check whether the player was black or white and append their rating accordingly.
                    if (game['white']['username']).lower() == i:
                        rating.append(game['white']['rating'])

                    elif (game['black']['username']).lower() == i:
                        rating.append(game['black']['rating'])


                    #if/elif/else to append the change in rating from one game to the next
                    #and keep count of the number og games accumulated each time.
                    if len(rating) < 1:
                        pass

                    elif len(rating) == 1:

                        rating_dif.append(0)
                        accum_bullet.append(num_bullet_games)

                    else:

                        rating_dif.append(rating[-1]-rating[0])
                        accum_bullet.append(num_bullet_games)

                        dif = rating[-1]-rating[-2]

                        #A rating >200 suggests that player has beaten another player much higher rated than themselves on Chess.com,
                        #this is highly improbable in chess and indicates that the player is higher rated than their rating suggests
                        #i.e. they were already high-rated when they joined Chess.com and this rating growth is fictitious.
                        if dif > 200:
                            joined_high_rated = True


        #This will prevent the code from performing needless processing of data once a player 
        #is found to have joined the site at an already high rating.
        else:
            pass
    

    #This will append data for those players that have been found to play an active amount of games on the site
    #and who joined at a representative rating.
    if num_bullet_games > 1300 and joined_high_rated == False:
        
        final_games.append(num_games-num_bullet_games)
        final_games_bull.append(num_bullet_games)
        final_rating.append(rating[-1]-rating[0])
        
                        
        total_rating_dif = total_rating_dif + rating_dif
        total_accum_bullet = total_accum_bullet + accum_bullet
        
        #Code to show each player's general stats as they're processed
        print('\n------\nChess.com Information on:', i, '\nRanked number:', rank, '\nNumber of games played =', 
              num_games, '\nNumber of bullet games played =', num_bullet_games, '\n------\n')

        #Scatter plot to present the total relationship between accumulated games and rating change for a portion of
        #the top 50 bullet chess players on Chess.com.
        #Each player's data receives a colour from a gradient based on their overall rank.
        plt.title('Change in rating vs number of accumulative games')
        plt.scatter(accum_bullet, rating_dif, marker='.', color=colors[rank])
        plt.xlabel ('Accumulative Number of Bullet Games Played')
        plt.ylabel ('Change in Rating')
        
    
    elif joined_high_rated == True:
        print('\n------\nUser:', i, '\nRanked Number:', rank, '\nJoined Chess.com with High Rating\n------\n')
    
    else:
        print('\n------\nUser:', i, '\nRanked Number:', rank, '\nNot Active Enough\n------\n')

#Logarithmic fit of data
x = np.sort(total_accum_bullet)
popt, pcov = curve_fit(func, total_accum_bullet, total_rating_dif)
fit_y = func(x, *popt)

plt.plot(x, fit_y, label="Fitted Curve", color='b')
plt.show()


#####################################################################################################
# Section 6
'''
Further analysis. Exploring any relationship between the number of a player's games (bullet or otherwise)
and the effect ths has on their rating change, in addition, the rank of a player is outlined in a 
colour gradient and used to observe if this has an effect on the number of games played.
'''

#Function representing the logarithmic fit equation.
print('The logarithmic fit equation: y = ({:.2f}).ln(x) + {:.2f}'.format(popt[0], popt[1]))

#Zoomed in graph of the logarithmic function, showing the a close up of the area of greatest improvement.
plt.title('Logarithmic fit of rating improvement with games played')
plt.plot(x, fit_y, label="Fitted Curve", color='r')
plt.xlabel ('Accumulative Number of Bullet Games Played')
plt.ylabel ('Change in Rating')
plt.axis([-100,3000,400,1400])
plt.show()


#Colour gradient to show higher rated players in red and lower rated players in yellow
colors1 = plt.cm.autumn(np.linspace(0,1,len(final_games_bull)))
colors2 = plt.cm.autumn(np.linspace(0,1,len(final_games)))

j=0
plt.title('Bullet games played vs change in rating')
for i,k in zip(final_games_bull, final_rating):
    plt.scatter(i, k, marker='o', c=colors1[j])
    j =j+1
plt.xlabel ('Number of Bullet Games Played')
plt.ylabel ('Change in Rating')
plt.show()

j=0
plt.title('Total games played outside of bullet vs change in rating')
for i,k in zip(final_games, final_rating):
    plt.scatter(i, k, marker='o', c=colors2[j])
    j =j+1
plt.xlabel ('Number of Total Games Played')
plt.ylabel ('Change in Rating')
plt.show()

