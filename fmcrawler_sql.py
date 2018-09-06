import os
import sqlite3
from pprint import pprint

import numpy as np
from collections import defaultdict

import fightmetric as fm

import hashlib
import time
from datetime import datetime


def init_db(dbfile='fighterdb.sqlite'):
    '''
    Function to initialise the database. This should be called the first
    time you run the crawler, or whenever you want to create a fresh database.

    Parameters
    ----------
    dbfile : string (optional)
    	Name of the database file.

    Returns
    -------
    Nothing.
    '''

    conn = sqlite3.connect(dbfile, timeout=10)
    cur = conn.cursor()

    cur.executescript('''
    DROP TABLE IF EXISTS Fighters;
    DROP TABLE IF EXISTS Fight;
    DROP TABLE IF EXISTS FighterURLs;
    DROP TABLE IF EXISTS Round;

    CREATE TABLE Fighters (
    id		INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name	TEXT UNIQUE,
    url     TEXT UNIQUE,  
    weight	REAL,
    height	REAL,
    slpm    REAL,
    stance	TEXT,
    sapm 	REAL,
    dob		TEXT,
    subavg	REAL,
    reach	REAL,
    tdacc	REAL,
    tddef	REAL,
    tdavg	REAL,
    stracc	REAL,
    strdef	REAL,
    wins	INTEGER,
    losses	INTEGER,
    cumtime	REAL
    );

    CREATE TABLE FighterURLs (
    id		INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    url		TEXT UNIQUE, 
    fighter_id  INTEGER UNIQUE,
    processed   INTEGER
    );

    CREATE TABLE Fights (
    id		INTEGER NOT NULL PRIMARY KEY UNIQUE,
    fighter1 	TEXT,
    fighter2 	TEXT,
    event	TEXT,
    date    TIMESTAMP,
    method	TEXT,
    pass1	REAL,
    pass2 	REAL,   
    round 	INTEGER,
    str1 	REAL,
    str2	REAL,
    sub1	REAL,
    sub2	REAL,
    td1 	REAL,
    td2		REAL,
    time	REAL,
    winner 	TEXT
    );
    
    CREATE TABLE Round (
    id		INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    round_number INTEGER,
    fightid     INTEGER,
    fighter1 	TEXT,
    fighter2 	TEXT,
    sigstr1     REAL,
    sigstr2     REAL,
    sigstrpt1   REAL,
    sigstrpt2   REAL,
    head1       REAL,
    head2       REAL,
    headpt1     REAL,
    headpt2     REAL,
    body1       REAL,
    body2       REAL,
    bodypt1     REAL,
    bodypt2     REAL,
    leg1        REAL,
    leg2        REAL,
    legpt1      REAL,
    legpt2      REAL, 
    distance1   REAL,
    distance2   REAL,
    distancept1 REAL,
    distancept2 REAL,
    clinch1     REAL,
    clinch2     REAL,
    clinchpt1   REAL,
    clinchpt2   REAL,
    ground1     REAL,
    ground2     REAL,
    groundpt1   REAL,
    groundpt2   REAL,
    tstr1       REAL,
    tstr2       REAL,
    td1         REAL,
    td2         REAL,
    tdpt1       REAL,
    tdpt2       REAL,
    subatt1     REAL,
    subatt2     REAL,
    pass1       REAL,
    pass2       REAL,
    rev1        REAL,
    rev2        REAL,
    FOREIGN KEY(fightid) REFERENCES Fight(id)
    );
    ''')

    conn.commit()

    conn.close()


def crawl_test(dbfile='mma_db.sqlite'):
    if dbfile not in os.listdir('./'):
        print("Database not found; initialising new database.")
        init_db(dbfile)

    # init sqlite stuff
    conn = sqlite3.connect(dbfile, timeout=10)

    cur = conn.cursor()

    urls = []
    with open('urls.txt', 'r') as f:
        lines = f.readlines()
    urls = [l.strip() for l in lines]

    write_page_to_database(urls[1], cur)

    conn.commit()


def crawl(initFighter='Mark Hunt', dbfile='fighterdb.sqlite', K=2):
    '''
     Basic Fightmetric crawler; will get URLs for all fighter profiles on Fightmetric
    
    Parameters
    ----------
    initFighter : str (optional)
        Name of the fighter to start the crawl. The URL will be fetched from Google.
    	Default is either a random fighter from existing data, or Mark Hunt if no
    	data exists.
    dbfile : str (optional)
	Name of the database file
    K : int (optional)
    	Degrees of separation to include. Default is K=2, meaning that the crawler
    	will parse initFighter (1), and the fighters on initFighter's page (2).

    Returns
    -------
    fighters : dict
    	Each entry corresponds to a fighter (with the fighter's name as key).
    	The dict contains fighter stats and all the fighter's fights

    '''

    if dbfile not in os.listdir('./'):
        print("Database not found; initialising new database.")
        init_db()

    # init sqlite stuff
    conn = sqlite3.connect(dbfile, timeout=10)

    cur = conn.cursor()

    # Create the base of the tree
    initFighterURL = fm.get_url(initFighter)[11:]

    write_page_to_database(initFighterURL, cur)

    conn.commit()

    fighterURLs = get_url_list(cur)

    for k in range(K):

        for fighterURL in fighterURLs:
            pauseInterval = np.random.rand() + 0.5

            # pause for some random time interval
            os.system('sleep %.2f' % pauseInterval)

            print('Running fighter: %s' % fighterURL)

            write_page_to_database(fighterURL, cur)

            conn.commit()

        fighterURLs = get_url_list(cur)

    return fighterURLs


def get_url_list(cur):
    '''
    Returns a list of all URLs which have been added to the database,
    but which still haven't been processed.
    
    Parameters
    ----------
    cur : a sqlite db cursor
    
    Returns
    -------
    fighterURLs : a list of URLs to Fightermetric web pages.
    

    '''

    cur.execute(''' SELECT url FROM FighterURLs WHERE processed = 0 ''')

    fighterURLs = [k[0] for k in cur.fetchall()]

    return fighterURLs


def add_to_url_list(fighterURLs, processed, cur):
    ''' Adds an entry into the FighterURLs database which lets us know
    whether or not this person's page has been processed.

    
    Parameters
    ----------
    fighterURLs : list of strings
    	a list of URLs for new fighters

    processed : int
    	0 or 1 denoting whether this page has been processed

    cur : sqlite3 cursor
    	Cursor pointing to the database.

    Returns
    -------
    Nothing.
    

    '''
    if processed:
        sqlExpression = '''INSERT OR REPLACE INTO FighterURLs (url,processed)
    		VALUES ( ?, ? )'''
    else:
        sqlExpression = '''INSERT OR IGNORE INTO FighterURLs (url,processed)
    		VALUES ( ?, ? )'''

    # convert to list if (presumably) a string is given
    if type(fighterURLs) == str: fighterURLs = set([fighterURLs])

    for fighterURL in fighterURLs:
        cur.execute(sqlExpression, (fighterURL, processed))


def write_fights_to_db(fights, cur):
    total_metrics = ['fighter', 'kd', 'sigstr', 'sigstrpt', 'tsrt', 'td', 'tdpt', 'subaat', 'pass', 'rev']
    sig_metrics = ['fighter', 'sigstr', 'sigstrpt', 'head', 'body', 'leg', 'distance', 'clinch', 'ground']

    for fight in fights:
        rounds = len(fight['Totals']['Fighter_0'])
        for r in range(1, rounds):
            table = 'Round'

            fightId = int(hashlib.sha1(fight['url'].encode()).hexdigest(), 16) % (10 ** 16)

            # Do a quick check
            cur.execute('SELECT fighter1,fighter2 FROM Fights WHERE id == ?', (fightId,))
            matches = cur.fetchall()
            if matches is None:
                raise AssertionError('There is no fight registered with that id')

            query = 'INSERT OR IGNORE INTO ' + table + ''' (fightId, round_number, fighter1, fighter2, sigstr1, sigstr2, sigstrpt1,
             sigstrpt2, head1, head2, headpt1, headpt2, body1, body2, bodypt1, bodypt2, leg1, leg2, legpt1, legpt2,
             distance1, distance2, distancept1, distancept2, clinch1, clinch2, clinchpt1, clinchpt2, ground1, ground2,
             groundpt1, groundpt2, tstr1, tstr2, td1, td2, tdpt1, tdpt2, subatt1, subatt2, pass1, pass2, rev1, rev2)
             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''

            round = r
            data = (fightId,
                    r,
                    fight['Totals']['Fighter_0'][round]['fighter1'],
                    fight['Totals']['Fighter_1'][round]['fighter2'],
                    fight['Totals']['Fighter_0'][round]['sigstr1'],
                    fight['Totals']['Fighter_1'][round]['sigstr2'],
                    fight['Totals']['Fighter_0'][round]['sigstrpt1'],
                    fight['Totals']['Fighter_1'][round]['sigstrpt2'],
                    fight['Significant Strikes']['Fighter_0'][round]['head1'],
                    fight['Significant Strikes']['Fighter_1'][round]['head2'],
                    fight['Significant Strikes']['Fighter_0'][round]['headpt1'],
                    fight['Significant Strikes']['Fighter_1'][round]['headpt2'],
                    fight['Significant Strikes']['Fighter_0'][round]['body1'],
                    fight['Significant Strikes']['Fighter_1'][round]['body2'],
                    fight['Significant Strikes']['Fighter_0'][round]['bodypt1'],
                    fight['Significant Strikes']['Fighter_1'][round]['bodypt2'],
                    fight['Significant Strikes']['Fighter_0'][round]['leg1'],
                    fight['Significant Strikes']['Fighter_1'][round]['leg2'],
                    fight['Significant Strikes']['Fighter_0'][round]['legpt1'],
                    fight['Significant Strikes']['Fighter_1'][round]['legpt2'],
                    fight['Significant Strikes']['Fighter_0'][round]['distance1'],
                    fight['Significant Strikes']['Fighter_1'][round]['distance2'],
                    fight['Significant Strikes']['Fighter_0'][round]['distancept1'],
                    fight['Significant Strikes']['Fighter_1'][round]['distancept2'],
                    fight['Significant Strikes']['Fighter_0'][round]['clinch1'],
                    fight['Significant Strikes']['Fighter_1'][round]['clinch2'],
                    fight['Significant Strikes']['Fighter_0'][round]['clinchpt1'],
                    fight['Significant Strikes']['Fighter_1'][round]['clinchpt2'],
                    fight['Significant Strikes']['Fighter_0'][round]['ground1'],
                    fight['Significant Strikes']['Fighter_1'][round]['ground2'],
                    fight['Significant Strikes']['Fighter_0'][round]['groundpt1'],
                    fight['Significant Strikes']['Fighter_1'][round]['groundpt2'],
                    fight['Totals']['Fighter_0'][round]['tsrt1'],
                    fight['Totals']['Fighter_1'][round]['tsrt2'],
                    fight['Totals']['Fighter_0'][round]['td1'],
                    fight['Totals']['Fighter_1'][round]['td2'],
                    fight['Totals']['Fighter_0'][round]['tdpt1'],
                    fight['Totals']['Fighter_1'][round]['tdpt2'],
                    fight['Totals']['Fighter_0'][round]['subatt1'],
                    fight['Totals']['Fighter_1'][round]['subatt2'],
                    fight['Totals']['Fighter_0'][round]['pass1'],
                    fight['Totals']['Fighter_1'][round]['pass2'],
                    fight['Totals']['Fighter_0'][round]['rev1'],
                    fight['Totals']['Fighter_1'][round]['rev2'],
                    )

            cur.execute(query, data)


def write_fights_to_database(fights, cur):
    ''' Writes list of fight dicts to the Fights table of the database.

    Parameters
    ----------
    fights : list
    	A list of dicts, with each dict giving details of a given fight.

    cur : sqlite3 cursor
	Cursor pointing to the database.

    Returns
    -------
    Nothing.

    '''

    for fight in fights:
        fightId = int(hashlib.sha1(fight['url'].encode()).hexdigest(), 16) % (10 ** 16)

        if fight['outcome'] == 'win':
            winner = fight['Fighter'][0]
        elif fight['outcome'] == 'loss':
            winner = fight['Fighter'][1]
        else:
            winner = 'Draw'

        # Do a quick check
        cur.execute('SELECT fighter1,fighter2 FROM Fights WHERE id == ?', (fightId,))
        matches = cur.fetchall()
        for match in matches:
            if (sorted(match) != sorted(fight['Fighter'])):
                raise AssertionError('Error: Fighters and fight id should match. Probably' + \
                                     ' means overlapping ids.')

        datetime_object = datetime.strptime(fight['Event'][1], '%b.%d,%Y').strftime('%Y%m%d')
        cur.execute(
            '''INSERT OR IGNORE INTO Fights (id, fighter1, fighter2,
            event, date, method, pass1, pass2, round, str1, str2, sub1, sub2,
    	    td1, td2, time, winner) VALUES ( ?, ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? )''',
            (fightId, fight['Fighter'][0], fight['Fighter'][1], \
             fight['Event'][0], datetime_object, fight['Method'], \
             fight['Pass'][0], fight['Pass'][1], fight['Round'], \
             fight['Str'][0], fight['Str'][1], fight['Sub'][0], \
             fight['Sub'][1], fight['Td'][0], fight['Td'][1], \
             fight['Time'], winner))


def write_fighter_to_database(stats, urls, cur):
    ''' Writes a fighter's stats to the database.

    Parameters
    ----------
    stats : dict
    	Contains stats for the fighter; obtained from parsing the Fightmetric page.

    urls : list
    	List of urls on that fighter's page. This gets added to a separate table.

    cur : sqlite3 cursor
	Cursor pointing to the database.

    Returns
    -------
    Nothing

    '''

    key2Sql = {key: strip_key(key) for key in stats.keys() if key is not 'fights'}

    fighterURL = stats['url']

    sqlExpression = 'INSERT OR REPLACE INTO Fighters ( '

    for i, k in enumerate(key2Sql):
        if i == (len(key2Sql) - 1):
            delim = ''
        else:
            delim = ', '

        sqlExpression += key2Sql[k] + delim

    sqlExpression += ') VALUES ( ' + '?, ' * (len(key2Sql) - 1) + '? )'

    dataTuple = tuple([stats[key] for key in key2Sql])

    cur.execute(sqlExpression, dataTuple)

    cur.execute(''' SELECT id FROM Fighters WHERE name = ? ''', (stats['Name'],))

    fighter_id = cur.fetchone()[0]

    cur.execute('''INSERT OR IGNORE INTO FighterURLs (url,fighter_id)
    		    VALUES ( ?, ? )''', (fighterURL, fighter_id))

    add_to_url_list(urls, 0, cur)

    add_to_url_list([fighterURL], 1, cur)


def write_page_to_database(fighterURL, cur):
    ''' This is a convenient wrapper for write_fighter_to_database and
    write_fights_to_database.

    Parameters
    ----------
    fighterURL : str
    	A valid Fightmetric.com URL for a fighter's profile page.

    cur : sqlite3 cursor
	A cursor pointing to the database.

    Returns
    -------
    Nothing

    '''

    fighterPage = fm.get_page(fighterURL)

    if fighterPage == ['Empty page']: return None

    stats, urls, fight_urls = fm.parse_page(fighterPage)

    fights = stats.pop('Fights')

    stats['url'] = fighterURL

    stats['wins'] = compute_wins(fights)

    stats['losses'] = compute_losses(fights)

    stats['cumtime'] = compute_cumtime(fights)

    write_fighter_to_database(stats, urls, cur)

    write_fights_to_database(fights, cur)

    write_fights_to_db(stats['fights'], cur)

def compute_wins(fights):
    y = np.sum([fight['outcome'] == 'win' for fight in fights])
    return y


def compute_losses(fights):
    y = np.sum([fight['outcome'] == 'loss' for fight in fights])
    return y


def compute_cumtime(fights):
    y = np.sum([fight['Time'] for fight in fights])
    return y


def strip_key(mykey):
    ''' Strips the string of periods and white space and converts to lower case.

    Parameters
    ----------
    mykey : str
    
    Returns
    -------
    newkey : str
    	A version of the input string with commas, white space, and capitalisation removed.

    '''

    newkey = mykey.replace('.', '').replace(' ', '').lower()

    return newkey


def populate_db(dbfile='mma_db.sqlite'):
    try:
        if dbfile not in os.listdir('./'):
            print("Database not found; initialising new database.")
            init_db(dbfile)

        # init sqlite stuff
        conn = sqlite3.connect(dbfile, timeout=10)

        cur = conn.cursor()

        with open('urls.txt', 'r') as f:
            lines = f.readlines()
        urls = [l.strip() for l in lines]
        for i, url in enumerate(urls):
            print('Starting to process url # %d,  %s' % (i, url))
            write_page_to_database(url, cur)

        conn.commit()
        conn.close()
    except KeyboardInterrupt:
        conn.commit()
        conn.close()


def get_rounds(dbfile='mma_db.sqlite'):
    # takes ~ 17s for 22589 rounds
    conn = sqlite3.connect(dbfile, timeout=10)

    cur = conn.cursor()

    cur.execute('SELECT id from Fights')

    ids = cur.fetchall()

    round_list = []

    for id in ids:
        cur.execute('SELECT * FROM Round WHERE fightid=(?) ORDER BY round_number', id)
        rounds = cur.fetchall()
        f1, f2 = [], []
        done = []

        for r in rounds:
            if r[1] in done:
                continue
            done.append(r[1])
            l1, l2 = [], []
            for i in range(5, len(r)):
                if i % 2 == 1:
                    l1.append(r[i])
                else:
                    l2.append(r[i])

            total_1 = np.subtract(l1, l2)
            total_2 = np.multiply(-1, total_1)

            f1.append(total_1.tolist())
            f2.append(total_2.tolist())

        if f1 != []:
            while len(f1) < 5:
                f1.append([0] * 20)
                f2.append([0] * 20)
            if len(f1) == 5:
                round_list.append(np.asarray(f1, dtype=np.float32))
            if len(f2) == 5:
                round_list.append(np.asarray(f2, dtype=np.float32))

    return np.asarray(round_list)


def get_fighters(dbfile='mma_db.sqlite'):
    conn = sqlite3.connect(dbfile, timeout=10)

    cur = conn.cursor()

    cur.execute('SELECT name, weight, height, reach, stance FROM Fighters')

    fighters = cur.fetchall()

    fighters_stats = {fighter[0]: list(fighter[1:5]) for fighter in fighters}

    for k in fighters_stats.keys():
        # pprint(fighters_stats[k])
        s = fighters_stats[k][3]
        fighters_stats[k] = fighters_stats[k][:3]
    #     fighters_stats[k].extend(stance_to_vector(s))

    cur.execute('SELECT * from Fights ORDER BY DATE')

    fights = cur.fetchall()

    fighter_dict = defaultdict(list)

    for fight in fights:
        cur.execute('SELECT * FROM Round WHERE fightid=' + str(fight[0]) + ' ORDER BY round_number')
        rounds = cur.fetchall()
        # f1, f2 = defaultdict(lambda: {'fight': [], 'winner': None, 'method': None, 'opponent': None, 'date': None}),\
        #          defaultdict(lambda: {'fight': [], 'winner': None, 'method': None, 'opponent': None, 'date': None})

        f1, f2 = {}, {}

        f1['fight'] = []
        f2['fight'] = []

        method = fight[5]
        if 'KO' in method:
            method = 'KO'
        elif 'SUB' in method:
            method = 'SUB'
        f1['method'] = method
        f2['method'] = method

        winner = fight[16]
        f1['winner'] = winner
        f2['winner'] = winner

        date = fight[4]
        f1['date'] = date
        f2['date'] = date

        done = []

        fighter1, fighter2 = '', ''

        for r in rounds:
            fighter1, fighter2 = r[3], r[4]
            if r[1] in done:
                continue
            done.append(r[1])
            l1, l2 = [], []
            for i in range(5, len(r)):
                if i % 2 == 1:
                    l1.append(r[i])
                else:
                    l2.append(r[i])

            # Stricking ratio
            try:
                strRatio1 = l1[0]/l2[0]
            except ZeroDivisionError:
                strRatio1 = l1[0]
            try:
                strRatio2 = l2[0]/l1[0]
            except ZeroDivisionError:
                strRatio2 = l2[0]

            l1.append(strRatio1)
            l2.append(strRatio2)

            # Strike diffential per Minute
            strDiff1 = (l1[0] - l2[0])/3
            strDiff2 = (l2[0] - l1[0])/3

            l1.append(strDiff1)
            l2.append(strDiff2)

            # total_1 = np.subtract(l1, l2)
            # total_2 = np.multiply(-1, total_1)

            if len(l1) != 22:
                print(len(l1))
                raise Exception

            if len(l2) != 22:
                print(len(l2))
                raise Exception

            if l1 != []:
                f1['fight'].append([l1, l2])
                f2['fight'].append([l2, l1])

                # f1[fighter1].append(total_1.tolist())
                # f2[fighter2].append(total_2.tolist())

        f1['opponent'] = fighter2
        f2['opponnent'] = fighter1

        f1['fight_stats'] = fight
        f2['fight_stats'] = fight

        try:
            f1['fighter_stats'] = fighters_stats[fighter1]
            f2['fighter_stats'] = fighters_stats[fighter2]
        except:
            continue

        if len(f1['fight']) != 0:
            while len(f1['fight']) < 5:
                f1['fight'].append([[0] * 22, [0] * 22])
                f2['fight'].append([[0] * 22, [0] * 22])

            # fighter_dict[fighter1].append((np.asarray(f1[fighter1], dtype=np.float32), winner, method, fighter2))
            # fighter_dict[fighter2].append((np.asarray(f2[fighter2], dtype=np.float32), winner, method, fighter1))

            # f1['stats'] = cum_stat_from_data(fight, f1['fight'], f2['fight'], True)
            # f2['stats'] = cum_stat_from_data(fight, f2['fight'], f1['fight'], False)

            fighter_dict[fighter1].append(f1)
            fighter_dict[fighter2].append(f2)

    # for k,v in fighter_dict.items():
    #     stats = cum_stat_from_data(k, v)

    for k,v in fighter_dict.items():
        fighter_dict[k] = np.asarray(v)

    return fighter_dict


def stance_to_vector(stance):
    if stance == '':
        s = [0, 0, 0, 0, 0]
    elif stance == 'Open Stance':
        s = [1, 0, 0, 0, 0]
    elif stance == 'Orthodox':
        s = [0, 1, 0, 0, 0]
    elif stance == 'Switch':
        s = [0, 0, 1, 0, 0]
    elif stance == 'Southpaw':
        s = [0, 0, 0, 1, 0]
    elif stance == 'Sideways':
        s = [0, 0, 0, 0, 1]
    else:
        raise ValueError('Stance %s does not exist' % stance)

    return s


def get_fighter_stats(dbfile='mma_db.sqlite'):
    conn = sqlite3.connect(dbfile, timeout=10)

    cur = conn.cursor()

    stats = {}

    cur.execute('SELECT name, weight, height, slpm, stance, sapm, subavg, reach, tdacc, tddef, tdavg, stracc, strdef '
                'FROM Fighters')

    fighters = cur.fetchall()

    for fighter in fighters:
        assert len(fighter) == 13
        fighter = list(fighter)
        # Chnage stance to one-hot vector
        stance = fighter[4]
        if stance == '':
            s = [0, 0, 0, 0, 0]
        elif stance == 'Open Stance':
            s = [1, 0, 0, 0, 0]
        elif stance == 'Orthodox':
            s = [0, 1, 0, 0, 0]
        elif stance == 'Switch':
            s = [0, 0, 1, 0, 0]
        elif stance == 'Southpaw':
            s = [0, 0, 0, 1, 0]
        elif stance == 'Sideways':
            s = [0, 0, 0, 0, 1]
        else:
            raise ValueError('Stance %s does not exist' % stance)
        stats[fighter[0]] = np.asarray(fighter[1:4] + s + fighter[5:]).flatten()

    maximum = np.max(list(stats.values()), axis=0)
    minimum = np.min(list(stats.values()), axis=0)

    for k,v in stats.items():
        stats[k] = np.divide(np.subtract(v, minimum), np.subtract(maximum, minimum))

    return stats


if __name__ == "__main__":
    # initFighters = ['Demetrious Johnson',
    #                 'TJ Dillashaw',
    #                 'Jose Aldo', 'Conor McGregor',
    #                 'Rafael dos Anjos',
    #                 'Donald Cerrone',
    #                 'Robert Whittaker', 'Anderson Silva',
    #                 'Jon Jones', 'Alexander Gustafsson',
    #                 'Mark Hunt', 'Stipe Miocic']
    #
    # for fighter in initFighters:
    #     print('Crawling using %s as root' % fighter)
    #     crawl(initFighter=fighter, K=4)
    # init_db('mma_db.sqlite')
    # crawl_test()
    # round_dict = get_rounds()
    # for v in round_dict[:10]:
    #     print(v.shape)
    # populate_db()
    fighters = get_fighters()
    pprint(fighters)