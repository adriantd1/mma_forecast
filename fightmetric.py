import importlib.util
import re
import urllib.request
from pprint import pprint
from time import sleep
import matplotlib
import time
import traceback

matplotlib.use('Agg')

from matplotlib import pyplot as plt


spec = importlib.util.spec_from_file_location('google',
                                              '/home/ml/atousi2/anaconda3/envs/atousi/lib/python3.6/site-packages/googlesearch/__init__.py')
# google = importlib.import_module('/home/ml/atousi2/anaconda3/envs/atousi/lib/python3.6/site-packages/googlesearch/__init__.py')
google = importlib.util.module_from_spec(spec)
spec.loader.exec_module(google)


def get_url(name):
    '''
    Searches Google for a specified name and searches fightmetric.com
    for the fighter profile.

    Parameters
    ----------
    name : str
    	Name of fighter

    Returns
    -------
    S : str
    	URL for the fighter's Fightmetric profile

    '''

    site = "http://fightmetric.com"

    query = '"' + name + '"' + ' site:' + site
    k = 5
    mysearch = google.search(query, stop=20, num=k + 1)

    for k in range(k):
        S = next(mysearch)

        if 'fighter-details' in S:
            return S

    return 'NaN'


def get_page(url):
    sleep(0.2)
    if 'fightmetric' not in url:
        url = 'http://fightmetric.com/' + url

    if 'http://' not in url:
        url = 'http://' + url

    n_attempts = 3

    page = ['Empty page']

    # http = urllib3.PoolManager()

    for k in range(n_attempts):
        try:
            url_generator = urllib.request.urlopen(url, timeout=5)
            page = url_generator.readlines()
            break
        except IOError as e:
            if k == n_attempts: return ['Empty page']
    p = []
    for l in page:
        if isinstance(l, str):
            p.append(l)
        else:
            p.append(l.decode())
    return p


def parse_page(page):
    '''
    Parses a Fighter's profile page, obtains stats and a list of fights

    Parameters
    ----------
    page : list
    	Fighter profile page obtained by calling get_page.

    Returns
    -------
    fighter_stats : dict
    	Contains all stats for the fighter, including list of fights.
    
    urls : list
    	List of all the URLs on the fighter's profile.

    '''
    fighter_stats = parse_stats(page)
    fights = parse_fights(page)

    fighter_stats['Fights'] = fights
    try:
        if len(fights) == 0:
            name = 'Unknown fighter'
        else:
            name = fights[0]['Fighter'][0]

    except IndexError:
        import pdb;
        pdb.set_trace()

    fighter_stats['Name'] = name

    urls = get_fighter_urls(page)

    fight_urls = get_fight_urls(page)

    fights_data = []
    for url in fight_urls:
        parsed_data = parse_fight_page(get_page(url), url)
        if parsed_data is not None:
            fights_data.append(parsed_data)

    fighter_stats['fights'] = fights_data

    return fighter_stats, urls, fight_urls


def parse_stats(page):
    '''
    Parse the stats on a fighter's Fightmetric page
    
    Parameters
    ----------
    page : list
    	A fighter's Fightmetric profile page (provided by get_page)

    Returns
    -------
    fighter_stats : dict
    	dict of statistics for fighter (see Fightmetric.com for explanation)

    '''
    metrics = ['Height', 'Weight', 'Reach', 'STANCE', 'DOB', 'SLpM', \
               'Str. Acc.', 'SApM', 'Str. Def', 'TD Avg.', 'TD Acc.', \
               'TD Def.', 'Sub. Avg.']

    fighter_stats = {}
    for metric in metrics:
        if metric == 'DOB' or metric == 'SLpM':
            k = 3
        else:
            k = 2

        raw_val = [page[i + k] for i, p in enumerate(page) if metric + ':' in p][0]

        val_str = raw_val.strip()

        if '%' in val_str:
            val = percent_to_prop(val_str)

        elif metric == 'Height':
            val = ft_to_cm(val_str)

        elif metric == 'Weight':
            val = lbs_to_kg(val_str)

        elif metric == 'Reach':
            val = in_to_cm(val_str)
            if val == 0:
                val = fighter_stats.get('Height', 0)

        # elif metric == 'STANCE':
        #     if val == None or val == '':
        #         val = 'Orthodox'

        elif metric != 'DOB' and metric != 'STANCE':
            val = float(val_str)
        else:
            val = val_str

        fighter_stats[metric] = val

    return fighter_stats


def parse_fight_page(page, url):
    '''
    Parse a fight page
    :param page:
    :return:
    '''

    t_metrics = ['fighter', 'kd', 'sigstr', 'sigstrpt', 'tsrt', 'td', 'tdpt', 'subatt', 'pass', 'rev']
    sig_metrics = ['fighter', 'sigstr', 'sigstrpt', 'head', 'body', 'leg', 'distance', 'clinch', 'ground']

    Title = 0
    Overall = 1
    Rounds = 2
    Sig_title = 3
    Sig_round = 4

    state = -1

    sections = {}
    open_section = False
    cur_section = ''
    cur_title = ''

    columns = []

    row = 0
    cur_td = None
    open_td = False
    stat = [[], []]

    open_td = False  # this is if the table row has opened

    open_outcome = False  # this is if the fight outcome has been mentioned

    total_found = True

    open_th = False

    open_tr = False

    parsing_total = False

    total_metrics = []

    fighters = {}

    cur_fighter = True
    try:
        for i, p in enumerate(page):
            if state > 4:
                break
            if '<section' in p and 'fight-details' in p:
                # print(url)
                state = state + 1
                open_section = True
                cur_section = ''
                # print('Opening section, current state is %d' % state)

            if '</section' in p and open_section:
                # section is complete and we can parse
                open_section = False
                # print('Closing section %d' % state)

                if state == Title or state == Sig_title:
                    cur_section = strip_html(cur_section).strip()
                    sections[cur_section] = {'Metrics': [], 'Fighter_0': [], 'Fighter_1': []}
                    cur_title = cur_section
                    # print('Adding %s to sections' % cur_title)

                # if state == Overall or Sig_title:
                    # parse cur_column and stats
                    # [c for c in columns if 'Round' not in c]

            if open_section:
                cur_section += ' ' + p

            if '<th' in p and open_tr:
                # print('Opening th')
                open_th = True
                cur_th = ''

            if '</th' in p and open_th:
                # then the row is complete and we can parse
                open_th = False
                columns.append(strip_html(cur_th).strip())
                # print('Closing th with columns as')
                # print(columns)

            if open_th:
                cur_th += ' ' + p

            if '<tr' in p:
                # print('Opening tr')
                cur_td = ''
                open_tr = True

            if '</tr' in p:
                # print('Closing tr')
                open_tr = False
                if len(columns) != 0:
                    # Parse columns
                    # print('Parsing columns')
                    # pprint(columns)
                    sections[cur_title]['Metrics'] = columns
                    columns = []
                if len(stat[0]) != 0:
                    # print('Parsing stats')
                    # print(stat)
                    sections[cur_title]['Fighter_0'].append(stat[0])
                    sections[cur_title]['Fighter_1'].append(stat[1])
                    stat = [[], []]

            if '<td' in p and (state == Overall or state == Rounds or state == Sig_round):
                # print('Opening td')
                cur_td = ''
                open_td = True

            if '</td' in p and open_td and state == Overall:
                # print('Closing td')
                # stat[row].append(strip_html(cur_td).strip())
                # row = (row + 1) % 2
                open_td = False

            if '</p' in p and open_td:
                # print('Addind stats')
                # print(strip_html(cur_td).strip())
                if not 'Significant' in strip_html(cur_td).strip():
                    stat[row].append(strip_html(cur_td).strip())
                row = (row + 1) % 2
                cur_td = ''

            if open_td:
                cur_td += ' ' + p

        # Final formatting
        for k in ['Totals', 'Significant Strikes']:
            if k == 'Totals':
                metrics = t_metrics
            else:
                metrics = sig_metrics
            for fighter in ['Fighter_0', 'Fighter_1']:
                index = str(int(fighter[-1]) + 1)
                for r in range(len(sections[k][fighter])):
                    data = {}
                    for j in range(len(sections[k][fighter][r])):
                        data[metrics[j] + index] = sections[k][fighter][r][j]
                    if k == 'Totals':
                        sections[k][fighter][r] = format_total_data(data)
                    else:
                        sections[k][fighter][r] = format_sig_data(data)

        sections['url'] = url

    except Exception:
        if state == -1:
            print('Request timeout error for page: %s' % url)
        else:
            print('Error with page %s' % url)
        return None

    if 'Round-by-round stats not currently available.' in sections.keys():
        return None

    return sections


def parse_fights(page):
    '''
    Parse the fights on a fighter's Fightmetric page
    
    Parameters
    ----------
    page : list
    	A fighter's Fightmetric profile page (provided by get_page)

    Returns
    -------
    fights : list
    	Contains all the fights listed on the fighter's profile page

    '''
    ctr = 0
    fights = []
    open_td = False  # this is if the table row has opened
    open_outcome = False  # this is if the fight outcome has been mentioned

    open_th = False

    columns = []
    current_col = 'None'
    for p in page:
        if '<th' in p:
            open_th = True
            current_th = ''
        if '</th' in p and open_th:
            # then the row is complete and we can parse
            open_th = False
            current_col = strip_html(current_th).strip()
            columns.append(current_col)

        if open_th:
            current_th += ' ' + p

        if 'win<i' in p:
            current_fight = {'outcome': 'win'}
            current_fight['url'] = [i for i in p.split('"') if 'fight-details' in i][0].replace('http://', '')
            open_outcome = True

        if 'loss<i' in p:
            current_fight = {'outcome': 'loss'}
            current_fight['url'] = [i for i in p.split('"') if 'fight-details' in i][0].replace('http://', '')
            open_outcome = True

        if '<td' in p and open_outcome:
            ctr += 1
            current_td = ''
            open_td = True

        if '</td' in p and open_outcome and open_td:
            # then the row is complete and we can start parsing
            open_td = False

            current_col = columns[ctr]
            current_val = strip_html(current_td).strip()
            current_val = current_val.replace('\n', '')

            if current_col not in ['Method', 'Round', 'Time']:
                mid_idx = int(len(current_val) / 2.)
                prop1 = current_val[0:mid_idx].replace(' ', '')
                prop2 = current_val[mid_idx:].replace(' ', '')
                if current_col not in ['W/L', 'Fighter', 'Event']:
                    prop1 = float(prop1)
                    prop2 = float(prop2)

                props = [prop1, prop2]
                current_val = props
            else:
                current_val = current_val.replace(' ', '')

                if current_col == 'Time':
                    current_val = mins_to_sec(current_val)

                if current_col == 'Round':
                    current_val = float(current_val)

            current_fight[current_col] = current_val

        if open_td and open_outcome:
            current_td += ' ' + p

        # this signals the end of the current row
        if '</tr>' in p and ctr == len(columns) - 1:
            for i, name_cat in enumerate(current_fight['Fighter']):
                myre = re.findall('[a-zA-Z][A-Z0-9]', name_cat)

                name = name_cat.replace(myre[0], myre[0][0] + ' ' + myre[0][1])
                current_fight['Fighter'][i] = name

            fights.append(current_fight)
            open_outcome = False
            ctr = 0

    return fights


def get_fighter_urls(page):
    '''
    Returns all fighter URLs on a fighter's profile page

    Parameters
    ----------
    page : list
    	A fighter's profile page

    Returns
    -------
    url_list : set
	Set of fighter URLs contained on the profile page

    '''
    # This gets the html from a page whose link is given in S and 
    # fetches all the /fighter/FirstName-LastName-ID phrases on that page

    url_list_all = [find_url(k) for k in page]
    url_list_clean = [k for k in url_list_all if k != []]
    url_list = set([link for flist in url_list_all for link in flist])

    return url_list


def get_fight_urls(page):
    '''
    Returns all fighter URLs on a fighter's profile page

    Parameters
    ----------
    page : list
    	A fighter's profile page

    Returns
    -------
    url_list : set
	Set of fighter URLs contained on the profile page

    '''
    # This gets the html from a page whose link is given in S and
    # fetches all the /fighter/FirstName-LastName-ID phrases on that page

    url_list_all = [find_fight_url(k) for k in page]
    url_list_clean = [k for k in url_list_all if k != []]
    url_list = set([link for flist in url_list_all for link in flist])
    final_list = set()
    for l in url_list:
        if 'class' in l:
            if 'fight-details' in l:
                final_list.add(l.split('"')[0])
    return final_list


def find_url(S):
    '''
    Finds the fightmetric.com URLs in a string
    
    Parameters
    ----------
    S : string
    	
    urls_fx : list
    	list of URLs contained in S   
    '''
    urlregex = 'fightmetric.com/fighter-details/.*"'
    urls = re.findall(urlregex, S)
    urls_fx = [k[0:-1] for k in urls]

    return urls_fx


def find_fight_url(S):
    '''
        Finds the fightmetric.com URLs in a string

        Parameters
        ----------
        S : string

        urls_fx : list
        	list of URLs contained in S
        '''
    urlregex = 'fightmetric.com/fight-details/.*"'
    urls = re.findall(urlregex, S)
    urls_fx = [k[0:-1] for k in urls]

    return urls_fx


def strip_html(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


######################################
### These are conversion functions ###
######################################
def mins_to_sec(S):
    S2 = S.replace(':', ' ').split()
    mins = float(S2[0])
    secs = float(S2[1])

    total_secs = mins * 60 + secs
    return total_secs


def percent_to_prop(S):
    myreg = S.replace('%', '')
    return float(myreg) / 100.


def ft_to_cm(S):
    val_str = S.replace('"', '').replace("'", '').split()
    if '--' in val_str:
        return 0

    feet = float(val_str[0])
    inches = float(val_str[1])
    i2cm = 2.54
    f2cm = 30.48
    cm = feet * f2cm + inches * i2cm
    return cm


def in_to_cm(S):
    i2cm = 2.54
    val_str = S.replace('"', '')
    if val_str == '--':
        return 0
    inches = float(val_str)
    cm = inches * i2cm
    return cm


def lbs_to_kg(S):
    val_str = S.replace('lbs.', '')
    if val_str == '--':
        return 0
    lbs = float(val_str)
    kg_per_lb = 0.454
    kg = lbs * kg_per_lb

    return kg


def get_all_urls():
    initFighters = ['Demetrious Johnson',
                    'TJ Dillashaw',
                    'Jose Aldo', 'Conor McGregor',
                    'Rafael dos Anjos',
                    'Donald Cerrone',
                    'Robert Whittaker', 'Anderson Silva',
                    'Jon Jones', 'Alexander Gustafsson',
                    'Mark Hunt', 'Stipe Miocic']

    urls = set()
    for fighter in initFighters:
        print('Crawling using %s as root' % fighter)
        url_list = ['http://www.fightmetric.com/fighter-details/c849740a3ff51931']
        while len(url_list) != 0:
            cur_url = url_list.pop()
            if cur_url not in urls:
                urls.add(cur_url)
                print('Adding' + cur_url)
                print('Urls length: %d' % len(urls))
                print('List size: %d' % len(url_list))

                for url in get_fighter_urls(get_page(cur_url)):
                    url_list.append(url)
            else:
                print('skipping')

    with open('urls.txt', 'w') as f:
        for u in list(urls):
            f.write(u)
            f.write('\n')


def get_fight_number_histogram():
    lst = []
    with open('urls.txt', 'r') as f:
        lines = f.readlines()
    urls = [l.strip() for l in lines]
    for i in range(len(urls)):
        print('Fetching for url number %d' % i)
        try:
            fights_url = get_fight_urls(get_page(urls[i]))
            lst.append(len(fights_url))
            if len(fights_url) > 10:
                print(urls[i])
        except Exception:
            continue

    plt.hist(lst, bins=30)
    plt.title("Fight distribution")
    plt.xlabel("Nb of fights")
    plt.ylabel("Frequency")
    plt.savefig('data/fight_distribution.png')

    plt.clf()


def format_total_data(data):
    for k in data.keys():
        if 'sigstr' in k:
            if 'pt' in k:
                data[k] = float(data[k][:-1])/100
            else:
                data[k] = data[k].split()[0]
        if 'td' in k:
            if 'pt' in k:
                data[k] = float(data[k][:-1])/100
            else:
                data[k] = data[k].split()[0]
        if 'tsrt' in k:
            data[k] = data[k].split()[0]
    return data


def format_sig_data(data):
    new_data = {}
    for k in data.keys():
        if 'body' in k:
            num, denum = data[k].split()[0], data[k].split()[2]
            data[k] = num
            try:
                v = float(num)/float(denum)
            except ZeroDivisionError:
                v = 0
            new_data['bodypt' + k[-1]] = round(v,2)
        if 'clinch' in k:
            num, denum = data[k].split()[0], data[k].split()[2]
            data[k] = num
            try:
                v = float(num)/float(denum)
            except ZeroDivisionError:
                v = 0
            new_data['clinchpt' + k[-1]] = round(v,2)
        if 'distance' in k:
            num, denum = data[k].split()[0], data[k].split()[2]
            data[k] = num
            try:
                v = float(num)/float(denum)
            except ZeroDivisionError:
                v = 0
            new_data['distancept' + k[-1]] = round(v,2)
        if 'head' in k:
            num, denum = data[k].split()[0], data[k].split()[2]
            data[k] = num
            try:
                v = float(num)/float(denum)
            except ZeroDivisionError:
                v = 0
            new_data['headpt' + k[-1]] = round(v,2)
        if 'leg' in k:
            num, denum = data[k].split()[0], data[k].split()[2]
            data[k] = num
            try:
                v = float(num)/float(denum)
            except ZeroDivisionError:
                v = 0
            new_data['legpt' + k[-1]] = round(v,2)
        if 'ground' in k:
            num, denum = data[k].split()[0], data[k].split()[2]
            data[k] = num
            try:
                v = float(num)/float(denum)
            except ZeroDivisionError:
                v = 0
            new_data['groundpt' + k[-1]] = round(v,2)

    return {**data, **new_data}


if __name__ == "__main__":
    # pprint(get_fighter_urls(get_page('http://www.fightmetric.com/fighter-details/c849740a3ff51931')))
    # crawl(initFighter=fighter, K=4)
    urls = []
    with open('urls.txt', 'r') as f:
        lines = f.readlines()
    urls = [l.strip() for l in lines]
    stats, url, fights_url = parse_page(get_page(urls[2]))
    pprint(stats)
    # l = list(fights_url)
    # pprint(stats)
    # pprint(parse_fights(get_page(urls[1])))

    # get_fight_number_histogram()
