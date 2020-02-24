import pandas as pd
from sodapy import Socrata
from tqdm import tqdm
import threading


def collect_all():
    #initialize cliant
    client = Socrata("data.cityofchicago.org", None)

    #get all requests
    #results = [y for x in (client.get("a9u4-3dwb", limit=2000, offset=i*1000, where="complaints IS NOT NULL") for i in range(0,100,2)) for y in x]

    results = []

    for i in tqdm(range(0,100,2)):
        try:
            sub_list = client.get("a9u4-3dwb", limit=2000, offset=i*1000, where="complaints IS NOT NULL")
            if len(sub_list) > 0:
                results.extend(sub_list)
            #returns empty list when offset > database size
            else:
                break
        except:
            print('err')

    #this is based on no rate limit errors,
    #at this sclae none have been encoutered
    #thus this will suffice for current use

    #save to df and print number of  entries
    results_df = pd.DataFrame.from_records(results)
    print(f'number of complaints found:  {results_df.shape[0]}')

    #save this to our pickle file for analysis
    f = './reports.pkl'
    results_df.to_pickle(f)
    print(f'Initial Reports DataFrame has been saved to {f}')
    return results_df

#thread class to speed up complaint download time
class myThread (threading.Thread):
    def __init__(self, threadID, name, counter, complaint_df):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.complaint_df = complaint_df
    def run(self):
        print(f"Starting {self.name}\n")
        self.complaint_df = worker(self.threadID, self.complaint_df)
        global complaints
        lock.acquire()
        #print(f'Size of df: {self.complaint_df.shape[0]}')
        complaints = complaints.append(self.complaint_df)
        lock.release()
        print (f"Exiting {self.name}\n")



def worker(start,complaint_df):
    for row in range(start,df.shape[0],8):
        if row % 100 == 0 :
            print(f'Completed {row} downloads',end='\r')
        if complaint_df is None:   
            complaint_df = pd.read_csv(df.complaints[row]['url'])
            complaint_df['complaints'] = pd.Series(dict)
            for j in range(complaint_df.shape[0]):
                complaint_df.complaints[j] = df.complaints[row]
        else:
            try:
                temp = pd.read_csv(df.complaints[row]['url'])
                temp['complaints'] = pd.Series(dict)
                for j in range(temp.shape[0]):
                    temp.complaints[j] = df.complaints[row]
                
                complaint_df = complaint_df.append(temp)
            except:
                 print(f'error on complaint: {row}')
    return complaint_df


#now we need to collect the complaints of each report to collect the actual text data
def collect_reports():
    print('Collecting reports\n')
    
        #thread initializing
    c1,c2,c3,c4,c5,c6,c7,c8 = [None for i in range(8)]
    thread1 = myThread(1, "Thread-1", 1, c1)
    thread2 = myThread(2, "Thread-2", 2, c2)
    thread3 = myThread(3, "Thread-3", 3, c3)
    thread4 = myThread(4, "Thread-4", 4, c4)
    thread5 = myThread(5, "Thread-5", 5, c5)
    thread6 = myThread(6, "Thread-6", 6, c6)
    thread7 = myThread(7, "Thread-7", 7, c7)
    thread8 = myThread(8, "Thread-8", 8, c8)

    #run threads
    threads = [thread1,thread2,thread3,thread4,thread5,thread6,thread7,thread8]
    for thread in threads:
        thread.start()
    #sync before saving to pkl
    for thread in threads:
        thread.join()

    f = './complaints_v2.pkl'
    complaints[1:].to_pickle(f)
    print(f'Complaints DataFrame has been saved to {f}')


if __name__=='__main__':
    
    df = collect_all()

    #initialize a lock for race conditions because of threading
    #this is not a bottleneck because we use it only once per thread
    lock = threading.Lock()
    
    #here we are just initiating our df with the appropriate datatypes
    complaints =  pd.read_csv(df.complaints[0]['url'])[0:0]
    #this is specifically for joining
    complaints['complaints'] = pd.Series(dict)

    
    collect_reports()
    
