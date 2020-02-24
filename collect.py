import pandas as pd
from sodapy import Socrata
from tqdm import tqdm

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
f = './complaints1.pkl'
results_df.to_pickle(f)
print(f'Complaints DataFrame has been saved to {f}')



#now we need to collect the complaints of each report to collect the actual text data


