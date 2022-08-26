import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Integer,VARCHAR,DATE,Float,BIGINT,TIME
def data_loading(file_name,type='replace',disp=False):
    with open(file_name,'r') as file:
        f=file.readlines()

        final_data=[]
        for index,data in enumerate(f):
            print(data)
            if index==2:
                line_data=data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                batch=filtering_data[-1]
            elif index==6:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                Reactor = filtering_data[-1]
            elif index == 7:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                position = filtering_data[-1]
            elif index == 8:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                end_date = filtering_data[-1]
            elif index == 9:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                end_time = filtering_data[-1]
            elif index == 10:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                end_duration = filtering_data[-1]
            elif index == 12:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                plate_type = filtering_data[-1]
            elif index>19:
                line_data = data.split(',')
                filtering_data = [x for x in line_data if x.strip()]
                row_values=[batch, Reactor, position, end_date,end_time, end_duration, plate_type]+filtering_data
                final_data.append(row_values)
        data_frame=pd.DataFrame(final_data,columns=['batch','reactor','position','enddate','endtime','duration','plate_type','plate','hole','sample_id', 'material','not_needed1','not_needed2','not_needed3','weight','exp_type','project','owner'])
        clean_data = data_frame.drop(
            columns=['not_needed1', 'not_needed2', 'not_needed3'])
        clean_data['enddate'] = pd.to_datetime(clean_data['enddate'])
        clean_data['endtime'] = pd.to_datetime(clean_data['endtime'])
        clean_data['endtime'] = [time.time() for time in clean_data['endtime']]
        clean_data['plate'] = clean_data['plate'].astype(int)
        clean_data['hole'] = clean_data['hole'].astype(int)
        clean_data['owner']=clean_data['owner'].replace('\n','',regex=True)
        if disp:
            print(clean_data)
        softurl = r"mariadb+mariadbconnector://root:dbpwx61@139.20.22.156:3306/ararsoftware"
        soft_engine = create_engine(softurl)
        software_conn = soft_engine.connect()
        clean_data.to_sql('ararirrdata', software_conn,if_exists=type, index=True
                          , dtype={'batch':VARCHAR(40), 'reactor':VARCHAR(50),
                                   'position': VARCHAR(40), 'enddate': DATE(), 'endtime': TIME(), 'duration': Float(3),
                            'plate_type':VARCHAR(40), 'plate':Integer(), 'hole':Integer(),
                                   'sample_id': VARCHAR(40),
                                   'material': VARCHAR(40), 'weight': Float(), 'exp_type': VARCHAR(40), 'project': VARCHAR(50), 'owner': VARCHAR(40)})
    print(file_name+' file data loaded into database')
    pass


if __name__ == "__main__":
    root_dir='.\data'
    onlyfiles = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                 os.path.isfile(os.path.join(root_dir, f))]
    for file in onlyfiles:
        file_ext = os.path.splitext(file)[-1].lower()
        if file_ext!='.irr':
            print('File format is not supported')
        else:
            data_loading(file, type='replace',disp=False)
