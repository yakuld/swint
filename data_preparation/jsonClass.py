import json
import os

output_path = '../dataset/DFDC/'

for i in range(1,5):
    f = open('../dataset/DFDC/dfdc_train_part_'+ str(i) +'/metadata.json')
    data=json.load(f)
    count={'REAL':0, 'FAKE':0}
    for k in data:
        count[data[k]['label']] += 1

    fake = 0
    for k in data:
        label = data[k]['label']
        old_path = os.path.join('../dataset/DFDC/dfdc_train_part_'+ str(i) +'/', k)
        new_file_path = os.path.join(output_path, label, k)

        if(not os.path.isfile(old_path)):
            continue
        if(label == 'FAKE'):
            fake += 1
        
        if(label == 'FAKE' and count['REAL'] <= fake):
            continue

        os.rename(old_path, new_file_path)

    f.close()