def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def dict_to_str_v2(src_dict):
    dst_str = src_dict['weighted avg']['f1-score']
    dst_str = " f1: %.4f " % dst_str
    return dst_str


def read_pickle(file_path):
    import pickle
    with open(file_path,'rb') as f:
        text = pickle.load(f)

    return text

def write_pickle(file_path,data):
    import pickle
    with open(file_path,'wb') as f:
        text = pickle.dump(data, f)

    return text


def read_json_files(file):
    import glob
    import json
    json_files = glob.glob(file)
    gather_eposide_dict = {}

    for file in json_files:
        eposide = json.load(open(file,encoding='utf8'))
        gather_eposide_dict.update(eposide)

    return gather_eposide_dict

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = [[2,3,4,5],[200,200,300]]
    b = [90,100]
    plt.figure()
    for value in a:
        plt.plot(value)
        # plt.plot(b)
        plt.legend('best',labels=['up'])
        plt.show()
        plt.close('all')


