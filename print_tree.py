import h5py

filename_hdf = "./mv_data/Slice2_0Mg_13-9-20_resample_100_channel_data.h5"


def h5_tree(val, pre=""):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                h5_tree(val, pre + "    ")
            else:
                try:
                    print(pre + "└── " + key + " (%d)" % len(val))
                except TypeError:
                    print(pre + "└── " + key + " (scalar)")
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                h5_tree(val, pre + "│   ")
            else:
                try:
                    print(pre + "├── " + key + " (%d)" % len(val))
                except TypeError:
                    print(pre + "├── " + key + " (scalar)")


with h5py.File(filename_hdf, "r") as hf:
    print(hf)
    h5_tree(hf)
