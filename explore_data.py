import h5py

# Open the file as read only
h5f = h5py.File('bat_ifc.hdf5', 'r')['bat_ifc']

# Get the data
for patient in h5f.values():
    for sample in patient.values():
        print(sample)