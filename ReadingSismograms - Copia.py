from obspy import read

st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')

print(st)

len(st)

tr = st[0]
print(tr)

print(tr.data[0:3])

print(len(tr))

st.plot()

