import csv
with open('fordima.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    # row[0] : startime, row[1]:endtime, row[2]:url
    isFirst = True
    currentURL = ''
    with open('dataprovider.js', 'w') as writefile:
        writefile.write('var source= [');
        for row in readCSV:
            if isFirst:
                isFirst = False
            else:
                writefile.write(',')

            starttime = int(float(row[0]) * 1000)
            endtime = int(float(row[1]) * 1000)
            writefile.write('{starttime:'+str(starttime)+', endtime:' + str(endtime) + ', duration:'+str(endtime-starttime)+', url:"'+row[2].strip()+'"}\n')
        writefile.write('];');