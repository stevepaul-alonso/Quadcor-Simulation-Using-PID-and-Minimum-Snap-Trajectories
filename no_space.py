import csv

input = open('output.csv','r')
output = open('output_2.csv', 'w')

writer = csv.writer(output, lineterminator = '\n')

for row in csv.reader(input):
	#print(row)
	if len(row) ==1:
		writer.writerow(row)
input.close()
output.close()