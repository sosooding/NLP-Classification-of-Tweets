def combine(f1, f2):
	f = open("combined.txt", "w")

	tmpf1 = open(f1, "r")
	for i in tmpf1.read():
		f.write(i)
	tmpf1.close()

	tmpf2 = open(f2, "r")
	for i in tmpf2.read():
		f.write(i)
	tmpf2.close()

	print("Combined contents of %s and %s into 'combined.txt'"%(f1, f2))

combine("SOLO-loneliness-tweets.txt", "SOLO-lonely-tweets.txt")