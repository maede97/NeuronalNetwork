data = open('t10k-images-idx3-ubyte','rb').read()
clear = [str(i) for i in data]

# clear[0:4] = magic number
# clear[4:8] = number of images = 60'000
# clear[8:12] = rows = 28
# clear[12:16] = cols = 28

with open('test_images.txt','w') as wr:
    for i in range(16,len(clear),28*28): 
        wr.write(" ".join(clear[i:i+28*28]))
        wr.write("\n")
