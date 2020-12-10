i=-1
while True:
    sess.run(iterator.initializer)
    i+=1
    if i%10000==0:
        print(i)
    a=iterator.get_next()
    inputs = a['input_ids']
    v = sess.run(inputs)
