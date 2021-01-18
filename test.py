i=-1
while True:
    sess.run(iterator.initializer)
    i+=1
    if i%10000==0:
        print(i)
    a=iterator.get_next()
    inputs = a['input_ids']
    v = sess.run(inputs)

if 1:
    iter = iter_data(path_train, tokenizer, max_seq_length, L, idx0, epochs=100,batch_size=64)
    data = next(iter)
    X_input_ids_,X_segment_ids_,X_input_mask_,Ybatch, epoch = data
    feed_dict = {input: X_input_ids_, segment_ids: X_segment_ids_,input_mask: X_input_mask_}
    for i in range(len(L)):
        feed_dict[Y[i]] = Ybatch[i]
    loss_train, acc_train, _ = session.run([Loss, Acc, model_train_op], feed_dict=feed_dict)