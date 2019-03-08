from bilby_pe import run

num_test_samp=3

for i in range(num_test_samp):
    run(run_label='samp_%d' % i,make_test_samp=True,make_train_samp=False)
