from bilby_pe import run

num_test_samp=9

for i in range(num_test_samp):
    run(run_label='samp_%d' % i,make_test_samp=True,make_train_samp=False,duration=1.,sampling_frequency=512.)

