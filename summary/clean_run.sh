rm ./event*
rm image/events*
rm uncertainty/events*

tensorboard --logdir=./ --host=0:0:0:0:0:0:0:0 --port 8008 
