SAVEDIR="models"
DROPOUT=0.1
TRAIN=0.5
LR=0.0001
EPOCHS=10
DECAY=0.003
ILLUM="none"
DEVNUM=1

#for nfilters in {2..10..2}
for nfilters in 4
do
	echo "Training with $nfilters filters"
	python learn_materialnet.py --experiment $1\
							  	--nfilters $nfilters\
								--dropout $DROPOUT\
								--savedir $SAVEDIR\
								--train $TRAIN\
								--learning_rate $LR\
								--epochs $EPOCHS\
								--decay $DECAY\
								--illuminant $ILLUM
done
