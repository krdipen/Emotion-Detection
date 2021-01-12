if [[ $1 == "part1" ]];
then
    python part1.py train.csv public_test.csv
elif [[ $1 == "part2" ]];
then
    python part2.py train.csv public_test.csv private.csv submission.csv
elif [[ $1 == "clean" ]];
then
	rm -rf submission.csv
else
    echo "Invalid Command"
fi
