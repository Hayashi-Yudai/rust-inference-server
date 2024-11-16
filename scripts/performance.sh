URL="http://localhost:8080/predict"
NUM_REQUESTS=1000

SEXES=("male" "female")
EMBARKEDS=("C" "Q" "S")

TOTAL_TIME=0.0
for i in `seq 1 $NUM_REQUESTS`;
do
    PCLASS=$((RANDOM % 3 + 1))
    AGE=$((RANDOM % 100))
    SIBBP=$((RANDOM % 8))
    PARCH=$((RANDOM % 6))

    SEX=${SEXES[$((RANDOM % 2))]}
    EMBARKED=${EMBARKEDS[$((RANDOM % 3))]}

    INPUT_JSON=`cat <<EOF
    {
        "pclass": $PCLASS, 
        "age": $AGE, 
        "sibsp": $SIBBP, 
        "parch": $PARCH, 
        "sex": "$SEX", 
        "embarked": "$EMBARKED"
    }
EOF
`
    # echo $INPUT_JSON

    TIME=`curl -w "%{time_total}\n" -o /dev/null -X POST $URL -H 'Content-Type: application/json' -d "$INPUT_JSON"`
    TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
done

TOTAL_TIME=$(echo "$TOTAL_TIME * 1000" | bc)

echo "scale=3; $TOTAL_TIME / $NUM_REQUESTS" | bc  # average time in ms