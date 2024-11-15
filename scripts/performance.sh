total_time=0
count=10
url="http://localhost:8080/json" 

for i in $(seq 1 $count); do
    time=$(curl -o /dev/null -s -w '%{time_total}\n' "$url" -H 'Content-Type: application/json' -d @./scripts/request_params)
    total_time=$(echo "$total_time + $time" | bc)
done

average_time=$(echo "scale=3; ($total_time / $count) * 1000" | bc)
echo "Average time: $average_time ms"