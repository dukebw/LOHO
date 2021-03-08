samples=('67172.jpg' '02602.jpg' '67172.jpg'
    '00761.jpg' '00761.jpg' '00018.jpg'
    '52364.jpg' '52364.jpg' '19501.jpg'
    '17754.jpg' '17658.jpg' '00148.jpg'
    '46826.jpg' '08244.jpg' '10446.jpg')

array_len=${#samples[@]}

for (( i=0; i<array_len; i=i+3 )) do
	python loho.py --image1 ${samples[$i]} --image2 ${samples[$((i+1))]} --image3 ${samples[$((i+2))]}
done
