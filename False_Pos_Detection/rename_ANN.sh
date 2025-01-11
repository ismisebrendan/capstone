path=$1

files=$(find $path -type f -name "*.keras")
echo $files

for file in $files; do
	end="${file:0-7}"
	new_name="${path}/${path}_ANN_${end}"
	echo $new_name
	mv $file $new_name	
done
