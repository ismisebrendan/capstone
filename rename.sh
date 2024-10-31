path=$1

files=$(find $path -type f -name "*.txt")


size_path=${#path}

len_file=$((2*size_path+7))

for file in $files; do
	if [ "${#file}" -eq $len_file ]; then
		end="${file:0-5}"
		new_name="${path}/${path}_0${end}"
		mv $file $new_name
	fi
done
