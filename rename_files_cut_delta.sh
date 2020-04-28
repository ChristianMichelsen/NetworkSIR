for file in *.csv
do
  mv "$file" "${file/delta_0.05_/}"
done
