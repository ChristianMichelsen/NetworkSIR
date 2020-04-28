for file in *.html
do
  mv "$file" "${file/delta_0.05_/}"
done
