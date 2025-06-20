cd /home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/RandomlySelectedPseudo3x/images

# Rename all .jpg to .png
for f in *.jpg; do
    mv "$f" "${f%.jpg}.png"
done

# Rename all .jpeg to .png
for f in *.jpeg; do
    mv "$f" "${f%.jpeg}.png"
done
