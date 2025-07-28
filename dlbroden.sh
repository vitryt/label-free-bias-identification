echo "Downloading broden1_224"
mkdir -p concept_dataset
pushd concept_dataset
wget --progress=bar \
   http://netdissect.csail.mit.edu/data/broden1_224.zip \
   -O broden1_224.zip
unzip broden1_224.zip
rm broden1_224.zip
popdt