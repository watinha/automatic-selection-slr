wget http://nlp.stanford.edu/data/glove.6B.zip
wget https://zenodo.org/record/1199620/files/SO_vectors_200.bin?download=1
mv 'SO_vectors_200.bin?download=1' SO_vectors_200.bin
git clone https://github.com/watinha/automatic-selection-slr
mkdir automatic-selection-slr/embeddings
mv SO_vectors_200.bin automatic-selection-slr/embeddings
mv glove.6B.zip automatic-selection-slr/embeddings
cd automatic-selection-slr/embeddings
unzip glove.6B.zip
cd ../
python3 setup.py
pip3 install np tensorflow bibtexparser keras gensim
