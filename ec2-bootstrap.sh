wget http://nlp.stanford.edu/data/glove.6B.zip
wget https://zenodo.org/record/1199620/files/SO_vectors_200.bin?download=1
mv 'SO_vectors_200.bin?download=1' SO_vectors_200.bin
mkdir embeddings
mv SO_vectors_200.bin embeddings
mv glove.6B.zip embeddings
cd embeddings
unzip glove.6B.zip
cd ../
python3 setup.py
pip3 install np tensorflow bibtexparser keras gensim
