FROM smizy/keras
ADD . /app
WORKDIR /app
EXPOSE 3000
RUN apk add --update python3 python3-dev gfortran py-pip build-base g++ gfortran file binutils \
                     musl-dev openblas-dev libstdc++ openblas libpng-dev freetype-dev
RUN pip3 install numpy
RUN pip3 install np
RUN pip3 install sklearn
RUN pip3 install nltk
RUN python3 setup.py
RUN pip3 install matplotlib
RUN pip3 install gensim

CMD ["ash"]
