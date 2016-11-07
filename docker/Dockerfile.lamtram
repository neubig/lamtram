FROM lamtram-deps:latest

# Lamtram (this version)
COPY . /opt/lamtram
RUN cd /opt/lamtram && \
        export LDFLAGS="-L/usr/local/cuda/lib64" && \
        autoreconf -i && \
        ./configure --with-dynet=/opt/dynet --with-eigen=/opt/eigen --with-cuda=/usr/local/cuda && \
        make -j16 install
RUN cp /opt/lamtram/script/* /usr/local/bin
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}
WORKDIR /work
