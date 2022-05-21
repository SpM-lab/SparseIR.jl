FROM julia:1.6.6

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3 \
    python3-dev \
    python3-distutils \
    curl \
    ca-certificates \
    git \
    wget \
    zip \
    && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    vim \
    openssh-server \
    tree \
    && \
    apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* # clean up

RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install \
    jupyter-book \
    jupytext \
    sparse-ir \
    xprec \
    matplotlib \
    ghp-import \
    && \
    echo Done

WORKDIR ${HOME}
USER ${USER}

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN mkdir -p ${HOME}/.julia/config && \
    echo '\
    # set environment variables\n\
    ENV["PYTHON"]=Sys.which("python3")\n\
    ENV["JUPYTER"]=Sys.which("jupyter")\n\
    ' >> ${HOME}/.julia/config/startup.jl && cat ${HOME}/.julia/config/startup.jl



USER ${USER}

WORKDIR /workspace/SparseIR.jl

USER root
RUN chown -R ${NB_UID} /workspace/SparseIR.jl
USER ${USER}


WORKDIR ${HOME}
USER ${USER}
# FIXME
# Install dependencies in a smarter way
RUN julia -e 'import Pkg; Pkg.add(["AssociatedLegendrePolynomials", "Revise", "Plots", "BenchmarkTools"])'
RUN julia -e 'import Pkg; Pkg.add(["DoubleFloats", "IntervalRootFinding", "LinearAlgebra", "QuadGK", "SpecialFunctions", "MultiFloats"])'

EXPOSE 8000

CMD ["julia"]
