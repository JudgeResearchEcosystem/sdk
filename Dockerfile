# Roughly following this blog recommended by the the Jupyter Docker docs:
# https://herrmann.tech/en/blog/2021/02/08/how-to-build-custom-environment-for-jupyter-in-docker.html

FROM jupyter/datascience-notebook:2022-04-19

# set locales, en_US.UTF-8 should already be a default but leaving this as a template for future requrirements
#RUN set -ex \
#   && sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
#   && locale-gen en_US.UTF-8 \
#   && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \

# install Python packages you often use
RUN set -ex \
   # FIXME: replace --quiet below after pinning versions
   && conda install --yes \
   # choose the Python packages you need
   'plotly==5.7.0' \
   'folium==0.12.1.post1' \
   'rise==5.7.1' \
   'TA-Lib==0.4.19' \
   'pandas-ta==0.3.14b' \
   'pandas-datareader==0.10.0' \
   'yfinance==0.1.70' \
   'alpha_vantage==2.3.1' \
   'jupyterthemes==0.20.0' \
   && conda clean --all -f -y \
   # install Jupyter Lab extensions you need
   && jupyter labextension install jupyterlab-plotly --no-build \
   && jupyter lab build -y \
   && jupyter lab clean -y \
   && rm -rf "/home/${NB_USER}/.cache/yarn" \
   && rm -rf "/home/${NB_USER}/.node-gyp" \
   && fix-permissions "${CONDA_DIR}" \
   && fix-permissions "/home/${NB_USER}" \
   # install coinbase SDK
   && git clone https://github.com/resy/coinbase_python3.git \
   && cd coinbase_python3 \
   && python3 setup.py install


# Additions below per MyBinder Dockerfile best practices #3 & #4:
# https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# The below is directly from the MyBinder docs; however, it's
# commented out because the command fails. The user jovyan already exists with correct UID
# leaving this in case the upstream build changes in the future

#USER root
#RUN adduser --disabled-password \
#    --gecos "Default user" \
#    --uid ${NB_UID} \
#    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID}:users ${HOME}
USER ${NB_USER}

# Import the workspace with preconfigured tabs
RUN jupyter lab workspaces import JudgeResearchNotebooks/default_workspaces.json

# Clean up unnecessary & confusing files
RUN rm -rf ${HOME}/coinbase_python3
RUN rm -f ${HOME}/README.md
RUN rm -f ${HOME}/docker-compose.yaml
RUN rm -f ${HOME}/Dockerfile
