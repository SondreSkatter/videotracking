###########
# BUILDER #
###########
FROM httpd:2.4 AS builder

# install build dependencies
RUN apt-get update && apt-get install --yes python3 python3-pip entr
RUN python3 -m pip install --no-cache-dir jinja2 PyYAML

# configure httpd to serve from /build/_render
RUN sed -i \
    -e 's|DocumentRoot "/usr/local/apache2/htdocs"|DocumentRoot "/build/_render"|' \
    -e 's|<Directory "/usr/local/apache2/htdocs">|<Directory "/build/_render">|' \
    conf/httpd.conf

COPY . /build
# RUN ls -s /usr/local/apache2/htdocs/ /build/_render

# write build configuration
ARG api_uri
RUN echo "api_uri: $api_uri" > /etc/config.yml

# do a build of the app (needed for chaining into prod stage)
RUN cd /build && ./scripts/render /etc/config.yml
RUN ls /build/_render

# when container starts, watch and rebuild in parallel to running httpd
CMD cd /build && mkdir -p _render && ./scripts/watch /etc/config.yml & httpd-foreground

########
# PROD #
########
FROM httpd:2.4 AS prod
COPY --from=builder /build/_render/* /usr/local/apache2/htdocs/