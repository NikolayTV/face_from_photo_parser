# pull official base image
FROM python:3.6

COPY . /usr/src/passport_photo_parser

# set work directory
WORKDIR /usr/src/passport_photo_parser


# Update pip and from requirements.txt install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 7777

# RUN chmod +x boot.sh
CMD gunicorn -c gunicorn.conf.py wsgi

