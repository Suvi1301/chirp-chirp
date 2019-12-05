FROM python:3
MAINTAINER Suvineet Singh <suvineet1998@gmail.com>

WORKDIR .

RUN apt-get update && apt-get install -y libhdf5-dev

ADD requirements.txt requirements.txt
RUN pip install --requirement requirements.txt

EXPOSE 5000

ADD . .

CMD ["waitress-serve", "--port=5000", "--url-scheme=https", "web.app:APP"]