FROM python:3.6.4-slim-stretch

ADD . /app

WORKDIR /app

RUN python -m pip install -U pip && pip install -r requirements.txt

ENTRYPOINT [""]

CMD ["python","app.py"]

EXPOSE 5000