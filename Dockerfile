FROM python:3.8

WORKDIR /app
COPY ./ .

RUN pip install -r requirements.txt
RUN python main.py

EXPOSE 8080

CMD [ "python", "./app.py" ]