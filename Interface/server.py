from flask import Flask, render_template, request, url_for, Markup
import sqlite3
from sqlite3 import Error
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

dict =  {
    1: "Autobiographical, Paths to Prison",
    2: "Education, Re-entry, Other Programs",
    3: "Family",
    4: "Health Care",
    5: "Judicial Misconduct and Legal Remediation",
    6: "Personal/Internal Change/Coping",
    7: "Physical Conditions and Security",
    8: "Political and Intellectual Labor among IP",
    9: "Prison Culture/Community/Society",
    10: "Prison Industry/Prison as Business",
    11: "Social Alienation, Indifference, Hostility",
    12: "Staff/prison Abuse of IP"}

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def select_essays_with_topic(conn, topic):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute('SELECT link, title FROM essays WHERE "{}" > 0 ORDER BY "{}" DESC'.format(topic, topic))

    d = {}
    count = 0
    rows = cur.fetchall()

    print("Total: ", len(rows))
    for row in rows:
        d[count] = (list(row)[0], list(row)[1])
        count += 1
    #   print(row)
    #print(list(rows[0]))
    return d

def makeHtml(query_results):
    #returns a list of hyperlinks in html format

    html = """
<html>
<body>
    <ul style="list-style-type:none;">
        """

    for link in query_results.values():
        html += '<li><a href="' + link[0] + '" target="_blank">' + link[1] + '</a></li>'

    html += '</ul></body></html>'
    return html

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/get-topic/', methods=['GET', 'POST'])
def get_topic():
    database = r"db.sqlite3"
    topic = dict[int(request.form.get("myDropdown"))]
    print("=======Topic is:======= ", topic)

    # create a database connection
    conn = create_connection(database)
    with conn:
      #print("Query essays of topic {}." % topic)
      query_results = select_essays_with_topic(conn, topic)
      results = Markup(makeHtml(query_results))
      return render_template('results.html', results=results, topic=topic)


if __name__ == '__main__':
  app.run(debug=True)
