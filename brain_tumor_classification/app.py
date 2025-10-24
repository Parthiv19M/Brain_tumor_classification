from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    team_members = [
        {"name": "Parthiv Sarma Meduri", "role": "Team Lead", "email": "parthiv.meduri@example.com"},
        {"name": "Nihitha Meduri", "role": "Research Lead", "email": "nihitha.meduri@example.com"},
        {"name": "Sai Pavani", "role": "Data Scientist", "email": "sai.pavani@example.com"},
        {"name": "Kavya", "role": "ML Engineer", "email": "kavya@example.com"}
    ]
    return render_template('index.html', team_members=team_members)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
