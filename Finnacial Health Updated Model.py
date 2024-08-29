from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///financial_health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    financial_data = db.relationship('FinancialData', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Financial data model
class FinancialData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    income = db.Column(db.Float, nullable=False)
    expenses = db.Column(db.Float, nullable=False)
    debts = db.Column(db.Float, nullable=False)
    investments = db.Column(db.Float, nullable=False)
    savings_rate = db.Column(db.Float, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load or train the ML model
def get_ml_model():
    model_path = 'financial_health_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Generate more realistic simulated data
        np.random.seed(42)
        n_samples = 1000
        data = {
            'income': np.random.normal(5000, 1500, n_samples),
            'expenses': np.random.normal(3000, 1000, n_samples),
            'debts': np.random.normal(10000, 5000, n_samples),
            'investments': np.random.normal(20000, 10000, n_samples),
            'savings_rate': np.random.normal(20, 10, n_samples)
        }
        df = pd.DataFrame(data)
        df['savings_rate'] = (df['income'] - df['expenses']) / df['income'] * 100
        df['savings_rate'] = df['savings_rate'].clip(0, 100)

        X = df[['income', 'expenses', 'debts', 'investments']]
        y = df['savings_rate']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        joblib.dump((model, scaler), model_path)
        return model, scaler

ml_model, scaler = get_ml_model()

# Function to calculate financial health metrics
def calculate_financial_health(income, expenses, debts, investments):
    savings_rate = ((income - expenses) / income) * 100 if income > 0 else 0
    debt_to_income_ratio = (debts / (income * 12)) * 100 if income > 0 else 0
    investment_to_income_ratio = (investments / (income * 12)) * 100 if income > 0 else 0
    return savings_rate, debt_to_income_ratio, investment_to_income_ratio

# Function to predict savings rate using the trained model
def predict_savings_rate(income, expenses, debts, investments):
    input_data = np.array([[income, expenses, debts, investments]])
    input_data_scaled = scaler.transform(input_data)
    prediction = ml_model.predict(input_data_scaled)
    return prediction[0]

# Function to get historical financial data
def get_historical_data(user_id):
    user_data = FinancialData.query.filter_by(user_id=user_id).order_by(FinancialData.date).all()
    
    if not user_data:
        return None

    data = {
        'dates': [data.date.strftime('%Y-%m-%d') for data in user_data],
        'incomes': [data.income for data in user_data],
        'expenses': [data.expenses for data in user_data],
        'debts': [data.debts for data in user_data],
        'investments': [data.investments for data in user_data],
        'savings_rates': [data.savings_rate for data in user_data]
    }

    return data

# Routes
@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Health Assessment</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto mt-8 text-center">
            <h1 class="text-4xl font-bold mb-8">Welcome to Financial Health Assessment</h1>
            <div class="space-x-4">
                <a href="{{ url_for('login') }}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Login</a>
                <a href="{{ url_for('register') }}" class="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">Register</a>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Register - Financial Health Assessment</title>
                <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto mt-8">
                    <h1 class="text-3xl font-bold mb-4">Register</h1>
                    <p class="text-red-500 mb-4">Username already exists</p>
                    <a href="{{ url_for('register') }}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Try Again</a>
                </div>
            </body>
            </html>
            ''')

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Register - Financial Health Assessment</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="bg-gray-100">
            <div class="container mx-auto mt-8">
                <h1 class="text-3xl font-bold mb-4">Register</h1>
                <p class="text-green-500 mb-4">Registration successful. Please log in.</p>
                <a href="{{ url_for('login') }}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Login</a>
            </div>
        </body>
        </html>
        ''')

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Register - Financial Health Assessment</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto mt-8">
            <h1 class="text-3xl font-bold mb-4">Register</h1>
            <form method="post" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
                <div class="mb-4">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="username">
                        Username
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="username" name="username" type="text" placeholder="Username" required>
                </div>
                <div class="mb-6">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="password">
                        Password
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline" id="password" name="password" type="password" placeholder="******" required>
                </div>
                <div class="flex items-center justify-between">
                    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" type="submit">
                        Register
                    </button>
                    <a class="inline-block align-baseline font-bold text-sm text-blue-500 hover:text-blue-800" href="{{ url_for('login') }}">
                        Already have an account? Login
                    </a>
                </div>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return render_template_string('''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Login - Financial Health Assessment</title>
                <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            </head>
            <body class="bg-gray-100">
                <div class="container mx-auto mt-8">
                    <h1 class="text-3xl font-bold mb-4">Login</h1>
                    <p class="text-red-500 mb-4">Invalid username or password</p>
                    <a href="{{ url_for('login') }}" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Try Again</a>
                </div>
            </body>
            </html>
            ''')

    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Login - Financial Health Assessment</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto mt-8">
            <h1 class="text-3xl font-bold mb-4">Login</h1>
            <form method="post" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
                <div class="mb-4">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="username">
                        Username
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="username" name="username" type="text" placeholder="Username" required>
                </div>
                <div class="mb-6">
                    <label class="block text-gray-700 text-sm font-bold mb-2" for="password">
                        Password
                    </label>
                    <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline" id="password" name="password" type="password" placeholder="******" required>
                </div>
                <div class="flex items-center justify-between">
                    <button class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline" type="submit">
                        Sign In
                    </button>
                    <a class="inline-block align-baseline font-bold text-sm text-blue-500 hover:text-blue-800" href="{{ url_for('register') }}">
                        Don't have an account? Register
                    </a>
                </div>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard - Financial Health Assessment</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body class="bg-gray-100">
        <nav class="bg-blue-600 p-4 text-white">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-2xl font-bold">Financial Health Dashboard</h1>
                <a href="{{ url_for('logout') }}" class="bg-red-500 hover:bg-red-600 px-4 py-2 rounded">Logout</a>
            </div>
        </nav>

        <div class="container mx-auto mt-8">
            <div class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
                <h2 class="text-xl font-bold mb-4">Add Financial Data</h2>
                <form id="financialDataForm" class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-gray-700 text-sm font-bold mb-2" for="income">
                            Monthly Income
                        </label>
                        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="income" type="number" step="0.01" required>
                    </div>
                    <div>
                        <label class="block text-gray-700 text-sm font-bold mb-2" for="expenses">
                            Monthly Expenses
                        </label>
                        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="expenses" type="number" step="0.01" required>
                    </div>
                    <div>
                        <label class="block text-gray-700 text-sm font-bold mb-2" for="debts">
                            Total Debts
                        </label>
                        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="debts" type="number" step="0.01" required>
                    </div>
                    <div>
                        <label class="block text-gray-700 text-sm font-bold mb-2" for="investments">
                            Total Investments
                        </label>
                        <input class="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline" id="investments" type="number" step="0.01" required>
                    </div>
                    <div class="col-span-2">
                        <button type="submit" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline">
                            Submit
                        </button>
                    </div>
                </form>
            </div>

            <div id="financialHealthResults" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4 hidden">
                <h2 class="text-xl font-bold mb-4">Financial Health Assessment</h2>
                <div id="results" class="grid grid-cols-2 gap-4"></div>
            </div>

            <div id="graphs" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4 hidden">
                <h2 class="text-xl font-bold mb-4">Financial Trends</h2>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <canvas id="incomeExpenseChart"></canvas>
                    </div>
                    <div>
                        <canvas id="savingsRateChart"></canvas>
                    </div>
                    <div>
                        <canvas id="debtInvestmentChart"></canvas>
                    </div>
                </div>
            </div>

            <div id="historicalData" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4 hidden">
                <h2 class="text-xl font-bold mb-4">Historical Financial Data</h2>
                <table id="dataTable" class="w-full">
                    <thead>
                        <tr>
                            <th class="px-4 py-2">Date</th>
                            <th class="px-4 py-2">Income</th>
                            <th class="px-4 py-2">Expenses</th>
                            <th class="px-4 py-2">Debts</th>
                            <th class="px-4 py-2">Investments</th>
                            <th class="px-4 py-2">Savings Rate</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>

        <script>
            $(document).ready(function() {
                $('#financialDataForm').on('submit', function(e) {
                    e.preventDefault();
                    $.ajax({
                        url: '/add_financial_data',
                        method: 'POST',
                        data: {
                            income: $('#income').val(),
                            expenses: $('#expenses').val(),
                            debts: $('#debts').val(),
                            investments: $('#investments').val()
                        },
                        success: function(response) {
                            if (response.success) {
                                alert('Financial data added successfully!');
                                updateFinancialHealth();
                            }
                        }
                    });
                });

                function updateFinancialHealth() {
                    $.ajax({
                        url: '/get_financial_health',
                        method: 'GET',
                        success: function(response) {
                            $('#financialHealthResults').removeClass('hidden');
                            $('#graphs').removeClass('hidden');
                            $('#historicalData').removeClass('hidden');
                            $('#results').html(`
                                <p><strong>Savings Rate:</strong> ${response.savings_rate.toFixed(2)}%</p>
                                <p><strong>Debt-to-Income Ratio:</strong> ${response.debt_to_income_ratio.toFixed(2)}%</p>
                                <p><strong>Investment-to-Income Ratio:</strong> ${response.investment_to_income_ratio.toFixed(2)}%</p>
                                <p><strong>Predicted Savings Rate:</strong> ${response.predicted_savings_rate.toFixed(2)}%</p>
                            `);

                            updateCharts(response.historical_data);
                            updateHistoricalDataTable(response.historical_data);
                        }
                    });
                }

                function updateCharts(data) {
                    new Chart(document.getElementById('incomeExpenseChart'), {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [
                                {
                                    label: 'Income',
                                    data: data.incomes,
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1
                                },
                                {
                                    label: 'Expenses',
                                    data: data.expenses,
                                    borderColor: 'rgb(255, 99, 132)',
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            title: {
                                display: true,
                                text: 'Income vs Expenses Over Time'
                            }
                        }
                    });

                    new Chart(document.getElementById('savingsRateChart'), {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [{
                                label: 'Savings Rate',
                                data: data.savings_rates,
                                borderColor: 'rgb(54, 162, 235)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            title: {
                                display: true,
                                text: 'Savings Rate Over Time'
                            }
                        }
                    });

                    new Chart(document.getElementById('debtInvestmentChart'), {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [
                                {
                                    label: 'Debts',
                                    data: data.debts,
                                    borderColor: 'rgb(255, 159, 64)',
                                    tension: 0.1
                                },
                                {
                                    label: 'Investments',
                                    data: data.investments,
                                    borderColor: 'rgb(153, 102, 255)',
                                    tension: 0.1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            title: {
                                display: true,
                                text: 'Debts vs Investments Over Time'
                            }
                        }
                    });
                }

                function updateHistoricalDataTable(data) {
                    let tableBody = $('#dataTable tbody');
                    tableBody.empty();
                    for (let i = 0; i < data.dates.length; i++) {
                        tableBody.append(`
                            <tr>
                                <td class="border px-4 py-2">${data.dates[i]}</td>
                                <td class="border px-4 py-2">$${data.incomes[i].toFixed(2)}</td>
                                <td class="border px-4 py-2">$${data.expenses[i].toFixed(2)}</td>
                                <td class="border px-4 py-2">$${data.debts[i].toFixed(2)}</td>
                                <td class="border px-4 py-2">$${data.investments[i].toFixed(2)}</td>
                                <td class="border px-4 py-2">${data.savings_rates[i].toFixed(2)}%</td>
                            </tr>
                        `);
                    }
                }

                updateFinancialHealth();
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/add_financial_data', methods=['POST'])
@login_required
def add_financial_data():
    income = float(request.form.get('income'))
    expenses = float(request.form.get('expenses'))
    debts = float(request.form.get('debts'))
    investments = float(request.form.get('investments'))

    savings_rate, _, _ = calculate_financial_health(income, expenses, debts, investments)

    new_data = FinancialData(
        user_id=current_user.id,
        income=income,
        expenses=expenses,
        debts=debts,
        investments=investments,
        savings_rate=savings_rate
    )

    db.session.add(new_data)
    db.session.commit()

    return jsonify({'success': True})

@app.route('/get_financial_health', methods=['GET'])
@login_required
def get_financial_health():
    latest_data = FinancialData.query.filter_by(user_id=current_user.id).order_by(FinancialData.date.desc()).first()

    if not latest_data:
        return jsonify({'error': 'No financial data available'})

    savings_rate, debt_to_income_ratio, investment_to_income_ratio = calculate_financial_health(
        latest_data.income, latest_data.expenses, latest_data.debts, latest_data.investments
    )

    predicted_savings_rate = predict_savings_rate(
        latest_data.income, latest_data.expenses, latest_data.debts, latest_data.investments
    )

    historical_data = get_historical_data(current_user.id)

    return jsonify({
        'savings_rate': savings_rate,
        'debt_to_income_ratio': debt_to_income_ratio,
        'investment_to_income_ratio': investment_to_income_ratio,
        'predicted_savings_rate': predicted_savings_rate,
        'historical_data': historical_data
    })

if __name__ == '__main__':
    app.run(debug=True)