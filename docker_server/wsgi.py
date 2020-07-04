from passport_parser import flask_app as application

if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0', port=7777)

