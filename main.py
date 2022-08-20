from apps import create_app
import os
app = create_app()

if __name__ == "__main__":
    #port = int(os.environ.get('PORT', 80))
    app.run(host='127.0.0.1', port=5050, debug=True)
    #app.run(debug=True)
    