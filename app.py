"""
Flask app entry point for IT Law Chatbot.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, send_from_directory

from flask_cors import CORS
from config import Config
from routes.chat import chat_bp


def create_app():
    app = Flask(__name__, static_folder="static", static_url_path="")
    CORS(app)

    # Register blueprints
    app.register_blueprint(chat_bp)

    # Serve frontend
    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"\n{'='*60}")
    print(f"  IT Law Chatbot - Tư vấn Luật Công nghệ thông tin")
    print(f"  Server running at http://localhost:{Config.FLASK_PORT}")
    print(f"{'='*60}\n")
    app.run(
        host="0.0.0.0",
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG,
    )
